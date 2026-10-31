//! CPU reference for the WGSL `trace()`: the same walk over the same packed [`Scene`], in the
//! same f32 operations in the same order, so tests can compare the two Voxel for Voxel.
//! `shaders/trace.wgsl` is the one that ships; keep them in step.
//!
//! A scene is walked level by level, the finest first, and the nearest hit of all is the one.
//! Level `n` has Voxels `2^n` times the size of level 0's, and lists only the bricks that the
//! LOD bands select for it (see [`crate::lod`]), so that every part of space is seen at one level
//! only: finer near the camera and coarser far from it. A level is walked inside the box of
//! regions it lists (see [`Level`](crate::scene::Level)), and not in its hole, the regions about
//! the camera that none of its bricks reach. Distances along the ray are in Voxels of level 0 at
//! every level.

use std::cell::Cell;

use crate::scene::{REGION_WORDS, SOLID_FLAG, Scene};

/// What one [`trace_counting`] did, to compare ways of walking a scene: each is a step of one
/// level of the walk, so the shader's cost is about their weighted sum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Steps {
    /// Levels that were looked at, and of those, the ones whose regions were walked at all (the
    /// others were clipped away).
    pub examined: u32,
    pub levels: u32,
    /// Regions stepped through, listed or not.
    pub regions: u32,
    /// Slots of the region table looked at to find them.
    pub probes: u32,
    /// Bricks stepped through in regions that have some.
    pub bricks: u32,
    /// Voxels stepped through in mixed bricks.
    pub voxels: u32,
}

impl Steps {
    /// Region, brick and Voxel steps.
    pub fn total(&self) -> u32 {
        self.regions + self.bricks + self.voxels
    }
}

thread_local! {
    /// The steps so far of the [`trace_counting`] that is running, if one is.
    static STEPS: Cell<Option<Steps>> = const { Cell::new(None) };
}

fn count(add: impl FnOnce(&mut Steps)) {
    STEPS.with(|cell| {
        if let Some(mut steps) = cell.get() {
            add(&mut steps);
            cell.set(Some(steps));
        }
    });
}

/// [`trace`], and the steps it took.
pub fn trace_counting(scene: &Scene, ray: &Ray) -> (Option<Hit>, Steps) {
    STEPS.with(|cell| cell.set(Some(Steps::default())));
    let hit = trace(scene, ray);
    (hit, STEPS.with(|cell| cell.take()).unwrap_or_default())
}

/// A ray: it starts at `cell` plus `origin`, and goes along the unit vector `dir`. Splitting the
/// start keeps every float small (see `Camera`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub cell: [i32; 3],
    pub origin: [f32; 3],
    pub dir: [f32; 3],
}

/// The first solid Voxel along a ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Distance along the ray to the face it entered the Voxel through.
    pub t: f32,
    /// Where that is, relative to the corner of the ray's `cell`.
    pub position: [f32; 3],
    /// Unit normal of that face, pointing back towards the ray. For a ray that starts inside a
    /// solid Voxel (`t` = 0) it faces against the ray's largest direction component.
    pub normal: [i32; 3],
    /// Absolute Planet-frame Voxel coordinate. For a Voxel of a coarser level, the Voxel of
    /// level 0 inside it that the hit is in: on the face it was entered through.
    pub voxel: [i32; 3],
    pub material: u16,
}

/// Brick entries are walked with at most this many steps: a ray crosses at most 3 * 8 - 2.
const BLOCK_STEPS: usize = 24;
const MAX_REGION_STEPS: usize = 4096;
/// Stands in for a zero direction component, so that 1 / d is finite.
const TINY: f32 = 1e-20;

/// What every level of the walk needs; see `Walk` in the shader.
struct Walk {
    d: [f32; 3],
    inv: [f32; 3],
    step: [i32; 3],
    upper: [i32; 3],
}

/// The axis a DDA steps along: the smallest of `t`, ties to x, then y, then z.
fn first_axis(t: [f32; 3]) -> usize {
    if t[0] <= t[1] && t[0] <= t[2] {
        0
    } else if t[1] <= t[2] {
        1
    } else {
        2
    }
}

/// The axis of the largest component of `v`, ties as above.
fn largest_axis(v: [f32; 3]) -> usize {
    if v[0] >= v[1] && v[0] >= v[2] {
        0
    } else if v[1] >= v[2] {
        1
    } else {
        2
    }
}

fn min3(v: [f32; 3]) -> f32 {
    v[0].min(v[1].min(v[2]))
}

fn max3(v: [f32; 3]) -> f32 {
    v[0].max(v[1].max(v[2]))
}

/// A unit vector along `axis`, pointing against the ray's step on it.
fn face(w: &Walk, axis: usize) -> [i32; 3] {
    let mut normal = [0; 3];
    normal[axis] = -w.step[axis];
    normal
}

/// Distance along the ray to each of the planes at absolute Voxel coordinate `plane`.
fn reach(ray: &Ray, w: &Walk, plane: [i32; 3]) -> [f32; 3] {
    std::array::from_fn(|a| {
        ((plane[a].wrapping_sub(ray.cell[a])) as f32 - ray.origin[a]) * w.inv[a]
    })
}

/// The Voxel the ray is in at distance `t`.
fn voxel_at(ray: &Ray, w: &Walk, t: f32) -> [i32; 3] {
    std::array::from_fn(|a| ray.cell[a].wrapping_add((ray.origin[a] + w.d[a] * t).floor() as i32))
}

fn hit_at(ray: &Ray, t: f32, normal: [i32; 3], voxel: [i32; 3], material: u32) -> Hit {
    Hit {
        t,
        position: std::array::from_fn(|a| ray.origin[a] + ray.dir[a] * t),
        normal,
        voxel,
        material: material as u16,
    }
}

/// The hit on the Voxel at `voxel`, in Voxels of mip `level`, entered at `t` through the face
/// with `normal`: as [`hit_at`], with the Voxel of level 0 that the ray is in at `t`, which is
/// the one inside `voxel` that touches the face. For level 0 that is `voxel`.
fn hit_level_voxel(
    ray: &Ray,
    w: &Walk,
    level: u32,
    t: f32,
    normal: [i32; 3],
    voxel: [i32; 3],
    material: u32,
) -> Hit {
    let low = voxel.map(|v| v << level);
    let high = low.map(|l| l + (1 << level) - 1);
    let at = clamp3(voxel_at(ray, w, t), low, high);
    let inner = std::array::from_fn(|a| match normal[a] {
        n if n < 0 => low[a],
        n if n > 0 => high[a],
        _ => at[a],
    });
    hit_at(ray, t, normal, inner, material)
}

fn clamp3(v: [i32; 3], lo: [i32; 3], hi: [i32; 3]) -> [i32; 3] {
    std::array::from_fn(|a| v[a].clamp(lo[a], hi[a]))
}

/// Whether `v` is outside `lo..=hi` on any axis.
fn outside(v: [i32; 3], lo: [i32; 3], hi: [i32; 3]) -> bool {
    (0..3).any(|a| v[a] < lo[a] || v[a] > hi[a])
}

#[derive(Clone, Copy)]
struct Entry {
    t: f32,
    normal: [i32; 3],
}

/// Walks the Voxels of the brick at absolute brick coordinate `brick` of mip `level` (entry
/// `entry` in `brick_refs`).
fn hit_brick(
    scene: &Scene,
    ray: &Ray,
    w: &Walk,
    level: u32,
    entry: u32,
    brick: [i32; 3],
    enter: Entry,
) -> Option<Hit> {
    let first = brick.map(|b| b * 8);
    let last = first.map(|f| f + 7);
    let mut voxel = clamp3(voxel_at(ray, w, enter.t).map(|v| v >> level), first, last);
    if entry & SOLID_FLAG != 0 {
        return Some(hit_level_voxel(
            ray,
            w,
            level,
            enter.t,
            enter.normal,
            voxel,
            entry & 0xffff,
        ));
    }
    let entry = entry as usize;
    let size = (1u32 << level) as f32;
    let mut next = reach(
        ray,
        w,
        std::array::from_fn(|a| (voxel[a] + w.upper[a]) << level),
    );
    let delta = w.inv.map(|i| (i * size).abs());
    let (mut t, mut normal) = (enter.t, enter.normal);
    for _ in 0..BLOCK_STEPS {
        count(|s| s.voxels += 1);
        let local: [u32; 3] = std::array::from_fn(|a| (voxel[a] - first[a]) as u32);
        let bit = local[0] | local[1] << 3 | local[2] << 6;
        if scene.bricks[entry + (bit >> 5) as usize] >> (bit & 31) & 1 != 0 {
            let material = mixed_material(scene, entry, bit);
            return Some(hit_level_voxel(ray, w, level, t, normal, voxel, material));
        }
        let axis = first_axis(next);
        t = min3(next);
        voxel[axis] += w.step[axis];
        next[axis] += delta[axis];
        normal = face(w, axis);
        if outside(voxel, first, last) {
            break;
        }
    }
    None
}

/// Material of Voxel `bit` in the mixed brick at `at` in `bricks`.
fn mixed_material(scene: &Scene, at: usize, bit: u32) -> u32 {
    let bricks = &scene.bricks;
    let header = bricks[at + 16];
    let bits = header & 0xff;
    let palette_len = (header >> 8) as usize;
    let palette = at + 17;
    let indices = palette + palette_len.div_ceil(2);
    let position = bit * bits;
    let index = (bricks[indices + (position / 32) as usize] >> (position % 32)) & ((1 << bits) - 1);
    (bricks[palette + (index / 2) as usize] >> ((index % 2) * 16)) & 0xffff
}

/// Walks the bricks of the region at absolute region coordinate `rc` of mip `level` (index
/// `region` in `regions`).
fn hit_region(
    scene: &Scene,
    ray: &Ray,
    w: &Walk,
    level: u32,
    region: usize,
    rc: [i32; 3],
    enter: Entry,
) -> Option<Hit> {
    let base = region * REGION_WORDS;
    let first = rc.map(|r| r * 8);
    let last = first.map(|f| f + 7);
    let first_ref = scene.regions[base + 16];
    if first_ref & SOLID_FLAG != 0 {
        // The whole region is one Material.
        let voxel = clamp3(
            voxel_at(ray, w, enter.t).map(|v| v >> level),
            first.map(|f| f * 8),
            last.map(|l| l * 8 + 7),
        );
        return Some(hit_level_voxel(
            ray,
            w,
            level,
            enter.t,
            enter.normal,
            voxel,
            first_ref & 0xffff,
        ));
    }
    let mut brick = clamp3(
        voxel_at(ray, w, enter.t).map(|v| v >> (level + 3)),
        first,
        last,
    );
    let mut next = reach(
        ray,
        w,
        std::array::from_fn(|a| ((brick[a] + w.upper[a]) * 8) << level),
    );
    let size = (1u32 << level) as f32;
    let delta = w.inv.map(|i| (8.0 * size * i).abs());
    let (mut t, mut normal) = (enter.t, enter.normal);
    for _ in 0..BLOCK_STEPS {
        count(|s| s.bricks += 1);
        let local: [u32; 3] = std::array::from_fn(|a| (brick[a] - first[a]) as u32);
        let bit = local[0] | local[1] << 3 | local[2] << 6;
        let word = scene.regions[base + (bit >> 5) as usize];
        if word >> (bit & 31) & 1 != 0 {
            // The entry follows one entry for each lower bit set in the mask.
            let mut rank = (word & ((1u32 << (bit & 31)) - 1)).count_ones();
            for j in 0..(bit >> 5) as usize {
                rank += scene.regions[base + j].count_ones();
            }
            let entry = scene.brick_refs[(first_ref + rank) as usize];
            if let hit @ Some(_) =
                hit_brick(scene, ray, w, level, entry, brick, Entry { t, normal })
            {
                return hit;
            }
        }
        let axis = first_axis(next);
        t = min3(next);
        brick[axis] += w.step[axis];
        next[axis] += delta[axis];
        normal = face(w, axis);
        if outside(brick, first, last) {
            break;
        }
    }
    None
}

/// The first solid Voxel of mip `level` along the ray, in its box and outside its hole, if it is
/// between `from` and `limit`.
fn hit_level(scene: &Scene, ray: &Ray, w: &Walk, level: u32, from: f32, limit: f32) -> Option<Hit> {
    count(|s| s.examined += 1);
    let lv = &scene.levels[level as usize];
    let shift = 6 + level;
    let (lowest, end) = (lv.occupied_first, lv.occupied_end);
    if (0..3).any(|a| end[a] <= lowest[a]) {
        return None;
    }
    // Clip the ray to the regions that the level lists.
    let t_a = reach(ray, w, lowest.map(|f| f << shift));
    let t_b = reach(ray, w, end.map(|e| e << shift));
    let t_near: [f32; 3] = std::array::from_fn(|a| t_a[a].min(t_b[a]));
    let t_far: [f32; 3] = std::array::from_fn(|a| t_a[a].max(t_b[a]));
    let t_enter = max3(t_near).max(from);
    let t_exit = min3(t_far);
    if t_enter > t_exit || t_enter >= limit {
        return None;
    }
    // The ray enters through the face it crosses last. If it starts inside the box, the normal
    // only matters for a start inside a solid Voxel: face against the largest component.
    let mut normal = if max3(t_near) > 0.0 {
        face(w, largest_axis(t_near))
    } else {
        face(w, largest_axis(w.d.map(f32::abs)))
    };

    let highest = end.map(|e| e - 1);
    let (hole_first, hole_end) = scene.hole(level as usize);
    let hole_last = hole_end.map(|e| e - 1);

    // A ray that starts in the hole has nothing to find in it: walk from where it leaves.
    let h_a = reach(ray, w, hole_first.map(|f| f << shift));
    let h_b = reach(ray, w, hole_end.map(|e| e << shift));
    let h_near: [f32; 3] = std::array::from_fn(|a| h_a[a].min(h_b[a]));
    let h_far: [f32; 3] = std::array::from_fn(|a| h_a[a].max(h_b[a]));
    let mut t_enter = t_enter;
    if max3(h_near) <= t_enter && t_enter < min3(h_far) {
        t_enter = min3(h_far);
        normal = face(w, first_axis(h_far));
        if t_enter >= limit {
            return None;
        }
        // The ray may have left the regions that the level lists while it was in the hole.
        if t_enter > t_exit {
            return None;
        }
    }
    count(|s| s.levels += 1);
    let mut t = t_enter;
    let mut rc = clamp3(voxel_at(ray, w, t).map(|v| v >> shift), lowest, highest);
    let mut next = reach(
        ray,
        w,
        std::array::from_fn(|a| (rc[a] + w.upper[a]) << shift),
    );
    let delta = w.inv.map(|i| ((64u32 << level) as f32 * i).abs());
    for _ in 0..MAX_REGION_STEPS {
        // Nothing from here on is nearer than the hit we have.
        if t >= limit {
            break;
        }
        count(|s| s.regions += 1);
        // The regions of the hole have no brick of this level.
        if outside(rc, hole_first, hole_last)
            && let Some(region) = {
                let (found, probes) = scene.find_region_probes(level as usize, rc);
                count(|s| s.probes += probes);
                found
            }
            && let hit @ Some(_) = hit_region(scene, ray, w, level, region, rc, Entry { t, normal })
        {
            return hit;
        }
        let axis = first_axis(next);
        t = min3(next);
        rc[axis] += w.step[axis];
        next[axis] += delta[axis];
        normal = face(w, axis);
        if outside(rc, lowest, highest) {
            break;
        }
    }
    None
}

/// The first solid Voxel along `ray`, if any.
pub fn trace(scene: &Scene, ray: &Ray) -> Option<Hit> {
    let d: [f32; 3] = std::array::from_fn(|a| {
        if ray.dir[a].abs() < TINY {
            TINY
        } else {
            ray.dir[a]
        }
    });
    let w = Walk {
        d,
        inv: d.map(|d| 1.0 / d),
        step: d.map(|d| if d > 0.0 { 1 } else { -1 }),
        upper: d.map(|d| i32::from(d > 0.0)),
    };

    // Clip the ray to the ball that has the whole Planet in it: nothing is found outside it.
    let (mut from, mut limit) = (0.0, f32::MAX);
    if let (Some(bound), Some(radius)) = (scene.bound, scene.clip_radius()) {
        let oc: [f32; 3] = std::array::from_fn(|a| {
            bound.centre[a].wrapping_sub(ray.cell[a]) as f32 - ray.origin[a]
        });
        let along = oc[0] * ray.dir[0] + oc[1] * ray.dir[1] + oc[2] * ray.dir[2];
        let off: [f32; 3] = std::array::from_fn(|a| oc[a] - ray.dir[a] * along);
        let h2 = radius * radius - (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]);
        if h2 < 0.0 {
            return None;
        }
        let h = h2.sqrt();
        if along + h < 0.0 {
            return None;
        }
        from = (along - h).max(0.0);
        limit = along + h;
    }

    // Every level is a part of space of its own, so the nearest hit of all is the one that is
    // nearest: a ray that does not start at the camera meets the levels in any order. Once there
    // is a hit, the levels after it only need looking at up to it.
    let mut nearest = None;
    for level in 0..scene.walked_levels() as u32 {
        if let Some(hit) = hit_level(scene, ray, &w, level, from, limit)
            && hit.t < limit
        {
            limit = hit.t;
            nearest = Some(hit);
        }
    }
    nearest
}

#[cfg(test)]
mod tests {
    use brick_store::{BrickCoord, BrickStore, Material, VoxelCoord};
    use planet_gen::generate_region;

    use super::*;
    use crate::scene::Core;

    const STONE: Material = Material(7);

    fn store_with(voxels: &[[i32; 3]]) -> BrickStore {
        let mut store = BrickStore::new();
        for &v in voxels {
            store.set(VoxelCoord(v), Some(STONE));
        }
        store
    }

    /// A ray from the middle of `cell` along `dir`, normalised.
    fn ray(cell: [i32; 3], dir: [f32; 3]) -> Ray {
        ray_from(cell, [0.5; 3], dir)
    }

    fn ray_from(cell: [i32; 3], origin: [f32; 3], dir: [f32; 3]) -> Ray {
        let len = dir.iter().map(|d| d * d).sum::<f32>().sqrt();
        Ray {
            cell,
            origin,
            dir: dir.map(|d| d / len),
        }
    }

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn a_ray_hits_the_face_of_a_voxel_it_looks_at() {
        let scene = Scene::new(&store_with(&[[10, 0, 0]]), None);
        let hit = trace(&scene, &ray([0, 0, 0], [1.0, 0.0, 0.0])).unwrap();
        assert_eq!(hit.voxel, [10, 0, 0]);
        assert_eq!(hit.normal, [-1, 0, 0]);
        assert_eq!(hit.material, 7);
        assert!(near(hit.t, 9.5));
        assert!(near(hit.position[0], 10.0) && near(hit.position[1], 0.5));
    }

    #[test]
    fn axis_aligned_rays_hit_the_right_face_from_all_six_sides() {
        let scene = Scene::new(&store_with(&[[0, 0, 0]]), None);
        for axis in 0..3 {
            for sign in [-1, 1] {
                let mut cell = [0; 3];
                cell[axis] = 5 * sign;
                let mut dir = [0.0; 3];
                dir[axis] = -sign as f32;
                let hit = trace(&scene, &ray(cell, dir)).unwrap();
                let mut normal = [0; 3];
                normal[axis] = sign;
                assert_eq!(hit.voxel, [0, 0, 0], "axis {axis} sign {sign}");
                assert_eq!(hit.normal, normal, "axis {axis} sign {sign}");
                // The ray starts 0.5 into its cell, 5 cells away, and meets the near face.
                assert!(near(hit.t, 4.5), "{}", hit.t);
            }
        }
    }

    #[test]
    fn the_nearer_of_two_voxels_is_hit() {
        let scene = Scene::new(&store_with(&[[20, 0, 0], [6, 0, 0], [13, 0, 0]]), None);
        let hit = trace(&scene, &ray([0, 0, 0], [1.0, 0.0, 0.0])).unwrap();
        assert_eq!(hit.voxel, [6, 0, 0]);
    }

    #[test]
    fn a_ray_that_looks_elsewhere_or_away_misses() {
        let scene = Scene::new(&store_with(&[[10, 0, 0]]), None);
        assert_eq!(trace(&scene, &ray([0, 0, 0], [-1.0, 0.0, 0.0])), None);
        assert_eq!(trace(&scene, &ray([0, 0, 0], [1.0, 0.1, 0.0])), None);
        assert_eq!(trace(&scene, &ray([0, 0, 0], [0.0, 1.0, 0.0])), None);
        // Starting beyond the voxel and heading away from it.
        assert_eq!(trace(&scene, &ray([11, 0, 0], [1.0, 0.0, 0.0])), None);
    }

    #[test]
    fn an_empty_scene_is_all_misses() {
        let scene = Scene::new(&BrickStore::new(), None);
        assert_eq!(trace(&scene, &ray([0, 0, 0], [1.0, 0.3, 0.2])), None);
    }

    #[test]
    fn a_ray_starting_inside_a_solid_voxel_hits_it_at_zero() {
        let scene = Scene::new(&store_with(&[[3, 4, 5]]), None);
        let hit = trace(&scene, &ray([3, 4, 5], [0.2, -1.0, 0.1])).unwrap();
        assert_eq!(hit.voxel, [3, 4, 5]);
        assert_eq!(hit.t, 0.0);
        // Facing against the largest component, which is -y.
        assert_eq!(hit.normal, [0, 1, 0]);
    }

    #[test]
    fn a_ray_starting_in_the_air_of_a_solid_brick_walks_to_the_next_voxel() {
        // A brick that is solid except for one dug-out Voxel.
        let mut store = BrickStore::new();
        store.insert(BrickCoord([0, 0, 0]), brick_store::Brick::solid(STONE));
        store.set(VoxelCoord([2, 2, 2]), None);
        let scene = Scene::new(&store, None);
        let hit = trace(&scene, &ray([2, 2, 2], [1.0, 0.0, 0.0])).unwrap();
        assert_eq!(hit.voxel, [3, 2, 2]);
        assert_eq!(hit.normal, [-1, 0, 0]);
        assert!(near(hit.t, 0.5));
    }

    #[test]
    fn a_solid_brick_is_hit_at_the_face_the_ray_enters() {
        let mut store = BrickStore::new();
        store.insert(BrickCoord([5, 0, 0]), brick_store::Brick::solid(STONE));
        let scene = Scene::new(&store, None);
        let hit = trace(&scene, &ray([0, 3, 4], [1.0, 0.0, 0.0])).unwrap();
        assert_eq!(hit.voxel, [40, 3, 4]);
        assert_eq!(hit.normal, [-1, 0, 0]);
        assert_eq!(hit.material, 7);
    }

    #[test]
    fn a_ray_crosses_empty_regions_to_a_distant_voxel() {
        let scene = Scene::new(&store_with(&[[-900, 35, 70], [1000, 3, -500]]), None);
        let hit = trace(&scene, &ray([-1000, 0, 0], [100.0, 35.0, 70.0])).unwrap();
        assert_eq!(hit.voxel, [-900, 35, 70]);
        let hit = trace(&scene, &ray([0, 0, 0], [1000.0, 3.0, -500.0])).unwrap();
        assert_eq!(hit.voxel, [1000, 3, -500]);
    }

    #[test]
    fn region_and_brick_boundaries_are_not_skipped() {
        // The first Voxel of a region, the last of the previous one, and a brick edge.
        let scene = Scene::new(
            &store_with(&[[64, 0, 0], [63, 5, 0], [8, 9, 0], [7, 9, 0]]),
            None,
        );
        assert_eq!(
            trace(&scene, &ray([40, 0, 0], [1.0, 0.0, 0.0]))
                .unwrap()
                .voxel,
            [64, 0, 0]
        );
        assert_eq!(
            trace(&scene, &ray([40, 5, 0], [1.0, 0.0, 0.0]))
                .unwrap()
                .voxel,
            [63, 5, 0]
        );
        assert_eq!(
            trace(&scene, &ray([-20, 9, 0], [1.0, 0.0, 0.0]))
                .unwrap()
                .voxel,
            [7, 9, 0]
        );
        assert_eq!(
            trace(&scene, &ray([20, 9, 0], [-1.0, 0.0, 0.0]))
                .unwrap()
                .voxel,
            [8, 9, 0]
        );
    }

    #[test]
    fn negative_coordinates_work_like_positive_ones() {
        let scene = Scene::new(&store_with(&[[-1, -1, -1], [-65, -9, -8]]), None);
        let hit = trace(&scene, &ray([-30, -1, -1], [-1.0, 0.0, 0.0]));
        assert_eq!(hit, None);
        let hit = trace(&scene, &ray([5, -1, -1], [-1.0, 0.0, 0.0])).unwrap();
        assert_eq!(hit.voxel, [-1, -1, -1]);
        assert_eq!(hit.normal, [1, 0, 0]);
        let hit = trace(&scene, &ray([-20, -9, -8], [-1.0, 0.0, 0.0])).unwrap();
        assert_eq!(hit.voxel, [-65, -9, -8]);
    }

    #[test]
    fn a_ray_grazing_along_a_face_plane_does_not_hang_or_hit_the_neighbour() {
        let scene = Scene::new(&store_with(&[[10, 1, 0]]), None);
        // Runs exactly along the underside plane y = 1 of the voxel: both sides are air.
        let grazing = ray_from([0, 1, 0], [0.0, 0.0, 0.5], [1.0, 0.0, 0.0]);
        let hit = trace(&scene, &grazing);
        assert!(hit.is_none() || hit.unwrap().voxel == [10, 1, 0]);
    }

    #[test]
    fn a_ray_far_from_the_origin_keeps_its_precision() {
        let far = 2_000_000;
        let scene = Scene::new(&store_with(&[[far + 10, 0, 0]]), None);
        let hit = trace(&scene, &ray([far, 0, 0], [1.0, 0.0, 0.0])).unwrap();
        assert_eq!(hit.voxel, [far + 10, 0, 0]);
        assert!(near(hit.t, 9.5));
    }

    #[test]
    fn materials_come_from_the_palette_of_a_mixed_brick() {
        let mut store = BrickStore::new();
        for (x, m) in [(1, 1), (2, 2), (3, 300), (4, 9)] {
            store.set(VoxelCoord([x, 0, 0]), Some(Material(m)));
        }
        let scene = Scene::new(&store, None);
        for (x, m) in [(1, 1), (2, 2), (3, 300), (4, 9)] {
            let hit = trace(&scene, &ray([x, 3, 0], [0.0, -1.0, 0.0])).unwrap();
            assert_eq!((hit.voxel, hit.material), ([x, 0, 0], m));
        }
    }

    /// Plain per-Voxel DDA over the store, with none of the Scene structure: the oracle.
    fn oracle(
        solid: impl Fn([i32; 3]) -> Option<u16>,
        ray: &Ray,
        limit: i32,
    ) -> Option<([i32; 3], [i32; 3], u16)> {
        let p0: [f64; 3] =
            std::array::from_fn(|a| f64::from(ray.cell[a]) + f64::from(ray.origin[a]));
        let d: [f64; 3] = ray.dir.map(f64::from);
        let mut voxel = ray.cell;
        let step: [i32; 3] = d.map(|d| if d >= 0.0 { 1 } else { -1 });
        let mut next: [f64; 3] = std::array::from_fn(|a| {
            if d[a] == 0.0 {
                f64::INFINITY
            } else {
                let plane = f64::from(voxel[a] + i32::from(step[a] > 0));
                (plane - p0[a]) / d[a]
            }
        });
        let mut normal = [0; 3];
        for _ in 0..limit {
            if let Some(m) = solid(voxel) {
                return Some((voxel, normal, m));
            }
            let a = (0..3).min_by(|&a, &b| next[a].total_cmp(&next[b])).unwrap();
            voxel[a] += step[a];
            normal = [0; 3];
            normal[a] = -step[a];
            next[a] += 1.0 / d[a].abs();
        }
        None
    }

    #[test]
    fn a_core_is_solid_through_its_uniform_and_partial_regions() {
        let radius = 120;
        let core = Core {
            radius,
            material: STONE,
        };
        let scene = Scene::new(&BrickStore::new(), Some(core));
        let solid = |v: [i32; 3]| VoxelCoord(v).brick().within_ball(radius).then_some(7);
        let mut rng = Lcg(5);
        let mut hits = 0;
        for _ in 0..300 {
            let cell = [
                -400,
                (rng.next() * 240.0 - 120.0) as i32,
                (rng.next() * 240.0 - 120.0) as i32,
            ];
            let r = ray_from(
                cell,
                [rng.next(), rng.next(), rng.next()],
                [1.0, rng.next() * 0.2 - 0.1, rng.next() * 0.2 - 0.1],
            );
            let got = trace(&scene, &r);
            assert_eq!(
                got.map(|h| (h.voxel, h.normal, h.material)),
                oracle(solid, &r, 2000),
                "{r:?}"
            );
            hits += usize::from(got.is_some());
        }
        assert!(hits > 100, "{hits}");
        // From inside the core the first Voxel is the one the ray starts in.
        let inside = trace(&scene, &ray([3, 4, 5], [0.0, 0.0, 1.0])).unwrap();
        assert_eq!((inside.t, inside.voxel), (0.0, [3, 4, 5]));
    }

    #[test]
    fn every_palette_width_decodes_to_the_right_material() {
        // Palettes of 2, 3, 5, 17 and 300 Materials use 1, 2, 4, 8 and 16 bits per index.
        let sizes = [2, 3, 5, 17, 300];
        let voxel_of = |i: usize, v: usize| {
            [
                i as i32 * 8 + (v % 8) as i32,
                (v / 8 % 8) as i32,
                (v / 64) as i32,
            ]
        };
        let mut store = BrickStore::new();
        for (i, n) in sizes.into_iter().enumerate() {
            for v in 0..n {
                let material = Material(100 * (i as u16 + 1) + v as u16);
                store.set(VoxelCoord(voxel_of(i, v)), Some(material));
            }
        }
        let scene = Scene::new(&store, None);
        for (i, n) in sizes.into_iter().enumerate() {
            for v in 0..n {
                let voxel = voxel_of(i, v);
                let hit = trace(&scene, &ray(voxel, [0.0, 0.0, 1.0])).unwrap();
                assert_eq!(
                    (hit.voxel, hit.material),
                    (voxel, 100 * (i as u16 + 1) + v as u16)
                );
            }
        }
    }

    /// A small deterministic generator, so the ray sets are fixed.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }
    }

    #[test]
    fn it_agrees_with_a_plain_voxel_walk_on_the_generated_planet() {
        // A patch over the +y pole, from below the soil to above the highest peak.
        let store = generate_region(7, BrickCoord([-12, 178, -12]), BrickCoord([12, 193, 12]));
        let scene = Scene::new(&store, None);
        let mut rng = Lcg(42);
        let mut hits = 0;
        for _ in 0..400 {
            let cell = [
                (rng.next() * 80.0 - 40.0) as i32,
                1560 + (rng.next() * 100.0) as i32,
                (rng.next() * 80.0 - 40.0) as i32,
            ];
            let dir = [
                rng.next() * 0.5 - 0.25,
                -0.6 - rng.next(),
                rng.next() * 0.5 - 0.25,
            ];
            let r = ray_from(cell, [rng.next(), rng.next(), rng.next()], dir);
            let want = oracle(|v| store.get(VoxelCoord(v)).map(|m| m.0), &r, 2000);
            let got = trace(&scene, &r);
            assert_eq!(got.map(|h| (h.voxel, h.normal, h.material)), want, "{r:?}");
            hits += usize::from(got.is_some());
        }
        assert!(hits > 350, "only {hits} of 400 rays hit the patch");
    }
}
