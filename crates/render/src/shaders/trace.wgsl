// trace(ray) -> Hit: the one place that knows how the Planet is stored and walked.
//
// The data layout is documented in scene.rs. reference.rs is a line-for-line CPU copy of this
// file, checked against it by tests: change them together.
//
// Voxel, brick and region coordinates are absolute i32 Planet-frame integers. Floats are only
// ever relative to the ray's `cell`, the integer Voxel the camera is in, so nothing loses
// precision far from the Planet's origin.
//
// The Planet is in levels, finest first: level n has Voxels 2^n times the size of level 0's, and
// lists only the bricks that the LOD bands select for it (see lod.rs), so that each part of space
// is seen at one level only, finer near the camera and coarser far from it. Voxel, brick and
// region coordinates are those of the level being walked; distances along the ray are in Voxels
// of level 0, at every level.

// Most levels there can be, as in scene.rs.
const MAX_LEVELS = 20;

struct Level {
    // The box of regions that the level lists, from the first up to, and not including, the end,
    // as in `Level` in scene.rs: nothing is walked outside it, and there is nothing to walk if
    // the end is not past the first. And the slot count of the level's part of `table` minus
    // one, and its first slot.
    occupied_first: vec3<i32>,
    table_mask: u32,
    occupied_end: vec3<i32>,
    table_base: u32,
    // The regions about the camera that no brick of this level is in: from the first up to, and
    // not including, the end. None if the end is not past the first.
    hole_first: vec3<i32>,
    pad0: u32,
    hole_end: vec3<i32>,
    pad1: u32,
}

struct SceneHeader {
    level_count: u32,
    // The ball that has the whole Planet in it, as `Bound` in scene.rs, with the radius that
    // `Scene::clip_radius` gives (which allows for coarse Voxels that stick out of it): nothing
    // is found outside it. A radius of 0 means there is no ball.
    bound_radius: f32,
    bound_centre: vec3<i32>,
    levels: array<Level, MAX_LEVELS>,
}

@group(0) @binding(0) var<uniform> scene: SceneHeader;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read> regions: array<u32>;
@group(0) @binding(3) var<storage, read> brick_refs: array<u32>;
@group(0) @binding(4) var<storage, read> bricks: array<u32>;

const REGION_WORDS = 17u;
const SOLID_FLAG = 0x80000000u;
// A ray crosses at most 3 * 8 - 2 = 22 cells of an 8 x 8 x 8 block.
const BLOCK_STEPS = 24;
const MAX_REGION_STEPS = 4096;
// Stands in for a zero direction component, so that 1 / d is finite.
const TINY = 1e-20;

struct Ray {
    // The Voxel the ray starts in.
    cell: vec3<i32>,
    // Start of the ray relative to the corner of `cell`, normally in 0..1.
    origin: vec3<f32>,
    // Unit length, so that distances are in Voxels.
    dir: vec3<f32>,
}

struct Hit {
    hit: bool,
    // Distance along the ray to the face it entered the Voxel through.
    t: f32,
    // Where that is, relative to the corner of `ray.cell`.
    position: vec3<f32>,
    // Unit normal of that face, pointing back towards the ray. For a ray that starts inside a
    // solid Voxel (t = 0) it faces against the ray's largest direction component.
    normal: vec3<f32>,
    // Absolute Planet-frame Voxel coordinate. For a Voxel of a coarser level, the Voxel of level 0
    // inside it that the hit is in: on the face the ray entered through.
    voxel: vec3<i32>,
    material: u32,
}

fn hash_region(key: vec3<i32>) -> u32 {
    let k = bitcast<vec3<u32>>(key);
    var h = (k.x * 0x8da6b343u) ^ (k.y * 0xd8163841u) ^ (k.z * 0xcb1ab31fu);
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    return h ^ (h >> 12u);
}

// Index of the region at absolute region coordinate `key` of `level` in `regions`, or -1.
fn find_region(level: u32, key: vec3<i32>) -> i32 {
    let k = bitcast<vec3<u32>>(key);
    let mask = scene.levels[level].table_mask;
    let base = scene.levels[level].table_base;
    var slot = hash_region(key) & mask;
    for (var i = 0u; i <= mask; i++) {
        let at = (base + slot) * 4u;
        let index = table[at + 3u];
        if index == 0u {
            return -1;
        }
        if table[at] == k.x && table[at + 1u] == k.y && table[at + 2u] == k.z {
            return i32(index - 1u);
        }
        slot = (slot + 1u) & mask;
    }
    return -1;
}

// The axis a DDA steps along: the smallest of `t`, with ties going to x, then y, then z.
fn first_axis(t: vec3<f32>) -> vec3<bool> {
    let x = t.x <= t.y && t.x <= t.z;
    let y = !x && t.y <= t.z;
    return vec3<bool>(x, y, !x && !y);
}

// The axis of the largest component of `v`, ties as above.
fn largest_axis(v: vec3<f32>) -> vec3<bool> {
    let x = v.x >= v.y && v.x >= v.z;
    let y = !x && v.y >= v.z;
    return vec3<bool>(x, y, !x && !y);
}

fn min3(v: vec3<f32>) -> f32 {
    return min(v.x, min(v.y, v.z));
}

fn max3(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

// Material of Voxel `bit` in the mixed brick at `at` in `bricks`.
fn mixed_material(at: u32, bit: u32) -> u32 {
    let header = bricks[at + 16u];
    let bits = header & 0xffu;
    let palette_len = header >> 8u;
    let palette = at + 17u;
    let indices = palette + (palette_len + 1u) / 2u;
    let position = bit * bits;
    let index = (bricks[indices + position / 32u] >> (position % 32u)) & ((1u << bits) - 1u);
    return (bricks[palette + index / 2u] >> ((index % 2u) * 16u)) & 0xffffu;
}

fn miss() -> Hit {
    var h: Hit;
    h.hit = false;
    return h;
}

fn hit_at(ray: Ray, t: f32, normal: vec3<f32>, voxel: vec3<i32>, material: u32) -> Hit {
    var h: Hit;
    h.hit = true;
    h.t = t;
    h.position = ray.origin + ray.dir * t;
    h.normal = normal;
    h.voxel = voxel;
    h.material = material;
    return h;
}

// The hit on the Voxel at `voxel` of `level`, entered at `t` through the face with `normal`: as
// hit_at, with the Voxel of level 0 that is inside it and on that face, where the ray is.
fn hit_level_voxel(ray: Ray, w: Walk, level: u32, t: f32, normal: vec3<f32>, voxel: vec3<i32>, material: u32) -> Hit {
    let low = voxel << vec3<u32>(level);
    let high = low + (1i << level) - 1;
    let at = clamp(voxel_at(ray, w, t), low, high);
    let inner = select(select(at, high, normal > vec3<f32>(0.0)), low, normal < vec3<f32>(0.0));
    return hit_at(ray, t, normal, inner, material);
}

// The ray with what every level of the walk needs. `d` is the direction with zeros nudged off
// zero, `inv` is 1 / d, `step` is the sign of d, and `upper` is 1 on axes that step upwards.
struct Walk {
    d: vec3<f32>,
    inv: vec3<f32>,
    step: vec3<i32>,
    upper: vec3<i32>,
}

// Distance along the ray to each of the planes at absolute Voxel coordinate `plane`.
fn reach(ray: Ray, w: Walk, plane: vec3<i32>) -> vec3<f32> {
    return (vec3<f32>(plane - ray.cell) - ray.origin) * w.inv;
}

// The Voxel the ray is in at distance `t`.
fn voxel_at(ray: Ray, w: Walk, t: f32) -> vec3<i32> {
    return ray.cell + vec3<i32>(floor(ray.origin + w.d * t));
}

// Walks the Voxels of the brick at absolute brick coordinate `brick` of `level` (entry `entry`
// in `brick_refs`), entered at `t_in` through the face with normal `normal_in`.
fn hit_brick(ray: Ray, w: Walk, level: u32, entry: u32, brick: vec3<i32>, t_in: f32, normal_in: vec3<f32>) -> Hit {
    let first = brick * 8;
    // The entry point is on the brick's boundary, so rounding can put it a hair outside.
    var voxel = clamp(voxel_at(ray, w, t_in) >> vec3<u32>(level), first, first + 7);
    if (entry & SOLID_FLAG) != 0u {
        return hit_level_voxel(ray, w, level, t_in, normal_in, voxel, entry & 0xffffu);
    }
    let size = f32(1u << level);
    var next = reach(ray, w, (voxel + w.upper) << vec3<u32>(level));
    let delta = abs(w.inv * size);
    var t = t_in;
    var normal = normal_in;
    for (var i = 0; i < BLOCK_STEPS; i++) {
        let local = vec3<u32>(voxel - first);
        let bit = local.x | (local.y << 3u) | (local.z << 6u);
        if ((bricks[entry + (bit >> 5u)] >> (bit & 31u)) & 1u) != 0u {
            return hit_level_voxel(ray, w, level, t, normal, voxel, mixed_material(entry, bit));
        }
        let axis = first_axis(next);
        t = min3(next);
        voxel += select(vec3<i32>(0), w.step, axis);
        next += select(vec3<f32>(0.0), delta, axis);
        normal = select(vec3<f32>(0.0), -vec3<f32>(w.step), axis);
        if any(voxel < first) || any(voxel > first + 7) {
            break;
        }
    }
    return miss();
}

// Walks the bricks of the region at absolute region coordinate `rc` of `level` (index `region`
// in `regions`), entered at `t_in` through the face with normal `normal_in`.
fn hit_region(ray: Ray, w: Walk, level: u32, region: u32, rc: vec3<i32>, t_in: f32, normal_in: vec3<f32>) -> Hit {
    let base = region * REGION_WORDS;
    let first = rc * 8;
    let first_ref = regions[base + 16u];
    if (first_ref & SOLID_FLAG) != 0u {
        // The whole region is one Material.
        let voxel = clamp(voxel_at(ray, w, t_in) >> vec3<u32>(level), first * 8, first * 8 + 63);
        return hit_level_voxel(ray, w, level, t_in, normal_in, voxel, first_ref & 0xffffu);
    }
    var brick = clamp(voxel_at(ray, w, t_in) >> vec3<u32>(level + 3u), first, first + 7);
    var next = reach(ray, w, ((brick + w.upper) * 8) << vec3<u32>(level));
    let size = f32(1u << level);
    let delta = abs(8.0 * size * w.inv);
    var t = t_in;
    var normal = normal_in;
    for (var i = 0; i < BLOCK_STEPS; i++) {
        let local = vec3<u32>(brick - first);
        let bit = local.x | (local.y << 3u) | (local.z << 6u);
        let word = regions[base + (bit >> 5u)];
        if ((word >> (bit & 31u)) & 1u) != 0u {
            // The entry follows one entry for each lower bit set in the mask.
            var rank = countOneBits(word & ((1u << (bit & 31u)) - 1u));
            for (var j = 0u; j < (bit >> 5u); j++) {
                rank += countOneBits(regions[base + j]);
            }
            let entry = brick_refs[first_ref + rank];
            let h = hit_brick(ray, w, level, entry, brick, t, normal);
            if h.hit {
                return h;
            }
        }
        let axis = first_axis(next);
        t = min3(next);
        brick += select(vec3<i32>(0), w.step, axis);
        next += select(vec3<f32>(0.0), delta, axis);
        normal = select(vec3<f32>(0.0), -vec3<f32>(w.step), axis);
        if any(brick < first) || any(brick > first + 7) {
            break;
        }
    }
    return miss();
}

// The first solid Voxel of `level` along the ray, in its box and outside its hole, if it is
// between `t_from` and `limit`, or a miss.
fn hit_level(ray: Ray, w: Walk, level: u32, t_from: f32, limit: f32) -> Hit {
    let lv = scene.levels[level];
    if any(lv.occupied_end <= lv.occupied_first) {
        return miss();
    }
    let shift = 6u + level;
    // Clip the ray to the regions that the level lists.
    let t_a = reach(ray, w, lv.occupied_first << vec3<u32>(shift));
    let t_b = reach(ray, w, lv.occupied_end << vec3<u32>(shift));
    let t_near = min(t_a, t_b);
    let t_enter = max(max3(t_near), t_from);
    let t_exit = min3(max(t_a, t_b));
    if t_enter > t_exit || t_enter >= limit {
        return miss();
    }
    // The ray enters through the face it crosses last. If it starts inside the box, the normal
    // only matters for a start inside a solid Voxel: face against the largest component.
    var normal = select(vec3<f32>(0.0), -vec3<f32>(w.step), largest_axis(abs(w.d)));
    if max3(t_near) > 0.0 {
        normal = select(vec3<f32>(0.0), -vec3<f32>(w.step), largest_axis(t_near));
    }

    let lowest = lv.occupied_first;
    let highest = lv.occupied_end - 1;
    let hole_last = lv.hole_end - 1;

    // A ray that starts in the hole has nothing to find in it: walk from where it leaves.
    let h_a = reach(ray, w, lv.hole_first << vec3<u32>(shift));
    let h_b = reach(ray, w, lv.hole_end << vec3<u32>(shift));
    let h_near = min(h_a, h_b);
    let h_far = max(h_a, h_b);
    var t_start = t_enter;
    if max3(h_near) <= t_start && t_start < min3(h_far) {
        t_start = min3(h_far);
        normal = select(vec3<f32>(0.0), -vec3<f32>(w.step), first_axis(h_far));
        if t_start >= limit {
            return miss();
        }
        // The ray may have left the regions that the level lists while it was in the hole.
        if t_start > t_exit {
            return miss();
        }
    }
    var t = t_start;
    var rc = clamp(voxel_at(ray, w, t) >> vec3<u32>(shift), lowest, highest);
    var next = reach(ray, w, (rc + w.upper) << vec3<u32>(shift));
    let delta = abs(f32(64u << level) * w.inv);
    for (var i = 0; i < MAX_REGION_STEPS; i++) {
        // Nothing from here on is nearer than the hit we have.
        if t >= limit {
            break;
        }
        // The regions of the hole have no brick of this level.
        if any(rc < lv.hole_first) || any(rc > hole_last) {
            let region = find_region(level, rc);
            if region >= 0 {
                let h = hit_region(ray, w, level, u32(region), rc, t, normal);
                if h.hit {
                    return h;
                }
            }
        }
        let axis = first_axis(next);
        t = min3(next);
        rc += select(vec3<i32>(0), w.step, axis);
        next += select(vec3<f32>(0.0), delta, axis);
        normal = select(vec3<f32>(0.0), -vec3<f32>(w.step), axis);
        if any(rc < lowest) || any(rc > highest) {
            break;
        }
    }
    return miss();
}

// The first solid Voxel along the ray, or a miss.
fn trace(ray: Ray) -> Hit {
    var w: Walk;
    w.d = select(ray.dir, vec3<f32>(TINY), abs(ray.dir) < vec3<f32>(TINY));
    w.inv = 1.0 / w.d;
    w.step = select(vec3<i32>(-1), vec3<i32>(1), w.d > vec3<f32>(0.0));
    w.upper = max(w.step, vec3<i32>(0));

    // Clip the ray to the ball that has the whole Planet in it: nothing is found outside it.
    var t_from = 0.0;
    var limit = 3.4e38;
    if scene.bound_radius > 0.0 {
        let oc = vec3<f32>(scene.bound_centre - ray.cell) - ray.origin;
        // Summed in the order that reference.rs does; whether the driver fuses them is its own.
        let along = oc.x * ray.dir.x + oc.y * ray.dir.y + oc.z * ray.dir.z;
        let off = oc - ray.dir * along;
        let h2 = scene.bound_radius * scene.bound_radius - (off.x * off.x + off.y * off.y + off.z * off.z);
        if h2 < 0.0 {
            return miss();
        }
        let h = sqrt(h2);
        if along + h < 0.0 {
            return miss();
        }
        t_from = max(along - h, 0.0);
        limit = along + h;
    }

    // Every level is a part of space of its own, so the nearest hit of all is the one that is
    // nearest: a ray that does not start at the camera meets the levels in any order. Once there
    // is a hit, the levels after it only need looking at up to it.
    var nearest = miss();
    for (var level = 0u; level < scene.level_count; level++) {
        let h = hit_level(ray, w, level, t_from, limit);
        if h.hit && h.t < limit {
            limit = h.t;
            nearest = h;
        }
    }
    return nearest;
}
