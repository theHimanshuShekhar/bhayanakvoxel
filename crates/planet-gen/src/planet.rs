//! The real Planet: 16 km in radius, made brick by brick from the [`WorldMap`].
//!
//! The world map gives the surface (an elevation and a biome per texel); this module gives what
//! lies above and below it. From the sky down, a Voxel is:
//! - **air**, or **water** if it is below sea level (static until Fluids), frozen to a thin
//!   sea ice at the top in polar seas;
//! - one of the biome's **strata** for [`STRATA_DEPTH`] Voxels below the ground, the topmost
//!   being the biome's surface Material (the sea floor has strata of its own);
//! - **deep rock**, below the strata, and
//! - the glowing **molten core**, within [`MOLTEN_RADIUS`] of the centre.
//!
//! The lowest the ground gets is [`TRENCH`] below sea level, so everything within
//! [`INTERIOR_RADIUS`] of the centre is core or deep rock whatever the world map says. That is
//! most of the Planet, and its bricks are uniform: one flag each, not memory. Only the bricks
//! near the ground need their Voxels evaluated, and most of those are decided by bounds on the
//! ground's height under them (see [`Planet::brick`]).
//!
//! Materials are named here by id constant, and a test checks each against the data file, which
//! is never read at run time: the output depends on these numbers and `GENERATOR_VERSION`, not
//! on what the data file calls a Material.

use std::fmt;
use std::sync::Arc;

use brick_store::{BRICK_SIZE, Brick, BrickCoord, BrickSource, BrickStore, Material, VoxelCoord};

pub use crate::SOIL;
use crate::world_map::{Biome, PEAK, SEA_LEVEL, TRENCH, Texel, WORLD_RADIUS, WorldMap};
use crate::{distance, par_bricks};

pub const GRASSY_SOIL: Material = Material(3);
pub const CLAY: Material = Material(4);
pub const GRAVEL: Material = Material(5);
pub const SAND: Material = Material(6);
pub const SANDSTONE: Material = Material(7);
pub const LIMESTONE: Material = Material(8);
pub const SHALE: Material = Material(9);
pub const GRANITE: Material = Material(10);
pub const BASALT: Material = Material(11);
pub const OBSIDIAN: Material = Material(12);
pub const DEEP_ROCK: Material = Material(13);
pub const SNOW: Material = Material(14);
pub const ICE: Material = Material(15);
/// The one Material of the terrain that gives off light.
pub const MOLTEN_CORE: Material = Material(16);
pub const WATER: Material = Material(27);

/// Radius of the molten core, in Voxels: 3.2 km.
pub const MOLTEN_RADIUS: i32 = 32_000;
/// Thickness of the strata, in Voxels: 48 m. Deep rock lies below.
pub const STRATA_DEPTH: i32 = 480;
/// Radius inside which a Voxel is core or deep rock whatever the world map says: below even the
/// lowest ground's strata.
pub const INTERIOR_RADIUS: i32 = WORLD_RADIUS - TRENCH - STRATA_DEPTH;

/// Every Voxel of the Planet is within this many Voxels of its centre: sea level's radius plus
/// the highest its relief goes. Nothing is generated beyond it, so a ray need not be walked
/// further.
pub const MAX_RADIUS: i32 = WORLD_RADIUS + PEAK;

/// The interior's layers, innermost first, as (outer radius in Voxels, Material).
const INTERIOR: [(i32, Material); 2] = [(MOLTEN_RADIUS, MOLTEN_CORE), (INTERIOR_RADIUS, DEEP_ROCK)];

/// What the ground is made of, topmost first, as (depth at which the layer ends in Voxels,
/// Material). The last ends at [`STRATA_DEPTH`].
type Strata = [(i32, Material)];

const TEMPERATE: &Strata = &[
    (4, GRASSY_SOIL),
    (40, SOIL),
    (120, CLAY),
    (300, LIMESTONE),
    (STRATA_DEPTH, SHALE),
];
const DESERT: &Strata = &[
    (10, SAND),
    (120, SANDSTONE),
    (300, LIMESTONE),
    (STRATA_DEPTH, SHALE),
];
const SNOWY: &Strata = &[
    (6, SNOW),
    (40, ICE),
    (100, GRAVEL),
    (300, SHALE),
    (STRATA_DEPTH, GRANITE),
];
const VOLCANIC: &Strata = &[
    (6, BASALT),
    (30, OBSIDIAN),
    (300, BASALT),
    (STRATA_DEPTH, GRANITE),
];
const SEABED: &Strata = &[
    (4, SAND),
    (60, CLAY),
    (300, LIMESTONE),
    (STRATA_DEPTH, SHALE),
];

/// The strata of the ground in a biome, or of the sea floor.
fn strata(biome: Biome, seabed: bool) -> &'static Strata {
    match (seabed, biome) {
        (true, _) => SEABED,
        (_, Biome::Temperate) => TEMPERATE,
        (_, Biome::Desert) => DESERT,
        (_, Biome::Snow) => SNOWY,
        (_, Biome::Volcanic) => VOLCANIC,
    }
}

/// The Material at `depth` Voxels below the ground in `strata`: deep rock below them.
fn at_depth(strata: &Strata, depth: f64) -> Material {
    let layer = strata.iter().find(|&&(end, _)| depth < f64::from(end));
    layer.map_or(DEEP_ROCK, |&(_, material)| material)
}

/// The Material if everything from depth `shallow` to `deep` in `strata` is one.
fn one_layer(strata: &Strata, shallow: f64, deep: f64) -> Option<Material> {
    let material = at_depth(strata, shallow);
    (at_depth(strata, deep) == material
        && strata
            .iter()
            .all(|&(end, _)| !(shallow..deep).contains(&f64::from(end))))
    .then_some(material)
}

/// The interior's Material at `radius` Voxels from the centre, if it is within the interior.
fn interior(radius: f64) -> Option<Material> {
    INTERIOR
        .iter()
        .find(|&&(outer, _)| radius <= f64::from(outer))
        .map(|&(_, material)| material)
}

/// The sea's surface, as a distance in Voxels from the centre.
const SEA_RADIUS: f64 = (WORLD_RADIUS + SEA_LEVEL as i32) as f64;

/// Thickness of the ice on the top of polar seas, those the world map marks as snow, in Voxels:
/// 60 cm.
const ICE_THICKNESS: f64 = 6.0;

/// An upper bound on how far a Voxel of a brick is from the middle of its Voxels: the Voxels are
/// at their minimum corners, 0 to 7 from the brick's origin, so the true figure is
/// 3.5 * sqrt(3) = 6.06.
const REACH: f64 = 6.5;

/// Slack, in Voxels, so that rounding in the bounds on a brick's depths cannot flip a decision.
const SLACK: f64 = 1e-6;

/// Heights above sea level, in Voxels, a spawn's ground may be at: 20 to 100 m.
const SPAWN_ELEVATIONS: std::ops::RangeInclusive<i16> = 200..=1_000;

/// Most the ground may rise or fall across a spawn's region, in Voxels: 40 m. More would make
/// the region a lot of bricks.
const GENTLE: f64 = 400.0;

/// Two unit vectors along the ground where `up`, a unit vector away from the centre, points: the
/// way ahead, which is towards the +x axis (towards +y at the poles of x), and the way across it.
pub fn tangents(up: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let axis = if up[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let along: f64 = (0..3).map(|a| axis[a] * up[a]).sum();
    let ahead: [f64; 3] = std::array::from_fn(|a| axis[a] - up[a] * along);
    let ahead = ahead.map(|c| c / distance(ahead));
    let aside = [
        ahead[1] * up[2] - ahead[2] * up[1],
        ahead[2] * up[0] - ahead[0] * up[2],
        ahead[0] * up[1] - ahead[1] * up[0],
    ];
    (ahead, aside)
}

#[cfg(test)]
thread_local! {
    /// How many times this thread has been through [`Planet::generate`]; tests count them.
    static BRICKS_MADE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// How many bricks this thread has had to work out Voxel by Voxel ([`Planet::evaluate`]).
    static EVALUATED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Where the Voxels of a brick of some mip level are sampled. A Voxel of mip `level` is
/// `2^level` Voxels of level 0 along each edge, and is sampled where those Voxels' middle is:
/// at level 0 a Voxel is sampled at its own coordinate, as a Voxel is measured at its minimum
/// corner.
#[derive(Clone, Copy)]
struct Samples {
    /// The point sampled for the brick's Voxel at (0, 0, 0).
    first: [f64; 3],
    /// How far apart the points are along an axis: the edge of a Voxel, in Voxels of level 0.
    size: f64,
}

impl Samples {
    /// The samples of the brick at `coord` of mip `level`, which is below 32.
    fn new(level: usize, coord: BrickCoord) -> Samples {
        debug_assert!(level < 32);
        let size = f64::from(1u32 << level);
        let width = f64::from(BRICK_SIZE) * size;
        Samples {
            first: coord.0.map(|c| f64::from(c) * width + (size - 1.0) / 2.0),
            size,
        }
    }

    /// The point sampled for the Voxel at `local`.
    fn at(&self, local: [u8; 3]) -> [f64; 3] {
        std::array::from_fn(|a| self.first[a] + f64::from(local[a]) * self.size)
    }

    /// The point sampled for the corner of the brick that is `far` (0 or 1) along each axis.
    fn corner(&self, far: [usize; 3]) -> [f64; 3] {
        std::array::from_fn(|a| self.first[a] + (far[a] * 7) as f64 * self.size)
    }

    /// The least and the greatest distance from the centre of any point in the box that holds the
    /// points sampled. They are exact, not rounded off: the distance only grows with each
    /// coordinate's size, so it is least at the point of the box nearest the centre, and greatest
    /// at a corner.
    fn radii(&self) -> (f64, f64) {
        radii_of_box(self.corner([0; 3]), self.corner([1; 3]))
    }

    /// The corners of the box that holds the points sampled and the ones beside them, which are
    /// one Voxel further out along an axis: those of the Voxels around the brick, and the ones
    /// in it.
    fn corners_with_neighbours(&self) -> [[f64; 3]; 8] {
        let low = self.first.map(|f| f - self.size);
        let high = self.first.map(|f| f + 8.0 * self.size);
        std::array::from_fn(|i| {
            std::array::from_fn(|a| if i >> a & 1 == 0 { low[a] } else { high[a] })
        })
    }
}

/// The least and the greatest distance from the centre of any point in the box between `low`
/// and `high`.
fn radii_of_box(low: [f64; 3], high: [f64; 3]) -> (f64, f64) {
    let near = distance(std::array::from_fn(|a| 0f64.clamp(low[a], high[a])));
    let far = distance(std::array::from_fn(|a| low[a].abs().max(high[a].abs())));
    (near, far)
}

/// The Planet, as bricks. Cloning shares the world map, which carries the seed.
#[derive(Clone)]
pub struct Planet {
    map: Arc<WorldMap>,
}

impl fmt::Debug for Planet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Planet ({0}x{0} world map)", self.map.size())
    }
}

impl Planet {
    pub fn new(map: WorldMap) -> Planet {
        Planet { map: Arc::new(map) }
    }

    pub fn map(&self) -> &WorldMap {
        &self.map
    }

    /// The Voxel at `coord`: `None` for air.
    pub fn voxel(&self, coord: VoxelCoord) -> Option<Material> {
        self.material_at(coord.0.map(f64::from))
    }

    /// The Material at `point`, in Voxels from the centre: `None` for air.
    fn material_at(&self, point: [f64; 3]) -> Option<Material> {
        let radius = distance(point);
        if let Some(material) = interior(radius) {
            return Some(material);
        }
        let elevation = self.map.elevation_at(point);
        let depth = SEA_RADIUS + elevation - radius;
        if depth <= 0.0 {
            // Above the ground: water below sea level, else air, and ice on a polar sea.
            if radius >= SEA_RADIUS {
                return None;
            }
            let frozen =
                radius >= SEA_RADIUS - ICE_THICKNESS && self.map.sample(point).biome == Biome::Snow;
            return Some(if frozen { ICE } else { WATER });
        }
        let strata = strata(self.map.sample(point).biome, elevation < 0.0);
        Some(at_depth(strata, depth))
    }

    /// Generates the brick at `coord`, from nothing but the world map and the coordinate.
    ///
    /// Most bricks are decided without looking at their Voxels: those in the interior by their
    /// distance from the centre, those above the highest peak, and the rest by bounds on the
    /// ground's height under the brick, from the few texels of the world map that it is over.
    /// Those say whether the brick is wholly sky, sea, deep rock or one stratum. A test checks
    /// that this agrees with [`Planet::voxel`] for every Voxel, which only a brick over a
    /// boundary between layers, or over the ground itself, needs.
    pub fn brick(&self, coord: BrickCoord) -> Brick {
        self.generate(0, coord)
    }

    /// Generates the brick at `coord` of mip `level` (at least 1) directly, not from the 8 to the
    /// power of `level` bricks of level 0 under it: each of its Voxels is the Planet at the
    /// middle of the `2^level` cubed Voxels it stands for (see [`Samples`]), except the exposed
    /// ones. A solid Voxel with air at the middle of the Voxel next to it, on a face, takes the
    /// Material at the top of the ground within it instead, as the pyramid's coarse Voxel takes
    /// that of its exposed children: otherwise the Voxel with the ground in it reads as what is
    /// deep under it. Whether a Voxel is solid does not change, and level 0 is not affected.
    ///
    /// That is approximately what downsampling those makes of the Planet (ADR 0002), and what
    /// the pyramid of a store reads where it holds nothing, in the cost of a level 0 brick: tests
    /// measure how closely, level by level. The two are the same where the Planet is uniform,
    /// which is nearly everywhere; they differ in the Voxels that a boundary between layers, or
    /// the ground itself, passes through. Like [`Planet::brick`] it is a function of the world
    /// map and the coordinate alone, and decides most bricks, the sky and the interior's,
    /// without looking at their Voxels.
    pub fn coarse_brick(&self, level: usize, coord: BrickCoord) -> Brick {
        debug_assert!(level > 0);
        self.generate(level, coord)
    }

    /// The brick at `coord` of mip `level`: [`Planet::brick`] and [`Planet::coarse_brick`].
    fn generate(&self, level: usize, coord: BrickCoord) -> Brick {
        #[cfg(test)]
        BRICKS_MADE.with(|made| made.set(made.get() + 1));
        let samples = Samples::new(level, coord);
        let (near, far) = samples.radii();
        if near >= f64::from(MAX_RADIUS) {
            return Brick::AIR;
        }
        // A Voxel of a coarse brick with air beside it takes the Material at the ground's surface
        // (see [`Planet::evaluate`]), which no shortcut that gives a whole brick the Material of
        // one layer knows. Those hold where no Voxel has air beside it. The sky and the sea
        // below are no layer, and have nothing to expose.
        let sealed = || level == 0 || self.sealed(&samples);
        if far <= f64::from(INTERIOR_RADIUS)
            && sealed()
            && let Some(brick) = self.interior_brick(&samples)
        {
            return brick;
        }
        let corners: [[f64; 3]; 8] =
            std::array::from_fn(|i| samples.corner([i & 1, i >> 1 & 1, i >> 2]));
        // Every Voxel is in the hull of the corners, so the ground under it is within these.
        let (low, high) = self.map.elevation_range(&corners);
        // The brick's Voxels are between these many Voxels below the ground.
        let shallowest = SEA_RADIUS + low - far - SLACK;
        let deepest = SEA_RADIUS + high - near + SLACK;
        if deepest <= 0.0 {
            // All above the ground: sky, or sea if it is below sea level.
            if low >= 0.0 || near >= SEA_RADIUS {
                return Brick::AIR;
            }
            // All water, unless it reaches the ice on a polar sea.
            if far < SEA_RADIUS
                && (far < SEA_RADIUS - ICE_THICKNESS
                    || self
                        .single_texel(&corners)
                        .is_some_and(|texel| texel.biome != Biome::Snow))
            {
                return Brick::solid(WATER);
            }
        } else if sealed() {
            if shallowest >= f64::from(STRATA_DEPTH) {
                return Brick::solid(DEEP_ROCK);
            }
            if shallowest > 0.0 && (low >= 0.0 || high < 0.0) {
                // Below the ground, all land or all sea floor, and one stratum if it is all in
                // one biome, which it is if the corners are in one texel.
                if let Some(texel) = self.single_texel(&corners) {
                    let strata = strata(texel.biome, high < 0.0);
                    if let Some(material) = one_layer(strata, shallowest, deepest) {
                        return Brick::solid(material);
                    }
                }
            }
        }
        self.evaluate(&samples)
    }

    /// Whether no Voxel of the brick sampled at `samples` can have air beside it, on one of its
    /// six faces. See [`Planet::air_floor`].
    fn sealed(&self, samples: &Samples) -> bool {
        self.air_floor(samples).is_none()
    }

    /// The least distance from the centre that air beside a Voxel of the brick sampled at
    /// `samples` can be at, or `None` if there is none because the brick is too far in. Air is at
    /// least [`SEA_RADIUS`] from the centre, and above the ground, which is at least the lowest
    /// of it under the box that holds the brick and the points one Voxel's width beyond it. So
    /// there is none beside a Voxel if every one of those points is nearer than that.
    fn air_floor(&self, samples: &Samples) -> Option<f64> {
        // A point beside a Voxel is a Voxel's width from it.
        let far = samples.radii().1 + samples.size;
        if far + SLACK < SEA_RADIUS {
            return None;
        }
        let (lowest, _) = self.map.elevation_range(&samples.corners_with_neighbours());
        let floor = SEA_RADIUS + lowest.max(0.0);
        (far + SLACK >= floor).then_some(floor)
    }

    /// The brick sampled at `samples` if it lies within the interior, where it takes no more than
    /// its distance from the centre to say what it is (and 512 of them at the edge of the core).
    fn interior_brick(&self, samples: &Samples) -> Option<Brick> {
        let (near, far) = samples.radii();
        if far > f64::from(INTERIOR_RADIUS) {
            return None;
        }
        let mut inner = f64::NEG_INFINITY;
        for &(outer, material) in &INTERIOR {
            if near > inner && far <= f64::from(outer) {
                return Some(Brick::solid(material));
            }
            inner = f64::from(outer);
        }
        Some(self.evaluate(samples))
    }

    /// The texel that all of a brick's corners are in, if they are in one. Then so is the whole
    /// brick, as a texel is a box in the face's cube coordinates and a brick is convex.
    fn single_texel(&self, corners: &[[f64; 3]; 8]) -> Option<Texel> {
        let at = self.map.locate(corners[0]);
        (corners.iter().all(|&c| self.map.locate(c) == at))
            .then(|| self.map.texel(at.0, at.1, at.2))
    }

    /// The brick sampled at `samples`, every Voxel worked out.
    ///
    /// A Voxel of a coarse brick is the Planet at its middle, except where it is exposed: solid,
    /// with air beside it on a face, at the middle of the Voxel next to it. Then the middle may
    /// be deep in the strata, with the ground in the upper part of the Voxel that is what is seen
    /// of it, so it takes the Material at the shallowest depth the Voxel reaches. That is the
    /// choice the pyramid makes (ADR 0002). Whether a Voxel is solid is not changed.
    fn evaluate(&self, samples: &Samples) -> Brick {
        #[cfg(test)]
        EVALUATED.with(|made| made.set(made.get() + 1));
        let mut brick = Brick::AIR;
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let local = [x, y, z];
                    brick.set(local, self.material_at(samples.at(local)));
                }
            }
        }
        if samples.size > 1.0
            && let Some(floor) = self.air_floor(samples)
        {
            self.expose(samples, floor, &mut brick);
        }
        brick
    }

    /// Gives the Voxels of `brick`, sampled at `samples`, that are exposed the Material at the
    /// ground's surface within them (see [`Planet::evaluate`]). `floor` is the [`Planet::air_floor`].
    fn expose(&self, samples: &Samples, floor: f64, brick: &mut Brick) {
        let size = samples.size;
        // Solidity as sampled, which the Materials set below leave as it is.
        let solid = brick.clone();
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let local = [x, y, z];
                    if solid.get(local).is_none() {
                        continue;
                    }
                    let point = samples.at(local);
                    // A Voxel's neighbour is a Voxel's width away.
                    if distance(point) + size + SLACK < floor {
                        continue;
                    }
                    let exposed = (0..6).any(|face| {
                        let step = if face % 2 == 0 { -1 } else { 1 };
                        let mut next = local.map(i32::from);
                        next[face / 2] += step;
                        match next[face / 2] {
                            // Within the brick, as sampled.
                            0..8 => solid.get(next.map(|c| c as u8)).is_none(),
                            // In the next brick over.
                            _ => {
                                let mut beside = point;
                                beside[face / 2] += f64::from(step) * size;
                                self.material_at(beside).is_none()
                            }
                        }
                    });
                    if exposed && let Some(surface) = self.top_within(point, size) {
                        brick.set(local, Some(surface));
                    }
                }
            }
        }
    }

    /// The Material at the top of a Voxel whose middle at `point` is below the ground, which is
    /// what is seen of it from above: the Voxel is `size` Voxels wide, and reaches half of that
    /// towards the sky along the way the ground faces, which is away from the centre. That is
    /// half its width only if it faces along an axis, and up to sqrt(3) times more across the
    /// corner of a cube. If it reaches into the air, the top is the surface: the ground's, or the
    /// sea's if the ground is under it. `None` if the middle is not below the ground.
    fn top_within(&self, point: [f64; 3], size: f64) -> Option<Material> {
        let radius = distance(point);
        let elevation = self.map.elevation_at(point);
        let ground = SEA_RADIUS + elevation;
        if ground <= radius || radius == 0.0 {
            return None;
        }
        let lean = (point[0].abs() + point[1].abs() + point[2].abs()) / radius;
        // The point of the Voxel's axis at `r` from the centre.
        let on_axis = |r: f64| point.map(|c| c * r / radius);
        let top = on_axis(radius + size / 2.0 * lean);
        // In the air, a little below the highest ground or sea there.
        self.material_at(top)
            .or_else(|| self.material_at(on_axis(ground.max(SEA_RADIUS) - 0.5)))
    }

    /// Bounds `(lowest, highest)` on the ground's height above sea level within `reach` Voxels
    /// of the line from the centre through `point`, which is near the ground. The whole Planet's
    /// bounds if that is not over one face of the world map.
    fn heights_near(&self, point: [f64; 3], reach: i32) -> (f64, f64) {
        let up = point.map(|c| c / distance(point));
        let (ahead, aside) = tangents(up);
        // A square that holds the circle, with room for a brick, at the top of the ground. The
        // region is a straight cylinder, so its directions fan out a little with height, but the
        // ground's top is only about 1.003 times as far out as the bottom of the strata, so the
        // factor of 1.1 is generous slack.
        let half = f64::from(reach + 2 * BRICK_SIZE) * 1.1;
        let corners = [(-half, -half), (half, -half), (-half, half), (half, half)]
            .map(|(s, t)| std::array::from_fn(|a| point[a] + ahead[a] * s + aside[a] * t));
        self.map.elevation_range(&corners)
    }

    /// A point on the ground to spawn on: gentle, temperate land if there is any, else gentle land
    /// of any kind, else the land that varies least, else the highest ground, so that every
    /// world map has one. Gentle, temperate land is the first, going through the world map's
    /// texels on a lattice, where the ground is 20 to 100 m above the sea and varies by 40 m at
    /// most within `reach` Voxels (see [`Planet::region_around`]). The same for the same map.
    pub fn spawn(&self, reach: i32) -> [f64; 3] {
        let size = self.map.size();
        let stride = (size / 32).max(1) as usize;
        let spots = || {
            (0..6).flat_map(move |face| {
                (stride / 2..size as usize)
                    .step_by(stride)
                    .flat_map(move |j| {
                        (stride / 2..size as usize)
                            .step_by(stride)
                            .map(move |i| (face, i as u32, j as u32))
                    })
            })
        };
        let ground = |&(face, i, j): &(usize, u32, u32)| {
            let texel = self.map.texel(face, i, j);
            let direction = self.map.texel_direction(face, i, j);
            (
                texel,
                direction.map(|d| d * (SEA_RADIUS + f64::from(texel.elevation))),
            )
        };
        let relief = |point: [f64; 3]| {
            let (low, high) = self.heights_near(point, reach);
            high - low
        };
        let land = |texel: &Texel| texel.elevation >= SEA_LEVEL;
        spots()
            .map(|spot| ground(&spot))
            .find(|(texel, point)| {
                texel.biome == Biome::Temperate
                    && SPAWN_ELEVATIONS.contains(&texel.elevation)
                    && relief(*point) <= GENTLE
            })
            .or_else(|| {
                (spots().map(|spot| ground(&spot)))
                    .find(|(texel, point)| land(texel) && relief(*point) <= GENTLE)
            })
            .or_else(|| {
                (spots()
                    .map(|spot| ground(&spot))
                    .filter(|(texel, _)| land(texel)))
                .min_by(|a, b| relief(a.1).total_cmp(&relief(b.1)))
            })
            .or_else(|| {
                (spots().map(|spot| ground(&spot))).max_by_key(|(texel, _)| texel.elevation)
            })
            .expect("a world map has texels")
            .1
    }

    /// A store of the ground within `reach` Voxels of the line from the centre through `spawn`
    /// (see [`Planet::spawn`]): every brick that is not air, from the bottom of the strata
    /// there to above the highest ground or the sea. Where it holds no brick it reads as the
    /// Planet's own at level 0 inside and below the region, and as air elsewhere, in constant
    /// time (see [`RegionSource`]): so the deep rock under the region costs a flag, not memory,
    /// and the ground just outside it is not there until it is streamed (#49).
    pub fn region_around(&self, spawn: [f64; 3], reach: i32) -> BrickStore {
        let (low, high) = self.heights_near(spawn, reach);
        let up = spawn.map(|c| c / distance(spawn));
        // Along `up`, from two bricks below the lowest strata (the sphere curves up at the sides
        // of the cylinder, by a few Voxels) to a brick above the highest ground or the sea;
        // across it, `reach` and a brick.
        let bottom = SEA_RADIUS + low - f64::from(STRATA_DEPTH + 2 * BRICK_SIZE);
        let top = SEA_RADIUS + high.max(0.0) + f64::from(BRICK_SIZE);
        let across = f64::from(reach + BRICK_SIZE) + REACH;
        let ends = [bottom, top].map(|a| up.map(|u| u * a));
        let corner = |pick: fn(f64, f64) -> f64, margin: f64| {
            BrickCoord(std::array::from_fn(|a| {
                ((pick(ends[0][a], ends[1][a]) + margin) / f64::from(BRICK_SIZE)).floor() as i32
            }))
        };
        let (min, max) = (corner(f64::min, -across), corner(f64::max, across));
        let source = RegionSource {
            planet: self.clone(),
            up,
            bottom,
            across,
        };
        let found = par_bricks(min, max, |coord| {
            let (along, sideways) = source.place(coord);
            if along < bottom || along > top || sideways > across * across {
                return None;
            }
            Some(self.brick(coord)).filter(|brick| !brick.is_air())
        });
        let mut store = BrickStore::new();
        store.set_source(source);
        for (coord, brick) in found {
            store.insert(coord, brick);
        }
        store
    }
}

/// Whether the brick at `coord` of mip `level` is wholly in the interior, which is solid all the
/// way down: what both sources of a store say to [`BrickSource::solid_below`].
fn inside_interior(level: usize, coord: BrickCoord) -> bool {
    coord.within_ball_at(level, INTERIOR_RADIUS)
}

/// What a store reads where it holds no brick, when the Planet is to be drawn at every distance
/// (the LOD bands, #84): the Planet itself, generated on demand at the level asked for. A brick
/// of level 0 is [`Planet::brick`] and a coarse one is [`Planet::coarse_brick`], so the ground
/// far from where the store holds bricks is drawn at the level its distance asks for, and
/// costs a brick's worth of Voxels for each brick that holds ground. Measured on the client's
/// first view of the full-size Planet (#84): 3.0 million bricks, 228 s on eight cores, so it is
/// for a streaming loader (#86, #49) to ask for, not for a frame to wait on. Nothing calls it yet
/// but its test: it is here for that loader.
///
/// A store holds the bricks that Edits changed and the region around the spawn; everything else
/// is this. Unedited bricks are a cache that can be regenerated from the seed (ADR 0002).
#[derive(Clone, Debug)]
pub struct PlanetSource {
    planet: Planet,
}

impl PlanetSource {
    pub fn new(planet: Planet) -> Self {
        Self { planet }
    }
}

impl BrickSource for PlanetSource {
    fn brick(&self, level: usize, coord: BrickCoord) -> Brick {
        if level == 0 {
            self.planet.brick(coord)
        } else {
            self.planet.coarse_brick(level, coord)
        }
    }

    fn solid_below(&self, level: usize, coord: BrickCoord) -> bool {
        inside_interior(level, coord)
    }
}

/// What the store of [`Planet::region_around`] reads where it holds no brick. The store holds
/// every brick of the region that is not air, so a brick it does not hold is one of these, each
/// answered by a distance or two and never by generating anything (walking moves the renderer's
/// rings over hundreds of thousands of them at a time):
/// - at level 0, a brick that lies within the interior is the Planet's own, which is a flag
///   except where it is over the edge of the core;
/// - at level 0, a brick below the region (along the cylinder, under its bottom) is deep rock,
///   as the Planet's strata end above it;
/// - any other brick is air, though the Planet has ground beyond the region;
/// - at the coarse levels, a brick wholly within one layer of the interior is that layer, and
///   every other brick is air.
///
/// So only level 0 and level 1 are right within the region. From level 2 up, a coarse brick is
/// made from level 1 bricks that the store does not hold wherever the region has no brick under
/// them, and this reads them as air where the Planet has ground (the region's sides and bottom
/// edge, for instance), so such a brick downsamples air in place of ground. The Planet's ground
/// at the coarse levels exists as [`Planet::coarse_brick`], but costs a brick's worth of Voxels
/// for each brick that holds ground, far too much for the rings to pay as they move: streaming
/// it in on worker threads (#49) is what gives the coarse rings real terrain.
#[derive(Debug)]
struct RegionSource {
    planet: Planet,
    /// The unit vector along the cylinder of the region, away from the centre.
    up: [f64; 3],
    /// Distance along `up` of the bottom of the region.
    bottom: f64,
    /// Distance from the axis of the edge of the region, counting the reach of a brick.
    across: f64,
}

impl RegionSource {
    /// The distance of the middle of the brick at `coord` along the axis, and the square of its
    /// distance from it.
    fn place(&self, coord: BrickCoord) -> (f64, f64) {
        let centre = coord.origin().0.map(|o| f64::from(o) + 3.5);
        let along: f64 = (0..3).map(|a| centre[a] * self.up[a]).sum();
        (
            along,
            centre.iter().map(|c| c * c).sum::<f64>() - along * along,
        )
    }
}

impl BrickSource for RegionSource {
    fn brick(&self, level: usize, coord: BrickCoord) -> Brick {
        if level == 0 {
            if let Some(brick) = self.planet.interior_brick(&Samples::new(0, coord)) {
                return brick;
            }
            let (along, sideways) = self.place(coord);
            return if along < self.bottom && sideways <= self.across * self.across {
                Brick::solid(DEEP_ROCK)
            } else {
                Brick::AIR
            };
        }
        INTERIOR
            .iter()
            .find(|&&(outer, _)| coord.within_ball_at(level, outer))
            .map_or(Brick::AIR, |&(_, material)| Brick::solid(material))
    }

    fn solid_below(&self, level: usize, coord: BrickCoord) -> bool {
        inside_interior(level, coord)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_map::{Biome, SEA_LEVEL, Texel};
    use std::sync::OnceLock;

    const SEED: u64 = 0xB4A7_A9AC;
    /// Texels along a face of the world map the tests share.
    const SIZE: u32 = 512;
    /// Voxels around the spawn the tests take: the world map of the tests is too coarse for more.
    const SPAWN_REACH: i32 = 100;

    fn planet() -> &'static Planet {
        static PLANET: OnceLock<Planet> = OnceLock::new();
        PLANET.get_or_init(|| Planet::new(WorldMap::generate(SEED, SIZE)))
    }

    /// The Voxel `radius` Voxels from the centre along `direction`.
    fn along(direction: [f64; 3], radius: f64) -> VoxelCoord {
        let length = direction.iter().map(|d| d * d).sum::<f64>().sqrt();
        VoxelCoord(direction.map(|d| (d / length * radius).round() as i32))
    }

    /// The direction of the middle of the first texel for which `pick` holds.
    fn find(pick: impl Fn(&Texel) -> bool) -> [f64; 3] {
        let map = planet().map();
        (0..6)
            .flat_map(|f| {
                (0..map.size()).flat_map(move |j| (0..map.size()).map(move |i| (f, i, j)))
            })
            .find(|&(f, i, j)| pick(&map.texel(f, i, j)))
            .map(|(f, i, j)| map.texel_direction(f, i, j))
            .expect("the test map has such a texel")
    }

    /// The Voxels of a column along `direction`, from `from` to `to` Voxels above sea level, as
    /// runs of equal ones, bottom first, with the height where each starts. Rounding a point to
    /// a Voxel moves it up to a Voxel off the column, which can flip a Voxel next to a boundary
    /// between layers, so a single Voxel unlike both its neighbours is taken to be like them.
    fn column(direction: [f64; 3], from: f64, to: f64) -> Vec<(f64, Option<Material>)> {
        let heights: Vec<f64> = (0..)
            .map(|k| from + f64::from(k))
            .take_while(|&h| h <= to)
            .collect();
        let mut voxels: Vec<_> = heights
            .iter()
            .map(|&h| planet().voxel(along(direction, SEA_RADIUS + h)))
            .collect();
        for k in 1..voxels.len() - 1 {
            if voxels[k - 1] == voxels[k + 1] {
                voxels[k] = voxels[k - 1];
            }
        }
        let mut runs: Vec<(f64, Option<Material>)> = Vec::new();
        for (h, voxel) in heights.into_iter().zip(voxels) {
            if runs.last().map(|r| r.1) != Some(voxel) {
                runs.push((h, voxel));
            }
        }
        runs
    }

    fn voxels_of(coord: BrickCoord) -> impl Iterator<Item = ([u8; 3], VoxelCoord)> {
        (0..512usize).map(move |i| {
            let local = [(i % 8) as u8, (i / 8 % 8) as u8, (i / 64) as u8];
            (local, coord.voxel(local))
        })
    }

    /// Worlds already generated hold these ids; changing what they mean needs a
    /// `GENERATOR_VERSION` bump and new golden hashes.
    #[test]
    fn the_planet_constants_are_materials_of_the_shared_table() {
        let table = materials::MaterialTable::built_in();
        for (material, name) in [
            (SOIL, "soil"),
            (GRASSY_SOIL, "grassy soil"),
            (CLAY, "clay"),
            (GRAVEL, "gravel"),
            (SAND, "sand"),
            (SANDSTONE, "sandstone"),
            (LIMESTONE, "limestone"),
            (SHALE, "shale"),
            (GRANITE, "granite"),
            (BASALT, "basalt"),
            (OBSIDIAN, "obsidian"),
            (DEEP_ROCK, "deep rock"),
            (SNOW, "snow"),
            (ICE, "ice"),
            (MOLTEN_CORE, "molten core"),
            (WATER, "water"),
        ] {
            assert_eq!(table.get(material).map(|d| d.name.as_str()), Some(name));
        }
    }

    #[test]
    fn the_core_glows_and_the_ocean_is_water() {
        let table = materials::MaterialTable::built_in();
        assert!(table.get(MOLTEN_CORE).unwrap().emissive_intensity > 1.0);
        assert_eq!(table.get(WATER).unwrap().fluid, materials::FluidKind::Water);
    }

    #[test]
    fn every_biome_has_strata_that_end_where_the_deep_rock_begins() {
        for strata in [TEMPERATE, DESERT, SNOWY, VOLCANIC, SEABED] {
            assert!(strata.windows(2).all(|w| w[0].0 < w[1].0));
            assert_eq!(strata.last().unwrap().0, STRATA_DEPTH);
        }
        // Nothing the world map decides is inside the interior.
        assert_eq!(INTERIOR_RADIUS, 149_520);
        const { assert!(MOLTEN_RADIUS < INTERIOR_RADIUS) };
    }

    #[test]
    fn the_centre_is_molten_then_deep_rock_going_out() {
        let at = |x: i32| planet().voxel(VoxelCoord([x, 0, 0]));
        assert_eq!(at(0), Some(MOLTEN_CORE));
        assert_eq!(at(MOLTEN_RADIUS), Some(MOLTEN_CORE));
        assert_eq!(at(MOLTEN_RADIUS + 1), Some(DEEP_ROCK));
        assert_eq!(at(INTERIOR_RADIUS), Some(DEEP_ROCK));
    }

    #[test]
    fn deep_bricks_are_uniform_and_cost_a_flag() {
        for (c, want) in [
            ([0, 0, 0], MOLTEN_CORE),
            ([-1, -1, -1], MOLTEN_CORE),
            ([2000, 0, 0], MOLTEN_CORE),
            ([0, -6000, 0], DEEP_ROCK),
            ([9000, 9000, 9000], DEEP_ROCK),
            ([0, 0, 17_000], DEEP_ROCK),
        ] {
            let brick = planet().brick(BrickCoord(c));
            assert!(brick.is_uniform(), "{c:?}");
            assert_eq!(brick.get([3, 3, 3]), Some(want), "{c:?}");
            assert_eq!(brick.storage_bytes(), 0, "{c:?}");
        }
    }

    #[test]
    fn a_column_on_temperate_land_goes_deep_rock_shale_limestone_clay_soil_grass_then_air() {
        let d = find(|t| t.biome == Biome::Temperate && (300..600).contains(&t.elevation));
        let surface = planet().map().elevation_at(d);
        let runs = column(d, surface - 520.0, surface + 20.0);
        let found: Vec<_> = runs.iter().map(|r| r.1).collect();
        let want = [DEEP_ROCK, SHALE, LIMESTONE, CLAY, SOIL, GRASSY_SOIL].map(Some);
        assert_eq!(found, [want.as_slice(), &[None]].concat());
        // Where each begins, in Voxels below the surface.
        for (run, depth) in runs[1..].iter().zip([480.0, 300.0, 120.0, 40.0, 4.0, 0.0]) {
            assert!((surface - run.0 - depth).abs() <= 2.0, "{run:?} {depth}");
        }
    }

    #[test]
    fn the_surface_follows_the_biome() {
        let top = |pick: fn(&Texel) -> bool| {
            let d = find(pick);
            let surface = planet().map().elevation_at(d);
            column(d, surface - 2.0, surface + 3.0)[0].1
        };
        assert_eq!(
            top(|t| t.biome == Biome::Temperate && t.elevation >= 100),
            Some(GRASSY_SOIL)
        );
        assert_eq!(
            top(|t| t.biome == Biome::Desert && t.elevation >= 100),
            Some(SAND)
        );
        assert_eq!(
            top(|t| t.biome == Biome::Snow && t.elevation >= 100),
            Some(SNOW)
        );
        assert_eq!(
            top(|t| t.biome == Biome::Volcanic && t.elevation >= 100),
            Some(BASALT)
        );
    }

    #[test]
    fn the_strata_below_the_surface_differ_by_biome() {
        let below = |biome| {
            let d = find(|t| t.biome == biome && t.elevation >= 100);
            let surface = planet().map().elevation_at(d);
            column(d, surface - 520.0, surface - 1.0)
                .iter()
                .map(|r| r.1.unwrap())
                .collect::<Vec<_>>()
        };
        let all = [
            Biome::Temperate,
            Biome::Desert,
            Biome::Snow,
            Biome::Volcanic,
        ]
        .map(below);
        for (k, a) in all.iter().enumerate() {
            assert_eq!(a[0], DEEP_ROCK);
            assert!(a.len() >= 5, "{a:?}");
            assert!(all[k + 1..].iter().all(|b| b != a), "{a:?}");
        }
    }

    #[test]
    fn an_ocean_column_is_seabed_then_water_up_to_sea_level_then_air() {
        let d = find(|t| t.elevation < -400);
        let surface = planet().map().elevation_at(d);
        let runs = column(d, surface - 3.0, 30.0);
        let found: Vec<_> = runs.iter().map(|r| r.1).collect();
        assert_eq!(found, [Some(SAND), Some(WATER), None]);
        // The water reaches sea level: the first air is there.
        assert!((runs[2].0 - f64::from(SEA_LEVEL)).abs() <= 2.0, "{runs:?}");
    }

    #[test]
    fn land_is_not_flooded_and_the_sea_floor_is_under_water() {
        let map = planet().map();
        for face in 0..6 {
            for j in (0..map.size()).step_by(5) {
                for i in (0..map.size()).step_by(5) {
                    let texel = map.texel(face, i, j);
                    let d = map.texel_direction(face, i, j);
                    if texel.elevation >= 0 {
                        // Just under the surface, and at sea level itself where it is higher.
                        for h in [f64::from(texel.elevation) - 2.0, -2.0, 0.0] {
                            let voxel = planet().voxel(along(d, SEA_RADIUS + h));
                            assert!(voxel != Some(WATER), "{face} {i} {j} {h}");
                        }
                    } else if texel.elevation < -20 {
                        let sea = planet().voxel(along(d, SEA_RADIUS - 9.0));
                        assert_eq!(sea, Some(WATER), "{face} {i} {j}");
                    }
                }
            }
        }
    }

    #[test]
    fn ocean_bricks_below_sea_level_are_water() {
        let d = find(|t| t.elevation < -400);
        let coord = along(d, SEA_RADIUS - 100.0).brick();
        let brick = planet().brick(coord);
        assert!(brick.is_uniform());
        assert_eq!(brick.get([0, 0, 0]), Some(WATER));
    }

    /// Directions from the centre through every kind of country, to take radial lines down.
    fn lines() -> Vec<[f64; 3]> {
        let mut lines = vec![[1.0, 0.0, 0.0], [0.3, -0.5, 0.8], [-0.2, -0.9, -0.4]];
        for pick in [
            (|t: &Texel| t.biome == Biome::Desert) as fn(&Texel) -> bool,
            |t| t.biome == Biome::Snow && t.elevation > 0,
            |t| t.biome == Biome::Volcanic,
            |t| t.elevation < -300,
            |t| (0..40).contains(&t.elevation),
            |t| t.elevation > 8000,
        ] {
            lines.push(find(pick));
        }
        lines
    }

    #[test]
    fn a_brick_is_what_its_voxels_are_wherever_it_is() {
        // Down radial lines, to the granite below the strata.
        for d in lines() {
            let surface = planet().map().elevation_at(d);
            let mut h = surface - 560.0;
            while h < surface.max(0.0) + 40.0 {
                let coord = along(d, SEA_RADIUS + h).brick();
                let brick = planet().brick(coord);
                for (local, voxel) in voxels_of(coord) {
                    assert_eq!(
                        brick.get(local),
                        planet().voxel(voxel),
                        "{d:?} {coord:?} {local:?}"
                    );
                }
                h += 5.0;
            }
        }
    }

    #[test]
    fn bricks_over_the_steepest_ground_and_a_texel_edge_are_what_their_voxels_are() {
        // The pair of neighbouring texels with the largest rise between them, and the middle of
        // them, which is where one texel's biome ends and the next begins.
        let map = planet().map();
        let (mut steepest, mut at) = (0, [0.0; 3]);
        for face in 0..6 {
            for j in 0..map.size() - 1 {
                for i in 0..map.size() - 1 {
                    let (a, b) = (map.texel(face, i, j), map.texel(face, i + 1, j));
                    let rise = (a.elevation - b.elevation).abs();
                    if rise > steepest {
                        let (da, db) = (
                            map.texel_direction(face, i, j),
                            map.texel_direction(face, i + 1, j),
                        );
                        (steepest, at) = (rise, [0, 1, 2].map(|k| da[k] + db[k]));
                    }
                }
            }
        }
        assert!(steepest > 100, "{steepest}");
        let ground = planet().map().elevation_at(at);
        let middle = along(at, SEA_RADIUS + ground).brick();
        for dx in -3..=3 {
            for dy in -3..=3 {
                for dz in -12..=12 {
                    let coord = BrickCoord([middle.0[0] + dx, middle.0[1] + dy, middle.0[2] + dz]);
                    let brick = planet().brick(coord);
                    for (local, voxel) in voxels_of(coord) {
                        assert_eq!(brick.get(local), planet().voxel(voxel), "{coord:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn bricks_across_the_interiors_boundaries_are_what_their_voxels_are() {
        for r in [MOLTEN_RADIUS, INTERIOR_RADIUS, 5] {
            for d in [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-0.3, 0.2, -1.0]] {
                for k in -12..=12 {
                    let coord = along(d, f64::from(r + k)).brick();
                    let brick = planet().brick(coord);
                    for (local, voxel) in voxels_of(coord) {
                        assert_eq!(brick.get(local), planet().voxel(voxel), "{coord:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_brick_is_generated_alone_and_the_same_every_time() {
        let d = find(|t| t.elevation > 100);
        let coord = along(d, SEA_RADIUS + planet().map().elevation_at(d)).brick();
        let a = planet().brick(coord);
        let again = Planet::new(WorldMap::generate(SEED, SIZE)).brick(coord);
        assert_eq!(a.content_hash(), again.content_hash());
        assert!(!a.is_uniform());
        let other = Planet::new(WorldMap::generate(SEED + 1, SIZE)).brick(coord);
        assert_ne!(a.content_hash(), other.content_hash());
    }

    #[test]
    fn far_space_is_air() {
        assert!(planet().brick(BrickCoord([30_000, 0, 0])).is_air());
        assert!(
            planet()
                .brick(BrickCoord([i32::MIN / 8, i32::MAX / 8, 5]))
                .is_air()
        );
    }

    #[test]
    fn the_spawn_is_on_gentle_temperate_land_with_air_above_it() {
        let spawn = planet().spawn(SPAWN_REACH);
        assert!(planet().map().elevation_at(spawn) >= 100.0);
        assert_eq!(planet().map().sample(spawn).biome, Biome::Temperate);
        let radius = spawn.iter().map(|c| c * c).sum::<f64>().sqrt();
        let voxel = |dr| planet().voxel(along(spawn, radius + dr));
        assert!(voxel(-3.0).is_some_and(|m| m != WATER));
        assert_eq!(voxel(3.0), None);
    }

    fn region() -> &'static BrickStore {
        static REGION: OnceLock<BrickStore> = OnceLock::new();
        REGION.get_or_init(|| {
            let spawn = planet().spawn(SPAWN_REACH);
            planet().region_around(spawn, SPAWN_REACH)
        })
    }

    #[test]
    fn the_spawn_has_a_fallback_for_a_planet_without_gentle_temperate_land() {
        // A reach so wide that the ground is no longer over one face of the map says nothing is
        // gentle, and then the land that varies least is the first of it.
        let spawn = planet().spawn(1_000_000);
        assert!(planet().map().elevation_at(spawn) >= 0.0);
        let radius = spawn.iter().map(|c| c * c).sum::<f64>().sqrt();
        let voxel = |dr| planet().voxel(along(spawn, radius + dr));
        assert!(voxel(-3.0).is_some_and(|m| m != WATER));
        assert_eq!(voxel(3.0), None);
    }

    #[test]
    fn the_region_around_the_spawn_holds_the_ground_and_reads_the_rest_as_the_planet() {
        let spawn = planet().spawn(SPAWN_REACH);
        let store = region();
        assert!(!store.is_empty());
        for coord in store.sorted_coords() {
            let brick = store.brick(coord).unwrap();
            assert!(!brick.is_air());
            assert_eq!(brick.content_hash(), planet().brick(coord).content_hash());
        }
        // Down the line through the spawn, and down lines at the edge of the region either side
        // of it, from above the ground to the centre, every brick reads as the Planet's own,
        // held or not.
        let radius = spawn.iter().map(|c| c * c).sum::<f64>().sqrt();
        let up = spawn.map(|c| c / radius);
        let (_, aside) = tangents(up);
        let reach = f64::from(SPAWN_REACH);
        for offset in [0.0, 60.0, -60.0, reach, -reach] {
            for r in (0..=radius as i32 + 60).rev().step_by(37) {
                let point: [f64; 3] =
                    std::array::from_fn(|a| up[a] * f64::from(r) + aside[a] * offset);
                let coord = VoxelCoord(point.map(|c| c.round() as i32)).brick();
                assert_eq!(
                    store.brick_at(0, coord).content_hash(),
                    planet().brick(coord).content_hash(),
                    "{coord:?} at {r}, {offset} across"
                );
            }
        }
        // Deep bricks under the region are not held: they cost a source, not memory.
        let deep = along(spawn, radius - 12_000.0).brick();
        assert!(store.brick(deep).is_none());
        assert_eq!(*store.brick_at(0, deep), Brick::solid(DEEP_ROCK));
        assert_eq!(
            *store.brick_at(0, BrickCoord([0; 3])),
            Brick::solid(MOLTEN_CORE)
        );
        // Beside the region the Planet has ground, but the store does not, until it is streamed.
        let beside = along(spawn, radius).brick().0;
        let far = BrickCoord(std::array::from_fn(|a| {
            beside[a] + (aside[a] * 20.0).round() as i32
        }));
        assert!(store.brick_at(0, far).is_air());
        assert!(!planet().brick(far).is_air());
    }

    #[test]
    fn reading_bricks_the_region_does_not_hold_generates_nothing() {
        // Walking makes the renderer read hundreds of thousands of them at a time, around the
        // camera, so none may be worked out from the world map.
        let spawn = planet().spawn(SPAWN_REACH);
        let store = region();
        let at = along(spawn, 160_000.0).brick().0;
        let before = BRICKS_MADE.with(|made| made.get());
        let mut read = 0;
        for x in -40..=40 {
            for y in -40..=40 {
                for z in -40..=40 {
                    let coord = BrickCoord([at[0] + x, at[1] + y, at[2] + z]);
                    if store.brick(coord).is_none() {
                        std::hint::black_box(store.brick_at(0, coord));
                        read += 1;
                    }
                }
            }
        }
        assert!(read > 100_000, "{read}");
        assert_eq!(BRICKS_MADE.with(|made| made.get()), before);
    }

    #[test]
    fn coarse_bricks_are_the_interiors_layers_where_they_lie_wholly_in_one() {
        let mut store = region().clone();
        store.build_lod(3);
        let at = |level, c| store.brick_at(level, BrickCoord(c)).into_owned();
        // Level 2 bricks are 32 Voxels across.
        assert_eq!(at(2, [100, 0, 0]), Brick::solid(MOLTEN_CORE));
        assert_eq!(at(2, [2000, 0, 0]), Brick::solid(DEEP_ROCK));
        // Over the core's edge (x = 32000 is brick 1000) it is the layer outside; over the
        // interior's edge, and beyond, it is air until coarse bricks are generated directly.
        assert_eq!(at(2, [1000, 0, 0]), Brick::solid(DEEP_ROCK));
        assert!(at(2, [4673, 0, 0]).is_air());
        assert!(at(2, [100_000, 0, 0]).is_air());
        assert_eq!(at(0, [0, 0, 0]), planet().brick(BrickCoord([0, 0, 0])));
    }

    /// The coarse brick above `coord`, `levels` levels up.
    fn ancestor(coord: BrickCoord, levels: usize) -> BrickCoord {
        (0..levels).fold(coord, |c, _| c.parent())
    }

    /// The Voxels of a brick, by position in it.
    fn locals() -> impl Iterator<Item = [u8; 3]> {
        (0..512usize).map(|i| [(i % 8) as u8, (i / 8 % 8) as u8, (i / 64) as u8])
    }

    /// Directions spread over the Planet, to take more radial lines than `lines` has.
    fn spread() -> impl Iterator<Item = [f64; 3]> {
        (1..=16).filter_map(|k: i32| {
            let d = [k * 37 % 17 - 8, k * 53 % 19 - 9, k * 71 % 23 - 11].map(f64::from);
            (d != [0.0; 3]).then_some(d)
        })
    }

    /// What a store reads where it holds no brick, for the tests of the pyramid: the Planet's
    /// bricks, at the mip level `above` levels over the store's own. The pyramid reads the Voxels
    /// beside the bricks that it makes a coarse brick from, to tell the ones that are exposed.
    #[derive(Debug)]
    struct Ground {
        above: usize,
    }

    impl BrickSource for Ground {
        fn brick(&self, level: usize, coord: BrickCoord) -> Brick {
            planet_brick(self.above + level, coord)
        }
    }

    /// The Planet's brick at `coord` of mip `level`, which may be 0.
    fn planet_brick(level: usize, coord: BrickCoord) -> Brick {
        match level {
            0 => planet().brick(coord),
            n => planet().coarse_brick(n, coord),
        }
    }

    /// The brick at `coord` of mip `level` as the pyramid's downsampling makes it from the Planet's
    /// bricks of level 0 under it, which are 8 to the power of `level`: the real thing.
    fn downsampled(level: usize, coord: BrickCoord) -> Brick {
        let side = 1 << level;
        let mut store = BrickStore::new();
        store.set_source(Ground { above: 0 });
        for i in 0..side * side * side {
            let child = BrickCoord(std::array::from_fn(|a| {
                coord.0[a] * side + (i >> (level * a) & (side - 1))
            }));
            let brick = planet().brick(child);
            if !brick.is_air() {
                store.insert(child, brick);
            }
        }
        store.build_lod(level);
        store
            .level(level)
            .and_then(|held| held.brick(coord))
            .cloned()
            .unwrap_or(Brick::AIR)
    }

    /// Most Voxels of the coarse bricks that hold ground that may differ from the majority
    /// downsample of the Planet, as a fraction, at levels 1, 2, ...: the numbers measured (printed
    /// by the tests below) with a margin. The ground is not a boundary between layers that a
    /// Voxel can sit on either side of with the same majority, so the fraction grows with the
    /// Voxel's size, from under 1% at 20 cm to about 4% at 12.8 m.
    const TOLERANCE: [f64; 7] = [0.010, 0.015, 0.022, 0.023, 0.040, 0.050, 0.050];

    /// The fraction of the Voxels of the coarse bricks of mip `level` that hold ground that
    /// differ between [`Planet::coarse_brick`] and [`downsampled`]. Takes 8 times as long for
    /// each level.
    fn differing(level: usize) -> f64 {
        let (mut differ, mut total) = (0, 0);
        // The lines through every kind of country are enough where a brick costs this much.
        let more = if level < 2 {
            spread().collect()
        } else {
            Vec::new()
        };
        for d in lines().into_iter().chain(more) {
            let surface = planet().map().elevation_at(d);
            let step = f64::from(BRICK_SIZE << level);
            let mut h = surface - 560.0;
            while h < surface.max(0.0) + 40.0 {
                let coord = ancestor(along(d, SEA_RADIUS + h).brick(), level);
                h += step;
                let made = planet().coarse_brick(level, coord);
                let real = downsampled(level, coord);
                if made.is_uniform() && made == real {
                    continue;
                }
                differ += locals().filter(|&l| made.get(l) != real.get(l)).count();
                total += 512;
            }
        }
        let fraction = differ as f64 / total as f64;
        eprintln!("level {level}: {differ} of {total} Voxels differ, {fraction:.4}");
        assert!(
            total > 10_000,
            "level {level}: only {total} Voxels compared"
        );
        fraction
    }

    #[test]
    fn a_coarse_brick_over_the_ground_is_the_planets_downsampled_but_for_a_few_voxels() {
        for level in 1..=3 {
            assert!(differing(level) <= TOLERANCE[level - 1], "level {level}");
        }
    }

    #[test]
    #[ignore = "takes minutes: it downsamples 8^4 to 8^7 bricks for each coarse brick"]
    fn a_coarser_brick_over_the_ground_is_the_planets_downsampled_but_for_a_few_voxels() {
        for level in 4..=TOLERANCE.len() {
            assert!(differing(level) <= TOLERANCE[level - 1], "level {level}");
        }
    }

    /// Most Voxels of the coarse bricks that hold ground that may differ from the majority
    /// downsample of the eight bricks one level finer that [`Planet::coarse_brick`] makes, as a
    /// fraction, at levels 8, 9, ...: the numbers measured (printed by the test below, 2.5% at
    /// level 8, 2.7% at level 9 and 1.9% to 2.4% from level 11) with a quarter more. These are
    /// the orbit levels, where the bricks under one are too many to make from the Planet, so this
    /// is the agreement that can be checked there. It is one step, not the whole way down: each
    /// finer brick is already a sample and not the Planet's majority.
    const ORBIT_TOLERANCE: [f64; 7] = [0.032, 0.034, 0.033, 0.025, 0.026, 0.024, 0.031];

    /// The brick at `coord` of mip `level` as the pyramid's downsampling makes it from the eight
    /// bricks one level finer that [`Planet::coarse_brick`] generates, by the pyramid's own rule.
    fn downsampled_from_finer(level: usize, coord: BrickCoord) -> Brick {
        let mut store = BrickStore::new();
        store.set_source(Ground { above: level - 1 });
        for child in coord.children() {
            let brick = planet_brick(level - 1, child);
            if !brick.is_air() {
                store.insert(child, brick);
            }
        }
        store.build_lod(1);
        store.brick_at(1, coord).into_owned()
    }

    /// The bricks of mip `level` over the ground along each of `directions`: the brick over it,
    /// and the six that touch it, which hold the rest of it.
    fn bricks_over_the_ground(level: usize, directions: Vec<[f64; 3]>) -> Vec<BrickCoord> {
        let mut bricks = Vec::new();
        for d in directions {
            let surface = planet().map().elevation_at(d);
            let over = ancestor(along(d, SEA_RADIUS + surface.max(0.0)).brick(), level).0;
            for i in 0..7_i32 {
                bricks.push(BrickCoord(std::array::from_fn(|a| {
                    over[a]
                        + if i > 0 && (i - 1) / 2 == a as i32 {
                            i % 2 * 2 - 1
                        } else {
                            0
                        }
                })));
            }
        }
        bricks
    }

    #[test]
    fn an_orbit_level_brick_is_its_eight_finer_bricks_downsampled_but_for_a_few_voxels() {
        for (n, tolerance) in ORBIT_TOLERANCE.into_iter().enumerate() {
            let level = 8 + n;
            let (mut differ, mut total) = (0, 0);
            let directions = lines().into_iter().chain(spread()).collect();
            for coord in bricks_over_the_ground(level, directions) {
                let made = planet().coarse_brick(level, coord);
                let real = downsampled_from_finer(level, coord);
                if made.is_uniform() && made == real {
                    continue;
                }
                differ += locals().filter(|&l| made.get(l) != real.get(l)).count();
                total += 512;
            }
            let fraction = differ as f64 / total as f64;
            eprintln!("level {level}: {differ} of {total} Voxels differ, {fraction:.4}");
            assert!(
                total >= 4_000,
                "level {level}: only {total} Voxels compared"
            );
            assert!(fraction <= tolerance, "level {level}: {fraction}");
        }
    }

    /// Most of the exposed Voxels of the coarse bricks that hold ground whose Material may differ
    /// from the pyramid's for the eight bricks one level finer that [`Planet::coarse_brick`]
    /// makes, as a fraction, at levels 1, 2, ...: the numbers measured (printed by the test
    /// below) with a quarter more. The Voxels that are not exposed take the Material at their
    /// middles, as the pyramid's buried ones take the most common of theirs, and are not what
    /// this is about. It is one step from the finer level, as `ORBIT_TOLERANCE` is, and a Voxel
    /// that is air in the pyramid and solid in the Planet counts against it.
    ///
    /// The fraction grows with the level, from none at levels 1 and 2 to 22% at level 14, because
    /// a Voxel then is wider than the ground's relief and the strata, and the two choices differ
    /// in what they look at: the Voxel's top for one, and for the other the Voxels of the finer
    /// level that have air beside them, which are lower where the ground slopes or is under
    /// the Voxel's top (a coarse Voxel's neighbour is far off). Without the surface rule it
    /// was 9% to 48% from level 3 to level 11, and 13% at level 14, where the Voxel's middle is
    /// in the deep rock.
    const EXPOSED_TOLERANCE: [f64; 14] = [
        0.005, 0.005, 0.038, 0.066, 0.036, 0.031, 0.030, 0.059, 0.099, 0.141, 0.141, 0.155, 0.191,
        0.277,
    ];

    #[test]
    fn an_exposed_voxel_is_what_the_pyramid_makes_of_its_eight_finer_voxels_but_for_a_few() {
        for (n, tolerance) in EXPOSED_TOLERANCE.into_iter().enumerate() {
            let level = n + 1;
            let size = f64::from(1 << level);
            let (mut differ, mut solidity, mut total) = (0, 0, 0);
            let directions = lines().into_iter().chain(spread().step_by(2)).collect();
            for coord in bricks_over_the_ground(level, directions) {
                let made = planet().coarse_brick(level, coord);
                if made.is_air() {
                    continue;
                }
                let real = downsampled_from_finer(level, coord);
                for local in locals().filter(|&l| made.get(l).is_some()) {
                    let exposed = (0..6).any(|face| {
                        let step = if face % 2 == 0 { -1 } else { 1 };
                        let mut next = local.map(i32::from);
                        next[face / 2] += step;
                        if (0..8).contains(&next[face / 2]) {
                            return made.get(next.map(|c| c as u8)).is_none();
                        }
                        let mut beside = middle(level, coord, local);
                        beside[face / 2] += f64::from(step) * size;
                        planet().material_at(beside).is_none()
                    });
                    if exposed {
                        total += 1;
                        differ += usize::from(real.get(local) != made.get(local));
                        solidity += usize::from(real.get(local).is_none());
                    }
                }
            }
            let fraction = differ as f64 / total as f64;
            eprintln!(
                "level {level}: {differ} of {total} exposed Voxels differ, {fraction:.4} \
                 ({solidity} are air in the pyramid)"
            );
            assert!(total >= 500, "level {level}: only {total} Voxels compared");
            assert!(fraction <= tolerance, "level {level}: {fraction}");
        }
    }

    /// Bricks of mip `level` down lines through every kind of country, and across the interior's
    /// boundaries.
    fn coarse_bricks_to_check(level: usize) -> Vec<BrickCoord> {
        let size = f64::from(1 << level);
        let mut coords = Vec::new();
        for d in lines() {
            let surface = planet().map().elevation_at(d);
            let mut h = surface - 560.0 - 8.0 * size;
            while h < surface.max(0.0) + 40.0 + 8.0 * size {
                coords.push(ancestor(along(d, SEA_RADIUS + h).brick(), level));
                h += 8.0 * size;
            }
        }
        for r in [MOLTEN_RADIUS, INTERIOR_RADIUS, 5] {
            for d in [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-0.3, 0.2, -1.0]] {
                for k in -3..=3 {
                    let at = along(d, f64::from(r) + f64::from(k) * 8.0 * size);
                    coords.push(ancestor(at.brick(), level));
                }
            }
        }
        coords
    }

    /// The point the Voxel at `local` of the brick `coord` of mip `level` is sampled at, worked
    /// out here from the Voxel's own size and not by [`Samples`]: the middle of the `2^level`
    /// Voxels it stands for.
    fn middle(level: usize, coord: BrickCoord, local: [u8; 3]) -> [f64; 3] {
        let size = f64::from(1 << level);
        let origin = coord.0.map(|c| f64::from(c) * 8.0 * size);
        [0, 1, 2].map(|a| origin[a] + f64::from(local[a]) * size + (size - 1.0) / 2.0)
    }

    /// Whether the Planet has air at the middle of one of the Voxels next to the one sampled at
    /// `point` in mip `level`, on a face of it.
    fn air_beside(level: usize, point: [f64; 3]) -> bool {
        let size = f64::from(1 << level);
        (0..6).any(|face| {
            let mut next = point;
            next[face / 2] += if face % 2 == 0 { -size } else { size };
            planet().material_at(next).is_none()
        })
    }

    #[test]
    fn a_coarse_brick_is_what_the_planet_is_at_the_middle_of_each_of_its_voxels_but_for_the_surface()
     {
        // The Voxels of a brick of mip `level` are 2^level wide. A Voxel that is solid with air
        // beside it can take another Material than the one at its middle; any other is exact,
        // and the exposed ones are solid at the middle all the same.
        let (mut exact, mut exposed) = (0, 0);
        for level in [1, 3, 7, 12] {
            for coord in coarse_bricks_to_check(level) {
                let brick = planet().coarse_brick(level, coord);
                for local in locals() {
                    let middle = middle(level, coord, local);
                    let at_middle = planet().material_at(middle);
                    if brick.get(local) == at_middle {
                        exact += 1;
                        continue;
                    }
                    exposed += 1;
                    assert!(at_middle.is_some(), "level {level} {coord:?} {local:?}");
                    assert!(
                        brick.get(local).is_some() && air_beside(level, middle),
                        "level {level} {coord:?} {local:?}"
                    );
                }
            }
        }
        // And there are some of both: the surface changes some Voxels, and not most.
        assert!(exposed > 100 && exact > 100 * exposed, "{exact} {exposed}");
    }

    /// The Material of the topmost solid Voxel, going down a radial line from above the ground,
    /// in the bricks of mip `level`: the first Voxel that a ray down it enters. The steps are
    /// short enough not to miss a corner of a Voxel that the line only clips.
    fn top_of_the_ground(direction: [f64; 3], level: usize) -> Option<Material> {
        let size = 1 << level;
        let surface = planet().map().elevation_at(direction);
        let mut h = surface + 3.0 * f64::from(size);
        while h > surface - 3.0 * f64::from(size) {
            let at = along(direction, SEA_RADIUS + h);
            let voxel = at.0.map(|c| c.div_euclid(size));
            let brick = BrickCoord(voxel.map(|v| v.div_euclid(8)));
            let local = voxel.map(|v| v.rem_euclid(8) as u8);
            if let Some(material) = planet().coarse_brick(level, brick).get(local) {
                return Some(material);
            }
            h -= f64::from(size) / 16.0;
        }
        None
    }

    #[test]
    fn the_top_layer_of_the_ground_is_what_a_coarse_voxel_over_it_is_at_every_level() {
        // Temperate land has four Voxels of grassy soil over the soil, clay and rock under them,
        // and snow, sand and basalt are 6 to 10 thick. A coarse Voxel is wider than all of it
        // from level 3 or 4, and the middle of one with ground in it is deep in the strata, but
        // the top of the ground is what is seen of it. This holds up to the level at which the
        // Voxels are wider than the country that the line goes down: beyond it, the Voxel at
        // the top is another biome's.
        let temperate = find(|t| t.biome == Biome::Temperate && (300..600).contains(&t.elevation));
        let snow = find(|t| t.biome == Biome::Snow && t.elevation >= 100);
        let desert = find(|t| t.biome == Biome::Desert && t.elevation >= 100);
        let volcanic = find(|t| t.biome == Biome::Volcanic && t.elevation >= 100);
        for (direction, top, levels) in [
            (temperate, GRASSY_SOIL, 10),
            (snow, SNOW, 10),
            (desert, SAND, 6),
            (volcanic, BASALT, 8),
        ] {
            for level in 1..=levels {
                assert_eq!(
                    top_of_the_ground(direction, level),
                    Some(top),
                    "{top:?} at level {level}"
                );
            }
        }
    }

    #[test]
    fn the_shortcuts_make_the_brick_that_working_out_every_voxel_does() {
        // `coarse_brick` decides a brick without its Voxels when it can. That must be the
        // brick its Voxels make, and it is not just by the Material at their middles: the
        // Voxels beside the air are the surface. Over the ground, the sea and the sky, and the
        // interior, at every level there is.
        let mut compared = 0;
        for level in 1..=14 {
            // The bricks around the ground are the ones a shortcut can get wrong: a brick that is
            // all one layer, with the air one Voxel beyond its edge.
            let around =
                bricks_over_the_ground(level, lines().into_iter().chain(spread()).collect());
            for coord in coarse_bricks_to_check(level).into_iter().chain(around) {
                let brick = planet().coarse_brick(level, coord);
                let all = planet().evaluate(&Samples::new(level, coord));
                assert_eq!(brick, all, "level {level} {coord:?}");
                compared += 1;
            }
        }
        assert!(compared > 3_000, "{compared}");
    }

    #[test]
    fn a_coarse_brick_far_from_the_ground_costs_no_voxels() {
        // The rings read every coarse brick of the regions that they move over, most of them sky
        // or deep rock, so those are decided by their distance from the centre alone.
        let spawn = planet().spawn(SPAWN_REACH);
        for (radius, level) in [
            (100_000.0, 1),
            (100_000.0, 5),
            (260_000.0, 3),
            (260_000.0, 9),
        ] {
            let at = ancestor(along(spawn, radius).brick(), level).0;
            let before = EVALUATED.with(|made| made.get());
            for x in -8..=8 {
                for y in -8..=8 {
                    for z in -8..=8 {
                        let coord = BrickCoord([at[0] + x, at[1] + y, at[2] + z]);
                        std::hint::black_box(planet().coarse_brick(level, coord));
                    }
                }
            }
            assert_eq!(
                EVALUATED.with(|made| made.get()),
                before,
                "{radius} {level}"
            );
        }
    }

    /// The Voxels of the brick `coord` of mip `level` that the brick `inner` of the finer mip
    /// `inner_level` covers.
    fn covered(
        level: usize,
        coord: BrickCoord,
        inner_level: usize,
        inner: BrickCoord,
    ) -> impl Iterator<Item = [u8; 3]> {
        let across = 1 << (level - inner_level);
        let width = BRICK_SIZE / across;
        locals().filter(move |local| {
            (0..3).all(|a| {
                let first = (inner.0[a] - coord.0[a] * across) * width;
                (first..first + width).contains(&i32::from(local[a]))
            })
        })
    }

    #[test]
    fn digging_the_region_shows_in_the_coarse_bricks_over_it_and_not_the_ground_generated() {
        let spawn = planet().spawn(SPAWN_REACH);
        let ground = along(spawn, distance(spawn)).brick();
        let mut store = region().clone();
        store.build_lod(3);
        let dug = ground.parent();
        assert!(!store.brick_at(1, dug).is_air());
        assert!(!planet().coarse_brick(1, dug).is_air());
        // The ground of the level 1 brick, dug out whole: it reads as air, and so do the Voxels
        // of the levels above it that it makes up, though the Planet has ground there.
        for child in dug.children() {
            store.insert(child, Brick::AIR);
        }
        store.refresh_lod(dug.children());
        assert!(store.brick_at(1, dug).is_air());
        for level in [2, 3] {
            let above = ancestor(dug, level - 1);
            let made = planet().coarse_brick(level, above);
            let read = store.brick_at(level, above);
            let mut ground_there = 0;
            for local in covered(level, above, 1, dug) {
                assert_eq!(read.get(local), None, "level {level} {local:?}");
                ground_there += usize::from(made.get(local).is_some());
            }
            assert!(ground_there > 0, "level {level}");
        }
    }

    #[test]
    fn a_polar_sea_is_ice_on_water_on_the_sea_floor() {
        let d = find(|t| t.biome == Biome::Snow && t.elevation < -300);
        let surface = planet().map().elevation_at(d);
        let runs = column(d, surface - 3.0, 30.0);
        let found: Vec<_> = runs.iter().map(|r| r.1).collect();
        assert_eq!(found, [Some(SAND), Some(WATER), Some(ICE), None]);
        // The ice is a thin sheet at sea level: the sea's top 60 cm.
        assert!((runs[2].0 + ICE_THICKNESS).abs() <= 2.0, "{runs:?}");
        assert!(runs[3].0.abs() <= 2.0, "{runs:?}");
        // Whole bricks agree, including the ones that are all ice and the ones that are water
        // below it.
        for dr in (-40..=10).step_by(2) {
            let coord = along(d, SEA_RADIUS + f64::from(dr)).brick();
            let brick = planet().brick(coord);
            for (local, voxel) in voxels_of(coord) {
                assert_eq!(brick.get(local), planet().voxel(voxel), "{coord:?}");
            }
        }
        // A sea that is not polar has no ice.
        let warm = find(|t| t.biome == Biome::Temperate && t.elevation < -300);
        let found: Vec<_> = column(warm, -30.0, 30.0).iter().map(|r| r.1).collect();
        assert_eq!(found, [Some(WATER), None]);
    }

    #[test]
    fn tangents_are_unit_vectors_along_the_ground() {
        for up in [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.6, 0.0, 0.8],
            [-0.48, 0.6, -0.64],
        ] {
            let (ahead, aside) = tangents(up);
            let dot = |a: [f64; 3], b: [f64; 3]| (0..3).map(|k| a[k] * b[k]).sum::<f64>();
            assert!((dot(ahead, ahead) - 1.0).abs() < 1e-12);
            assert!((dot(aside, aside) - 1.0).abs() < 1e-12);
            assert!(dot(ahead, up).abs() < 1e-12 && dot(aside, up).abs() < 1e-12);
            assert!(dot(ahead, aside).abs() < 1e-12);
        }
    }

    #[test]
    fn the_planet_source_gives_the_planets_own_bricks_at_every_level() {
        let source = PlanetSource::new(planet().clone());
        // On the ground, in the sky and deep inside, at level 0 and at coarse levels.
        for radius in [
            f64::from(WORLD_RADIUS) + 30.0,
            f64::from(WORLD_RADIUS) + 400_000.0,
            1000.0,
        ] {
            let coord = along([0.3, 0.8, 0.5], radius).brick();
            assert_eq!(source.brick(0, coord), planet().brick(coord));
            for level in 1..5 {
                let coarse = BrickCoord(coord.0.map(|c| c >> level));
                assert_eq!(
                    source.brick(level, coarse),
                    planet().coarse_brick(level, coarse)
                );
            }
        }
    }
}
