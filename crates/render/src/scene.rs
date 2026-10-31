//! The Planet's bricks packed for the GPU: a shallow sparse tree over bricks.
//!
//! Three levels, each found from the one above:
//!
//! 1. **Regions** are 8×8×8 bricks (64³ Voxels). They are found by hashing the region's integer
//!    coordinate into `table`, an open-addressed hash table, so only regions that hold a brick
//!    cost anything and the Planet's extent never matters.
//! 2. **Bricks** (8³ Voxels). A region is 17 words in `regions`: a 512-bit mask of which of its
//!    bricks exist (bit `x | y << 3 | z << 6`), then the offset of its first entry in
//!    `brick_refs`. A brick's entry is at that offset plus the number of lower mask bits set.
//!    A region that is one Material throughout has `SOLID_FLAG | Material` there instead, with
//!    no entries and an all-ones mask: the Planet's core is such regions.
//! 3. **Voxels.** An entry in `brick_refs` is either `SOLID_FLAG | Material` for a brick that
//!    is one Material throughout, which needs no more storage, or the offset in `bricks` of a
//!    mixed brick:
//!
//!    | words      | content                                                              |
//!    |------------|----------------------------------------------------------------------|
//!    | 0..16      | 512-bit occupancy mask, Voxel `x \| y << 3 \| z << 6` is bit `i % 32` of word `i / 32` |
//!    | 16         | `bits` (bits per palette index) in the low 8 bits, palette length above |
//!    | 17..       | palette: two 16-bit Materials per word, low half first               |
//!    | after that | one `bits`-wide palette index per Voxel, packed from the lowest bit  |
//!
//! `table` holds 4 words per slot: the region's coordinate (3 × i32 as u32), then its index in
//! `regions` plus one, or 0 for an empty slot.
//!
//! Voxel and region coordinates are absolute `i32` Planet-frame integers on both sides, so
//! nothing in the structure loses precision far from the origin. The box of a level bounds its
//! regions, so a ray from far away can jump straight to them.
//!
//! **Level of detail.** A scene is one or more *levels*, and level `n` is mip `n` of the brick
//! store (ADR 0002): bricks, and so Voxels, `2^n` times the size of level 0's. Every level has
//! its own part of `table`, and its regions are in the one `regions`; the bricks of all levels
//! share `brick_refs` and `bricks`, so that the GPU needs no more buffers for more levels. A
//! region of level `n` is 8×8×8 bricks of that level, and so `64 << n` Voxels across. The
//! coordinates in `table`, `regions` and in the walk are those of the level's own Voxels, bricks
//! and regions; only the boxes ([`Level`]) are in level 0 Voxels, so that all levels share one
//! frame.
//!
//! [`Scene::new`] makes the one level of an ordinary scene, which holds all of its bricks.
//! [`Scene::rings`] makes a level for each mip, each listing the bricks that the LOD bands (see
//! [`crate::lod`]) select for it; they are described at that function.

use std::collections::{BTreeMap, BTreeSet};

use brick_store::{Brick, BrickCoord, BrickStore, BrickView, Material};

mod rings;
pub mod select;

pub use rings::{RingPatch, RingsFull, Room};

/// Bricks along each edge of a region.
pub const REGION_BRICKS: i32 = 8;
/// Voxels along each edge of a region.
pub const REGION_VOXELS: i32 = 64;
/// Words per region record: 16 mask words and the offset of its first brick entry.
pub const REGION_WORDS: usize = 17;
/// Where the offset of the first brick entry is in a region record, after the 16 words of the mask
/// (so also how many words the mask has).
pub const REGION_FIRST_REF: usize = REGION_WORDS - 1;
/// Set in a brick entry for a brick that is one Material throughout.
pub const SOLID_FLAG: u32 = 0x8000_0000;
/// Most levels a scene can have. Level `n` has regions `64 << n` Voxels across, which must stay
/// within `i32` Voxel coordinates: that is about 22 levels, so 20 leaves room.
/// `MAX_LEVELS` in `trace.wgsl` is the same number.
pub const MAX_LEVELS: usize = 20;

/// A ball of one Material at the Planet's centre, standing in for the bricks inside it so that
/// they need not be stored. A brick the store holds wins over the ball.
///
/// A brick is in the ball when every Voxel of it is within `radius` (see
/// [`BrickCoord::within_ball`]).
#[derive(Clone, Copy, Debug)]
pub struct Core {
    pub radius: i32,
    pub material: Material,
}

/// The box of regions that a scene holds of one mip level, whose index in [`Scene::levels`] is
/// the level. The box is in level 0 Voxels, so that every level is in one frame; a region of
/// level `n` is `REGION_VOXELS << n` of them across.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Level {
    /// Level 0 Voxel coordinate of the box's minimum corner: a multiple of the level's region
    /// size.
    pub origin: [i32; 3],
    /// Extent of the box in regions of this level. The walk goes by the occupied box, which is
    /// inside it.
    pub dims: [u32; 3],
    /// The smallest box of regions of this level, from the first to the one after the last, that
    /// has every region the level lists: the walk goes no further than it, so a ray that is
    /// clear of it, or a level with nothing, costs no steps. The two are equal for a level that
    /// lists nothing.
    pub occupied_first: [i32; 3],
    pub occupied_end: [i32; 3],
    /// First slot of the level's part of [`Scene::table`], and its slot count minus one. Slots
    /// are a power of two.
    pub table_base: u32,
    pub table_mask: u32,
    /// The regions of the level that no brick of it can be in, because they are inside the ball
    /// about the camera that the level's band starts outside of: from the first up to, and not
    /// including, the end, in regions of this level. The walk skips them. None if the end is not
    /// past the first, and for a scene that is not rings.
    pub hole_first: [i32; 3],
    pub hole_end: [i32; 3],
}

impl Level {
    /// Coordinate, in regions of mip `level`, of the box's first region.
    pub fn first_region(&self, level: usize) -> [i32; 3] {
        self.origin.map(|o| o >> (6 + level))
    }
}

/// A ball that has the whole Planet in it: nothing is outside it. The walk goes no further than
/// where a ray is in the ball, which is exact for any Planet that is inside it.
///
/// The ball bounds the Voxels of level 0. A Voxel of a coarser level is drawn as a whole cube,
/// and is solid when half of its children are, so it can stick out of the ball by its diagonal;
/// [`Scene::clip_radius`] allows for that. An Edit that adds matter beyond the ball (building,
/// #60) must grow `radius` to hold it: digging only takes matter away, so it never does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bound {
    /// The ball's centre, in Voxels.
    pub centre: [i32; 3],
    /// Its radius in Voxels: at least 1, and below 2^24 so that the GPU has it exactly. The
    /// Planet's mean radius plus the highest it goes (`planet_gen::MAX_RADIUS` for the test
    /// Planet, `planet_gen::planet::MAX_RADIUS` for the full-size one), worked out from the
    /// world, and not from the box of regions that it fills.
    pub radius: i32,
}

/// Voxels added to the radius that rays are clipped to, which are more than the clip's rounding
/// can come to: a ray at the edge of the ball is a hair to either side of it in f32, never a
/// Voxel.
pub const BOUND_SLACK: f32 = 8.0;

/// Packed bricks and the index over them. Built by [`Scene::new`] or [`Scene::rings`], uploaded
/// as-is.
#[derive(Debug, Default)]
pub struct Scene {
    /// The boxes of regions held, finest level first. Empty when there are no bricks.
    pub levels: Vec<Level>,
    pub table: Vec<u32>,
    pub regions: Vec<u32>,
    pub brick_refs: Vec<u32>,
    pub bricks: Vec<u32>,
    /// Where nothing is beyond, if that is known. Rebuilding a scene keeps it.
    pub bound: Option<Bound>,
    /// What [`Scene::update`] resolves bricks the store does not hold against. Rings ask the
    /// store instead, which has its own source (see [`brick_store::BrickSource`]).
    core: Option<Core>,
    /// Words of `brick_refs` and `bricks` that updates have replaced and nothing refers to.
    garbage: usize,
    /// What a scene of rings keeps besides the buffers; set if this is one.
    rings: Option<rings::Rings>,
}

/// Room to leave in a buffer of `len` words for it to grow by: a sixteenth, and some. The GPU
/// buffers are made this much bigger than the scene, and [`Scene::update`] packs the scene again
/// once its garbage passes this much.
pub(crate) fn spare_words(len: usize) -> usize {
    len / 16 + 4096
}

/// What [`Scene::update`] changed, for copying to a GPU copy of the scene.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SceneUpdate {
    /// `brick_refs` and `bricks` only grew: everything from these offsets (in words) to the end
    /// is new. The records in `regions` listed (by index, ascending) were rewritten in place.
    /// The header (`levels`) and the table are as before.
    Patched {
        regions: Vec<usize>,
        brick_refs_from: usize,
        bricks_from: usize,
    },
    /// A scene of rings was moved or edited. The patch lists what changed, wherever in the
    /// buffers it is.
    Rings(RingPatch),
    /// The scene was built again from scratch, so all of it is new.
    Rebuilt,
}

/// Mixes a region coordinate into a table slot. The WGSL `hash_region` is the same function.
pub fn hash_region(key: [i32; 3]) -> u32 {
    let [x, y, z] = key.map(|k| k as u32);
    let mut h =
        x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ z.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^ (h >> 12)
}

/// What one region holds while the scene is being built.
#[derive(Default)]
struct Region<'a> {
    /// Set when the whole region is one Material and `bricks` is empty.
    uniform: Option<Material>,
    /// (mask bit, brick); a `None` brick is the core's.
    bricks: Vec<(usize, Option<&'a Brick>)>,
    /// Mask bits of bricks the store holds as air, which the core must not fill.
    air: Vec<usize>,
}

fn region_of(brick: BrickCoord) -> [i32; 3] {
    brick.0.map(|c| c.div_euclid(REGION_BRICKS))
}

/// A brick's bit in its region's mask.
fn bit_of(brick: BrickCoord) -> usize {
    let [x, y, z] = brick.0.map(|c| c.rem_euclid(REGION_BRICKS) as usize);
    x | y << 3 | z << 6
}

/// Appends the words of a mixed brick to `out`, laid out as in the table in the module docs.
fn encode_mixed(
    out: &mut Vec<u32>,
    occupancy: &[u64; 8],
    palette: &[Material],
    bits: u32,
    indices: &[u8],
) {
    for word in occupancy {
        out.extend([*word as u32, (*word >> 32) as u32]);
    }
    out.push(bits | (palette.len() as u32) << 8);
    for pair in palette.chunks(2) {
        let low = u32::from(pair[0].0);
        let high = pair.get(1).map_or(0, |m| u32::from(m.0));
        out.push(low | high << 16);
    }
    // The CPU packs indices into bytes from the lowest bit; little-endian words keep that order.
    out.extend(
        indices
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("indices are whole words"))),
    );
}

/// Fills in the bricks of `core` that the store does not hold.
fn add_core<'a>(by_region: &mut BTreeMap<[i32; 3], Region<'a>>, core: Core) {
    // Regions that can touch the ball: its bounding cube, in regions, rounded out.
    let reach = core.radius.div_euclid(REGION_VOXELS) + 1;
    for x in -reach..=reach {
        for y in -reach..=reach {
            for z in -reach..=reach {
                let key = [x, y, z];
                let first = BrickCoord(key.map(|k| k * REGION_BRICKS));
                // The ball is convex, so a region is inside it when its eight corner bricks are.
                let corner = |i: usize| {
                    BrickCoord(std::array::from_fn(|a| {
                        first.0[a] + (i >> a & 1) as i32 * (REGION_BRICKS - 1)
                    }))
                };
                let inside = (0..8).all(|i| corner(i).within_ball(core.radius));
                // A region the ball does not touch has all eight corners beyond it by at least a
                // region's diagonal; skip those without testing every brick.
                if !inside
                    && !(0..8).any(|i| corner(i).within_ball(core.radius + 2 * REGION_VOXELS))
                {
                    continue;
                }
                let region = by_region.entry(key).or_default();
                if inside && region.bricks.is_empty() && region.air.is_empty() {
                    region.uniform = Some(core.material);
                    continue;
                }
                let held: Vec<usize> = region
                    .bricks
                    .iter()
                    .map(|&(bit, _)| bit)
                    .chain(region.air.iter().copied())
                    .collect();
                for bit in (0..512).filter(|bit| !held.contains(bit)) {
                    let brick = BrickCoord([
                        first.0[0] + (bit & 7) as i32,
                        first.0[1] + (bit >> 3 & 7) as i32,
                        first.0[2] + (bit >> 6) as i32,
                    ]);
                    if brick.within_ball(core.radius) {
                        region.bricks.push((bit, None));
                    }
                }
            }
        }
    }
}

impl Scene {
    /// Packs every brick in `store` and, if given, the `core`. The same input always gives the
    /// same buffers.
    pub fn new(store: &BrickStore, core: Option<Core>) -> Scene {
        // Group bricks by region. BTreeMap and sorted coordinates keep the output deterministic.
        // A brick is `None` where the core stands in for it.
        let mut by_region: BTreeMap<[i32; 3], Region<'_>> = BTreeMap::new();
        for coord in store.sorted_coords() {
            let brick = store.brick(coord).expect("sorted_coords lists held bricks");
            let region = by_region.entry(region_of(coord)).or_default();
            if brick.is_air() {
                region.air.push(bit_of(coord));
            } else {
                region.bricks.push((bit_of(coord), Some(brick)));
            }
        }
        if let Some(core) = core {
            add_core(&mut by_region, core);
        }
        // Regions that ended up with nothing in them, such as one of only stored air.
        by_region.retain(|_, region| region.uniform.is_some() || !region.bricks.is_empty());
        let Some(first) = by_region.keys().next() else {
            return Scene {
                core,
                ..Scene::default()
            };
        };
        let (mut min, mut max) = (*first, *first);
        for region in by_region.keys() {
            for axis in 0..3 {
                min[axis] = min[axis].min(region[axis]);
                max[axis] = max[axis].max(region[axis]);
            }
        }

        let slots = (by_region.len() * 2).next_power_of_two().max(2);
        let mut scene = Scene {
            levels: vec![Level {
                origin: min.map(|m| m * REGION_VOXELS),
                dims: [0, 1, 2].map(|axis| (max[axis] - min[axis] + 1) as u32),
                occupied_first: min,
                occupied_end: max.map(|m| m + 1),
                table_base: 0,
                table_mask: slots as u32 - 1,
                ..Level::default()
            }],
            table: vec![0; slots * 4],
            core,
            ..Scene::default()
        };
        for (index, (key, region)) in by_region.into_iter().enumerate() {
            if let Some(material) = region.uniform {
                scene.regions.extend([u32::MAX; 16]);
                scene.regions.push(SOLID_FLAG | u32::from(material.0));
            } else {
                let mut bricks = region.bricks;
                bricks.sort_by_key(|&(bit, _)| bit);
                let mut mask = [0u32; 16];
                let first_ref = scene.brick_refs.len() as u32;
                for (bit, brick) in bricks {
                    mask[bit / 32] |= 1 << (bit % 32);
                    let view = brick.map_or_else(
                        || {
                            BrickView::Solid(
                                core.expect("only the core adds absent bricks").material,
                            )
                        },
                        Brick::view,
                    );
                    let entry = scene.pack_brick(view);
                    scene.brick_refs.push(entry);
                }
                scene.regions.extend(mask);
                scene.regions.push(first_ref);
            }
            scene.insert_region(0, key, index as u32);
        }
        scene
    }

    /// Appends `view`'s data and returns its entry for `brick_refs`.
    fn pack_brick(&mut self, view: BrickView<'_>) -> u32 {
        match view {
            BrickView::Air => unreachable!("air bricks are skipped"),
            BrickView::Solid(material) => SOLID_FLAG | u32::from(material.0),
            BrickView::Mixed {
                occupancy,
                palette,
                bits,
                indices,
            } => {
                let offset = self.bricks.len() as u32;
                encode_mixed(&mut self.bricks, occupancy, palette, bits, indices);
                offset
            }
        }
    }

    /// Lists the region at `key` of mip `level`, whose record in `regions` is `index`. Returns
    /// the slot of `table` that it took.
    fn insert_region(&mut self, level: usize, key: [i32; 3], index: u32) -> usize {
        let Level {
            table_base,
            table_mask: mask,
            ..
        } = self.levels[level];
        let mut slot = hash_region(key) & mask;
        while self.table[(table_base + slot) as usize * 4 + 3] != 0 {
            slot = (slot + 1) & mask;
        }
        let at = (table_base + slot) as usize * 4;
        self.table[at..at + 3].copy_from_slice(&key.map(|r| r as u32));
        self.table[at + 3] = index + 1;
        (table_base + slot) as usize
    }

    /// Unlists the region at `key` of mip `level` and returns its record's index in `regions`.
    /// The slots of `table` that changed are added to `touched`.
    ///
    /// A search for a region ends at an empty slot, and regions that collided with the one
    /// removed may sit after its slot, so those are moved back to close the gap.
    fn remove_region(
        &mut self,
        level: usize,
        key: [i32; 3],
        touched: &mut BTreeSet<usize>,
    ) -> Option<usize> {
        let Level {
            table_base,
            table_mask: mask,
            ..
        } = *self.levels.get(level)?;
        let at = |slot: u32| (table_base + slot) as usize * 4;
        let mut hole = hash_region(key) & mask;
        loop {
            if self.table[at(hole) + 3] == 0 {
                return None;
            }
            if self.table[at(hole)..at(hole) + 3] == key.map(|k| k as u32) {
                break;
            }
            hole = (hole + 1) & mask;
        }
        let index = self.table[at(hole) + 3] as usize - 1;
        let mut next = hole;
        loop {
            next = (next + 1) & mask;
            let entry: [u32; 4] = self.table[at(next)..at(next) + 4].try_into().unwrap();
            if entry[3] == 0 {
                break;
            }
            let home = hash_region([entry[0], entry[1], entry[2]].map(|k| k as i32)) & mask;
            // It stays if its home slot is after the hole and no later than itself, going round
            // the end of the table if need be: moving it to the hole would put it before home.
            let stays = if hole <= next {
                hole < home && home <= next
            } else {
                hole < home || home <= next
            };
            if !stays {
                self.table[at(hole)..at(hole) + 4].copy_from_slice(&entry);
                touched.insert((table_base + hole) as usize);
                hole = next;
            }
        }
        self.table[at(hole)..at(hole) + 4].fill(0);
        touched.insert((table_base + hole) as usize);
        Some(index)
    }

    /// Brings the scene up to date with `store` after the bricks `changed` (in any order, and
    /// possibly including ones that did not change) were edited, without packing all of it again.
    ///
    /// Each region holding a changed brick gets a new record in place. Its changed bricks are
    /// packed again at the end of `brick_refs` and `bricks`, and its other bricks are referred
    /// to where they already were. What was replaced stays behind as garbage, so that the
    /// buffers only ever grow, until there is more of it than [`spare_words`] of what is live.
    /// Then the scene is packed from scratch, which drops the garbage but takes about 1.6 s for
    /// the test Planet. The same happens if a changed brick is in a region that the scene lacks.
    /// Both are reported as [`SceneUpdate::Rebuilt`]. A region that Edits empty keeps a record
    /// with no bricks in it.
    ///
    /// `changed` are bricks of level 0. A scene of rings also takes in the coarser bricks over
    /// them, and can run out of room to do it even when packed again from scratch: that is
    /// [`RingsFull`], and then the scene is not to be used again. No other scene fails.
    pub fn update(
        &mut self,
        store: &BrickStore,
        changed: &[BrickCoord],
    ) -> Result<SceneUpdate, RingsFull> {
        if self.rings.is_some() {
            return self.update_rings(store, changed);
        }
        Ok(self.update_flat(store, changed))
    }

    /// [`Scene::update`] for a scene that is not rings.
    fn update_flat(&mut self, store: &BrickStore, changed: &[BrickCoord]) -> SceneUpdate {
        let mut by_region: BTreeMap<usize, Vec<BrickCoord>> = BTreeMap::new();
        for &brick in changed {
            let Some(index) = self.find_region(0, region_of(brick)) else {
                *self = Scene {
                    bound: self.bound,
                    ..Scene::new(store, self.core)
                };
                return SceneUpdate::Rebuilt;
            };
            by_region.entry(index).or_default().push(brick);
        }
        let (brick_refs_from, bricks_from) = (self.brick_refs.len(), self.bricks.len());
        for (&index, changed) in &by_region {
            let origin = region_of(changed[0]).map(|r| r * REGION_BRICKS);
            let first_ref = self.brick_refs.len() as u32;
            let mut mask = [0u32; 16];
            self.garbage += self.region_refs(index);
            for bit in 0..512 {
                let coord = BrickCoord(std::array::from_fn(|a| {
                    origin[a] + (bit >> (3 * a) & 7) as i32
                }));
                let entry = if changed.contains(&coord) {
                    let old = self.entry(index, bit);
                    self.garbage += old.map_or(0, |entry| self.mixed_words(entry));
                    self.resolve(store, coord).map(|view| self.pack_brick(view))
                } else {
                    self.entry(index, bit)
                };
                if let Some(entry) = entry {
                    mask[bit / 32] |= 1 << (bit % 32);
                    self.brick_refs.push(entry);
                }
            }
            let at = index * REGION_WORDS;
            self.regions[at..at + 16].copy_from_slice(&mask);
            self.regions[at + 16] = first_ref;
        }
        let live = self.brick_refs.len() + self.bricks.len() - self.garbage;
        if self.garbage > spare_words(live) {
            *self = Scene {
                bound: self.bound,
                ..Scene::new(store, self.core)
            };
            return SceneUpdate::Rebuilt;
        }
        SceneUpdate::Patched {
            regions: by_region.into_keys().collect(),
            brick_refs_from,
            bricks_from,
        }
    }

    /// Entries that the region record at `index` has in `brick_refs`.
    fn region_refs(&self, index: usize) -> usize {
        let record = &self.regions[index * REGION_WORDS..][..REGION_WORDS];
        if record[16] & SOLID_FLAG != 0 {
            return 0;
        }
        record[..16].iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Words that the brick with `brick_refs` entry `entry` has in `bricks`: none for a brick
    /// that is one Material throughout.
    fn mixed_words(&self, entry: u32) -> usize {
        if entry & SOLID_FLAG != 0 {
            return 0;
        }
        let header = self.bricks[entry as usize + 16];
        let (bits, palette_len) = ((header & 0xff) as usize, (header >> 8) as usize);
        // Occupancy and header, the palette, then 512 indices of `bits` bits.
        17 + palette_len.div_ceil(2) + 512 * bits / 32
    }

    /// How the scene shows the brick at `coord`: as the store has it, else as the core, else air.
    fn resolve<'a>(&self, store: &'a BrickStore, coord: BrickCoord) -> Option<BrickView<'a>> {
        match store.brick(coord) {
            Some(brick) if brick.is_air() => None,
            Some(brick) => Some(brick.view()),
            None => self
                .core
                .filter(|core| coord.within_ball(core.radius))
                .map(|core| BrickView::Solid(core.material)),
        }
    }

    /// The `brick_refs` entry for mask bit `bit` of the region record at `index`, if the region
    /// has that brick.
    fn entry(&self, index: usize, bit: usize) -> Option<u32> {
        let record = &self.regions[index * REGION_WORDS..][..REGION_WORDS];
        let first_ref = record[16];
        if first_ref & SOLID_FLAG != 0 {
            return Some(first_ref);
        }
        let mask = &record[..16];
        if mask[bit / 32] >> (bit % 32) & 1 == 0 {
            return None;
        }
        let lower: u32 = mask[..bit / 32].iter().map(|w| w.count_ones()).sum::<u32>()
            + (mask[bit / 32] & ((1u32 << (bit % 32)) - 1)).count_ones();
        Some(self.brick_refs[(first_ref + lower) as usize])
    }

    /// The radius that rays are clipped to, if the scene has a [`Bound`]: its radius, and the
    /// diagonal of a Voxel of the coarsest level the scene draws (which is as far as such a Voxel
    /// can stick out), and [`BOUND_SLACK`].
    pub fn clip_radius(&self) -> Option<f32> {
        let bound = self.bound?;
        let coarsest_level = self
            .coarsest_drawn()
            .unwrap_or(self.levels.len().saturating_sub(1));
        let coarsest = (1u32 << coarsest_level) as f32;
        Some(bound.radius as f32 + coarsest * 3f32.sqrt() + BOUND_SLACK)
    }

    /// How many levels the walk goes through: those up to the coarsest that lists a region. The
    /// levels past it have nothing to find, so rays do not look at them.
    pub fn walked_levels(&self) -> usize {
        self.levels
            .iter()
            .rposition(|l| (0..3).all(|a| l.occupied_end[a] > l.occupied_first[a]))
            .map_or(0, |last| last + 1)
    }

    /// Index in `regions` of the region at absolute region coordinate `key` of mip `level`.
    pub fn find_region(&self, level: usize, key: [i32; 3]) -> Option<usize> {
        self.find_region_probes(level, key).0
    }

    /// [`Scene::find_region`], and how many slots of `table` it looked at.
    pub fn find_region_probes(&self, level: usize, key: [i32; 3]) -> (Option<usize>, u32) {
        let Some(&Level {
            table_base,
            table_mask: mask,
            ..
        }) = self.levels.get(level)
        else {
            return (None, 0);
        };
        let mut slot = hash_region(key) & mask;
        let mut probes = 0;
        for _ in 0..=mask {
            probes += 1;
            let at = (table_base + slot) as usize * 4;
            let Some(&index) = self.table.get(at + 3) else {
                return (None, probes);
            };
            if index == 0 {
                return (None, probes);
            }
            if self.table[at..at + 3] == key.map(|k| k as u32) {
                return (Some(index as usize - 1), probes);
            }
            slot = (slot + 1) & mask;
        }
        (None, probes)
    }
}

#[cfg(test)]
mod tests {
    use brick_store::{Brick, BrickCoord, Material, VoxelCoord};

    use super::*;

    const ROCK: Material = Material(1);
    const SOIL: Material = Material(2);

    #[test]
    fn no_bricks_make_an_empty_scene() {
        let scene = Scene::new(&BrickStore::new(), None);
        assert!(scene.levels.is_empty());
        assert_eq!(scene.find_region(0, [0, 0, 0]), None);
    }

    #[test]
    fn origin_and_extent_cover_every_region_including_negative_ones() {
        let mut store = BrickStore::new();
        store.insert(BrickCoord([-1, 0, 0]), Brick::solid(ROCK));
        store.insert(BrickCoord([17, 9, -30]), Brick::solid(ROCK));
        let scene = Scene::new(&store, None);
        // Bricks -1 and 17 are in regions -1 and 2; 0 and 9 in 0 and 1; -30 in -4 (floor).
        assert_eq!(scene.levels[0].origin, [-64, 0, -256]);
        assert_eq!(scene.levels[0].dims, [4, 2, 5]);
    }

    #[test]
    fn every_region_is_found_and_empty_places_are_not() {
        let mut store = BrickStore::new();
        let mut regions = Vec::new();
        for i in 0..200 {
            let coord = BrickCoord([i * 8, (i * 5) % 40 * 8, -i * 8 - 1]);
            store.insert(coord, Brick::solid(ROCK));
            regions.push(coord.0.map(|c| c.div_euclid(8)));
        }
        let scene = Scene::new(&store, None);
        let mut found: Vec<_> = regions
            .iter()
            .map(|&r| scene.find_region(0, r).unwrap())
            .collect();
        found.sort_unstable();
        found.dedup();
        assert_eq!(found.len(), regions.len(), "each region has its own record");
        assert_eq!(scene.find_region(0, [1000, 0, 0]), None);
        assert_eq!(scene.find_region(0, [0, 0, 0]), None);
    }

    #[test]
    fn a_core_is_uniform_regions_inside_and_solid_bricks_where_it_crosses_a_region() {
        let core = Core {
            radius: 120,
            material: ROCK,
        };
        let scene = Scene::new(&BrickStore::new(), Some(core));
        // Region (0, 0, 0) spans 0..64 on each axis: its far corner is 109 from the centre.
        let inner = scene.find_region(0, [0, 0, 0]).unwrap() * REGION_WORDS;
        assert_eq!(scene.regions[inner + 16], SOLID_FLAG | 1);
        assert_eq!(scene.regions[inner], u32::MAX);
        // Region (1, 0, 0) is cut by the ball: some of its bricks are the core's, others absent.
        let edge = scene.find_region(0, [1, 0, 0]).unwrap() * REGION_WORDS;
        assert_ne!(scene.regions[edge + 16] & SOLID_FLAG, SOLID_FLAG);
        let present: u32 = scene.regions[edge..edge + 16]
            .iter()
            .map(|w| w.count_ones())
            .sum();
        assert!(present > 0 && present < 512, "{present}");
        // A region the ball cannot reach is not there at all.
        assert_eq!(scene.find_region(0, [2, 2, 2]), None);
        assert_eq!(scene.find_region(0, [5, 0, 0]), None);
    }

    #[test]
    fn stored_bricks_win_over_the_core() {
        let core = Core {
            radius: 120,
            material: ROCK,
        };
        let mut store = BrickStore::new();
        store.insert(BrickCoord([1, 1, 1]), Brick::solid(SOIL));
        let scene = Scene::new(&store, Some(core));
        // The stored brick sits in a region that would otherwise be uniform: now it is a
        // normal region with the SOIL brick among core bricks.
        let region = scene.find_region(0, [0, 0, 0]).unwrap() * REGION_WORDS;
        assert_eq!(scene.regions[region + 16] & SOLID_FLAG, 0);
        let bit = 1 | 1 << 3 | 1 << 6;
        let (word, shift) = (bit / 32, bit % 32);
        let before: u32 = scene.regions[region..region + word]
            .iter()
            .map(|w| w.count_ones())
            .sum::<u32>()
            + (scene.regions[region + word] & ((1 << shift) - 1)).count_ones();
        let entry = scene.brick_refs[(scene.regions[region + 16] + before) as usize];
        assert_eq!(entry, SOLID_FLAG | 2);
    }

    #[test]
    fn a_stored_air_brick_inside_the_core_stays_air() {
        let core = Core {
            radius: 120,
            material: ROCK,
        };
        let mut store = BrickStore::new();
        store.insert(BrickCoord([1, 1, 1]), Brick::AIR);
        let scene = Scene::new(&store, Some(core));
        let region = scene.find_region(0, [0, 0, 0]).unwrap() * REGION_WORDS;
        assert_eq!(scene.regions[region + 16] & SOLID_FLAG, 0);
        let bit = 1 | 1 << 3 | 1 << 6;
        let mask = &scene.regions[region..region + 16];
        assert_eq!(
            mask[bit / 32] >> (bit % 32) & 1,
            0,
            "the air brick is not listed"
        );
        let present: u32 = mask.iter().map(|w| w.count_ones()).sum();
        assert_eq!(
            present, 511,
            "every other brick of the region is the core's"
        );
        // Stored air outside the core makes no region of its own.
        let mut store = BrickStore::new();
        store.insert(BrickCoord([500, 0, 0]), Brick::AIR);
        assert!(Scene::new(&store, None).levels.is_empty());
    }

    #[test]
    fn a_solid_brick_is_a_flagged_entry_with_no_voxel_data() {
        let mut store = BrickStore::new();
        store.insert(BrickCoord([9, 0, 0]), Brick::solid(SOIL));
        let scene = Scene::new(&store, None);
        assert_eq!(scene.brick_refs, [SOLID_FLAG | 2]);
        assert!(scene.bricks.is_empty());
        // Region (1, 0, 0), brick (1, 0, 0) inside it is mask bit 1.
        let region = scene.find_region(0, [1, 0, 0]).unwrap() * REGION_WORDS;
        assert_eq!(scene.regions[region], 0b10);
        assert_eq!(scene.regions[region + 16], 0);
    }

    #[test]
    fn a_mixed_brick_packs_occupancy_palette_and_indices() {
        let mut brick = Brick::AIR;
        brick.set([1, 0, 0], Some(ROCK));
        brick.set([2, 0, 0], Some(SOIL));
        brick.set([0, 0, 4], Some(SOIL)); // Voxel 256: word 8, bit 0.
        let mut store = BrickStore::new();
        store.insert(BrickCoord([0, 0, 0]), brick);
        let scene = Scene::new(&store, None);
        assert_eq!(scene.brick_refs, [0]);
        let b = &scene.bricks;
        assert_eq!(b[0], 0b110);
        assert_eq!(b[8], 1);
        assert_eq!(b[16], 1 | 2 << 8, "1 bit per index, two palette entries");
        assert_eq!(b[17], 1 | 2 << 16, "ROCK then SOIL in one word");
        // 512 one-bit indices are 16 words; voxel 2 is index 1, voxel 256 is index 1.
        assert_eq!(b.len(), 17 + 1 + 16);
        assert_eq!(b[18], 0b100);
        assert_eq!(b[18 + 8], 1);
    }

    #[test]
    fn a_region_lists_its_bricks_in_mask_bit_order() {
        let mut store = BrickStore::new();
        // Inserted out of order: bits 3, 8 (y = 1) and 64 (z = 1) of region (0, 0, 0).
        for (c, m) in [([0, 0, 1], 30), ([3, 0, 0], 10), ([0, 1, 0], 20)] {
            store.insert(BrickCoord(c), Brick::solid(Material(m)));
        }
        let scene = Scene::new(&store, None);
        assert_eq!(
            scene.brick_refs,
            [SOLID_FLAG | 10, SOLID_FLAG | 20, SOLID_FLAG | 30]
        );
        let mask = &scene.regions[..16];
        assert_eq!((mask[0], mask[2]), (1 << 3 | 1 << 8, 1));
    }

    /// The Material at `voxel` as the reference tracer sees it: a ray that starts inside a
    /// solid Voxel hits that Voxel at once.
    fn material_at(scene: &Scene, voxel: [i32; 3]) -> Option<u16> {
        let ray = crate::reference::Ray {
            cell: voxel,
            origin: [0.5; 3],
            dir: [1.0, 0.0, 0.0],
        };
        crate::reference::trace(scene, &ray)
            .filter(|hit| hit.voxel == voxel)
            .map(|hit| hit.material)
    }

    /// Digs a ball like an Edit does: a core brick is stored as the rock it is before it is dug.
    fn dig(store: &mut BrickStore, core: Core, centre: [i32; 3], radius: i32) -> Vec<BrickCoord> {
        let mut changed = std::collections::BTreeSet::new();
        for x in -radius..=radius {
            for y in -radius..=radius {
                for z in -radius..=radius {
                    if x * x + y * y + z * z > radius * radius {
                        continue;
                    }
                    let voxel = VoxelCoord([centre[0] + x, centre[1] + y, centre[2] + z]);
                    let brick = voxel.brick();
                    if store.brick(brick).is_none() && brick.within_ball(core.radius) {
                        store.insert(brick, Brick::solid(core.material));
                    }
                    if store.get(voxel).is_some() {
                        changed.insert(brick);
                    }
                    store.set(voxel, None);
                }
            }
        }
        changed.into_iter().collect()
    }

    fn assert_same_voxels(a: &Scene, b: &Scene, lo: [i32; 3], hi: [i32; 3]) {
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    let at = [x, y, z];
                    assert_eq!(material_at(a, at), material_at(b, at), "{at:?}");
                }
            }
        }
    }

    #[test]
    fn an_updated_scene_shows_the_same_voxels_as_one_built_from_scratch() {
        let core = Core {
            radius: 120,
            material: ROCK,
        };
        let mut store = BrickStore::new();
        // A mixed brick and a solid one outside the core, in a region of their own.
        store.set(VoxelCoord([200, 3, 3]), Some(SOIL));
        store.insert(BrickCoord([26, 0, 0]), Brick::solid(SOIL));
        let mut scene = Scene::new(&store, Some(core));

        // Digs inside the core's uniform region, across a region boundary into negative
        // coordinates, through the mixed brick, and out of a whole solid brick.
        let mut changed = dig(&mut store, core, [8, 8, 8], 6);
        changed.extend(dig(&mut store, core, [-2, 3, -60], 9));
        changed.extend(dig(&mut store, core, [200, 3, 3], 1));
        changed.extend(dig(&mut store, core, [212, 3, 3], 8));
        let patch = scene.update(&store, &changed).unwrap();
        let SceneUpdate::Patched { regions, .. } = patch else {
            panic!("a dig needs no new region: {patch:?}");
        };
        assert!(regions.len() >= 3 && regions.is_sorted());

        let fresh = Scene::new(&store, Some(core));
        assert_same_voxels(&scene, &fresh, [-14, -4, -72], [22, 20, 20]);
        assert_same_voxels(&scene, &fresh, [190, 0, 0], [225, 12, 12]);
        // Beyond what was edited nothing differs either.
        assert_same_voxels(&scene, &fresh, [60, 60, 60], [70, 70, 70]);
    }

    #[test]
    fn an_update_only_appends_to_the_buffers() {
        let core = Core {
            radius: 120,
            material: ROCK,
        };
        let mut store = BrickStore::new();
        let mut scene = Scene::new(&store, Some(core));
        let before = (
            scene.regions.clone(),
            scene.brick_refs.clone(),
            scene.bricks.clone(),
        );
        let changed = dig(&mut store, core, [8, 8, 8], 3);
        let SceneUpdate::Patched {
            brick_refs_from,
            bricks_from,
            ..
        } = scene.update(&store, &changed).unwrap()
        else {
            panic!("a patch");
        };
        assert_eq!(
            (brick_refs_from, bricks_from),
            (before.1.len(), before.2.len())
        );
        assert_eq!(scene.regions.len(), before.0.len());
        assert_eq!(scene.brick_refs[..brick_refs_from], before.1[..]);
        assert_eq!(scene.bricks[..bricks_from], before.2[..]);
        assert!(scene.bricks.len() > bricks_from, "the dug bricks are mixed");
    }

    #[test]
    fn replaced_records_are_counted_and_too_many_pack_the_scene_again() {
        let core = Core {
            radius: 120,
            material: ROCK,
        };
        let mut store = BrickStore::new();
        let mut scene = Scene::new(&store, Some(core));
        let mut rebuilt = 0;
        for i in 0..40 {
            // Digs that overlap, so that bricks are replaced again and again.
            let changed = dig(&mut store, core, [i % 5 * 3, i % 3 * 4, 10], 9);
            let update = scene.update(&store, &changed).unwrap();
            let fresh = Scene::new(&store, Some(core));
            if update == SceneUpdate::Rebuilt {
                rebuilt += 1;
                assert_eq!(scene.garbage, 0);
                assert_eq!(
                    (&scene.brick_refs, &scene.bricks),
                    (&fresh.brick_refs, &fresh.bricks)
                );
            } else {
                let live = fresh.brick_refs.len() + fresh.bricks.len();
                let all = scene.brick_refs.len() + scene.bricks.len();
                assert_eq!(all - scene.garbage, live, "update {i}");
                assert!(scene.garbage <= spare_words(live));
            }
        }
        assert!(rebuilt > 0, "the garbage was cleared at some point");
        assert_same_voxels(
            &scene,
            &Scene::new(&store, Some(core)),
            [-2, -2, 0],
            [20, 20, 24],
        );
    }

    #[test]
    fn a_region_dug_out_completely_stays_in_the_table_and_reads_as_air() {
        let mut store = BrickStore::new();
        store.insert(BrickCoord([0, 0, 0]), Brick::solid(SOIL));
        let mut scene = Scene::new(&store, None);
        assert_eq!(material_at(&scene, [3, 3, 3]), Some(2));
        store.insert(BrickCoord([0, 0, 0]), Brick::AIR);
        scene.update(&store, &[BrickCoord([0, 0, 0])]).unwrap();
        assert_eq!(material_at(&scene, [3, 3, 3]), None);
        assert!(scene.find_region(0, [0, 0, 0]).is_some());
    }

    #[test]
    fn a_change_outside_every_region_rebuilds_the_scene() {
        let mut store = BrickStore::new();
        store.insert(BrickCoord([0, 0, 0]), Brick::solid(SOIL));
        let mut scene = Scene::new(&store, None);
        store.insert(BrickCoord([100, 0, 0]), Brick::solid(SOIL));
        let patch = scene.update(&store, &[BrickCoord([100, 0, 0])]).unwrap();
        assert_eq!(patch, SceneUpdate::Rebuilt);
        assert_eq!(material_at(&scene, [800, 3, 3]), Some(2));
    }

    #[test]
    fn building_twice_gives_identical_buffers() {
        let mut store = BrickStore::new();
        for i in 0..50 {
            store.set(VoxelCoord([i * 13, -i * 7, i * 3]), Some(ROCK));
        }
        let (a, b) = (Scene::new(&store, None), Scene::new(&store, None));
        assert_eq!(
            (a.table, a.regions, a.brick_refs, a.bricks),
            (b.table, b.regions, b.brick_refs, b.bricks)
        );
    }
}
