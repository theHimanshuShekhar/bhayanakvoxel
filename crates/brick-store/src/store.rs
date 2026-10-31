//! Sparse hash index of bricks keyed by integer brick coordinate.

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use crate::brick::{Brick, Material};
use crate::coord::{BrickCoord, VoxelCoord};
use crate::lod::{BrickSource, downsample};

/// The Planet's bricks. A brick that is absent reads as air.
///
/// The index is a `HashMap`, whose iteration order is random per process, so it is never
/// exposed: use [`Self::sorted_coords`] wherever order matters.
///
/// A brick that is not held can read as something other than air: see [`BrickSource`].
/// [`Self::brick`], [`Self::get`] and [`Self::content_hash`] only see the bricks held;
/// [`Self::brick_at`] and [`Self::voxel_at`] read what is there, at any level.
///
/// A store can also carry a pyramid of coarser mip levels (ADR 0002), built by
/// [`Self::build_lod`]. Level `n` is a store of its own, in bricks of Voxels `2^n` times the
/// size of level 0's. A coarse brick is held exactly when it covers a brick held at level 0,
/// and is then the downsampling of the eight bricks below it, each read as [`Self::brick_at`]
/// does. The pyramid is not updated by [`Self::insert`], [`Self::remove`] or [`Self::set`]:
/// after changing level 0 bricks, tell it which with [`Self::refresh_lod`]. `sim::apply` does.
#[derive(Clone, Debug, Default)]
pub struct BrickStore {
    bricks: HashMap<BrickCoord, Brick>,
    /// In the stores of the coarse levels: the bricks held there that have only solid Voxels of
    /// level 0 under them, exactly (see [`BrickStore::solid_below`]). Empty at level 0.
    solid_below: HashSet<BrickCoord>,
    /// Mip levels 1 and up, finest first. Empty when there is no pyramid; the stores in it have
    /// no pyramid or source of their own.
    coarse: Vec<BrickStore>,
    source: Option<Arc<dyn BrickSource>>,
}

/// The six bricks that share a face with `coord`.
fn touching(coord: BrickCoord) -> impl Iterator<Item = BrickCoord> {
    (0..6).map(move |face| {
        let mut next = coord.0;
        next[face / 2] = if face % 2 == 0 {
            next[face / 2].saturating_sub(1)
        } else {
            next[face / 2].saturating_add(1)
        };
        BrickCoord(next)
    })
}

/// What [`BrickStore::refresh_lod`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LodRefresh {
    /// How many coarse bricks it remade: the ancestors of the bricks it was told had changed,
    /// and those of the bricks touching them, since a coarse brick reads the Voxels beside the
    /// bricks it is made from. It stops at the first level at which no brick reads differently or
    /// changes whether it is held.
    pub remade: usize,
    /// The coarse bricks that read differently afterwards, as `(level, coordinate)`, finest
    /// level first and ascending within a level. These are the ones to send on to wherever
    /// the coarse levels are drawn from.
    pub changed: Vec<(usize, BrickCoord)>,
}

impl BrickStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of bricks held.
    pub fn len(&self) -> usize {
        self.bricks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bricks.is_empty()
    }

    pub fn brick(&self, coord: BrickCoord) -> Option<&Brick> {
        self.bricks.get(&coord)
    }

    /// Stores `brick` at `coord`, replacing any brick there.
    pub fn insert(&mut self, coord: BrickCoord, brick: Brick) {
        self.bricks.insert(coord, brick);
    }

    /// Removes the brick at `coord`, which then reads as air.
    pub fn remove(&mut self, coord: BrickCoord) -> Option<Brick> {
        self.bricks.remove(&coord)
    }

    /// FNV-1a hash of the Planet's content: the coordinate and [`Brick::content_hash`] of every
    /// brick held, in ascending coordinate order. Bricks held as air count: inside the Planet's
    /// core a brick that is not held reads as rock, so one dug out to air differs from one never
    /// touched, and two stores that differ only in that must not hash alike.
    pub fn content_hash(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for coord in self.sorted_coords() {
            let brick = &self.bricks[&coord];
            let bytes = coord
                .0
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .chain(brick.content_hash().to_le_bytes());
            for byte in bytes {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    /// Sets what a brick reads as where none is held, replacing any source, and remakes the
    /// pyramid to match. Set it before holding bricks that depend on it.
    pub fn set_source(&mut self, source: impl BrickSource + 'static) {
        self.source = Some(Arc::new(source));
        self.build_lod(self.coarse.len());
    }

    /// The brick at `coord` of mip `level`, as it reads: the one held, or else what the source
    /// gives, or else air. Level 0 is this store's own bricks. Panics if there is no such level.
    pub fn brick_at(&self, level: usize, coord: BrickCoord) -> Cow<'_, Brick> {
        match self.level(level).expect("no such level").brick(coord) {
            Some(brick) => Cow::Borrowed(brick),
            None => Cow::Owned(self.unheld(level, coord)),
        }
    }

    /// Whether the source says that every Voxel of level 0 under the brick at `coord` of `level`
    /// is solid, for a brick that is not held (see [`BrickSource::solid_below`]).
    pub fn source_solid_below(&self, level: usize, coord: BrickCoord) -> bool {
        self.source
            .as_ref()
            .is_some_and(|source| source.solid_below(level, coord))
    }

    /// Whether every Voxel of level 0 under the brick at `coord` of mip `level` is solid, exactly.
    /// A coarse brick that reads as full may still have air under it, since a coarse Voxel is
    /// solid when half of its children are: the pyramid keeps this beside the brick, made with it.
    /// Where the store holds no brick the source says (see [`BrickSource::solid_below`]).
    pub fn solid_below(&self, level: usize, coord: BrickCoord) -> bool {
        let Some(store) = self.level(level) else {
            return false;
        };
        match store.bricks.get(&coord) {
            Some(brick) if level == 0 => brick.is_full(),
            Some(_) => store.solid_below.contains(&coord),
            None if level == 0 => self.brick_at(0, coord).is_full(),
            None => self.source_solid_below(level, coord),
        }
    }

    /// The Voxel at `coord` of mip `level`, as [`Self::brick_at`] reads it: `None` for air.
    pub fn voxel_at(&self, level: usize, coord: VoxelCoord) -> Option<Material> {
        self.brick_at(level, coord.brick()).get(coord.local())
    }

    /// Replaces the pyramid with `coarse_levels` levels above this one. The Planet needs about
    /// 19 or 20 to reach from 10 cm Voxels to its disc.
    pub fn build_lod(&mut self, coarse_levels: usize) {
        self.coarse = vec![BrickStore::new(); coarse_levels];
        self.remake_above(self.bricks.keys().copied().collect(), false);
    }

    /// Number of mip levels, counting this one as level 0: 1 when there is no pyramid.
    pub fn level_count(&self) -> usize {
        1 + self.coarse.len()
    }

    /// The bricks held at mip level `level`, or `None` beyond the pyramid. Level 0 is this
    /// store. Reading it ignores the source: use [`Self::brick_at`] for that.
    pub fn level(&self, level: usize) -> Option<&BrickStore> {
        match level {
            0 => Some(self),
            n => self.coarse.get(n - 1),
        }
    }

    /// Makes the pyramid match the level 0 bricks at `changed` after they were changed (held,
    /// replaced or removed), by remaking only the coarse bricks over them and over the bricks
    /// that touch them, as far up as any brick reads differently. Does nothing without a
    /// pyramid. The result is the pyramid that [`Self::build_lod`] would make.
    pub fn refresh_lod(&mut self, changed: impl IntoIterator<Item = BrickCoord>) -> LodRefresh {
        self.remake_above(changed.into_iter().collect(), true)
    }

    /// FNV-1a hash of the pyramid: the number and [`Self::content_hash`] of each coarse level,
    /// finest first. Level 0 is left to `content_hash`, so what the two hash is disjoint.
    pub fn lod_hash(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for (n, level) in (1u32..).zip(&self.coarse) {
            let bytes = n
                .to_le_bytes()
                .into_iter()
                .chain(level.content_hash().to_le_bytes());
            for byte in bytes {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    /// Remakes the coarse bricks over the level 0 bricks `changed`, level by level.
    ///
    /// A coarse brick is made from the eight bricks it covers, and from the Voxels beside them:
    /// whether a Voxel of the eight is exposed to air depends on the Voxels next to it, in the
    /// bricks that touch theirs (see [`downsample`]). So a brick that changed makes the coarse
    /// bricks over it and over the six bricks touching it be remade, when `beside` is set. It
    /// is not set to build every brick afresh, where all are remade anyway.
    ///
    /// A level is remade only from the bricks of the level below that changed, held or read
    /// differently. When none did, nothing above it can have, and the rest is left alone.
    fn remake_above(&mut self, changed: BTreeSet<BrickCoord>, beside: bool) -> LodRefresh {
        let mut refresh = LodRefresh::default();
        let mut moved = changed;
        // The bricks whose `solid_below` changed, on the level just made. That is kept up past
        // where the bricks stop changing (a hole too small to show on a coarse level still keeps
        // the brick over it from being solid all the way down), but it remakes no brick.
        let mut flipped: BTreeSet<BrickCoord> = BTreeSet::new();
        for level in 1..=self.coarse.len() {
            let over: BTreeSet<_> = moved
                .iter()
                .flat_map(|&coord| {
                    let touching = beside.then(|| touching(coord)).into_iter().flatten();
                    std::iter::once(coord).chain(touching)
                })
                .map(BrickCoord::parent)
                .collect();
            let flags: BTreeSet<_> = flipped.iter().map(|c| c.parent()).collect();
            moved = BTreeSet::new();
            flipped = BTreeSet::new();
            for &coord in &over {
                let held = self.remake(level, coord);
                refresh.remade += 1;
                let (was_held, before) = match held {
                    Some(brick) => (true, brick),
                    None => (false, self.unheld(level, coord)),
                };
                let now_held = self.coarse[level - 1].bricks.contains_key(&coord);
                let read_differently = before != *self.brick_at(level, coord);
                if read_differently {
                    refresh.changed.push((level, coord));
                }
                if read_differently || was_held != now_held {
                    moved.insert(coord);
                }
            }
            for &coord in over.union(&flags) {
                if self.remake_solid_below(level, coord) {
                    flipped.insert(coord);
                }
            }
            if moved.is_empty() && flipped.is_empty() {
                break;
            }
        }
        refresh
    }

    /// Remakes brick `coord` of coarse level `level` from the eight bricks of the level below:
    /// held if any of them is held, and not held otherwise. Returns the brick that was held.
    fn remake(&mut self, level: usize, coord: BrickCoord) -> Option<Brick> {
        let finer = self.level(level - 1).expect("the level below exists");
        let children = coord.children();
        let brick = children
            .iter()
            .any(|c| finer.bricks.contains_key(c))
            .then(|| {
                let read = children.map(|child| self.brick_at(level - 1, child));
                let around = |at: [i32; 3]| {
                    self.brick_at(
                        level - 1,
                        BrickCoord([0, 1, 2].map(|a| children[0].0[a] + at[a])),
                    )
                };
                downsample(read.each_ref().map(|c| &**c), around)
            });
        let coarse = &mut self.coarse[level - 1].bricks;
        match brick {
            Some(brick) => coarse.insert(coord, brick),
            None => coarse.remove(&coord),
        }
    }

    /// Works out [`Self::solid_below`] of the held brick `coord` of coarse level `level` again,
    /// from the brick and the eight under it. Returns whether it changed.
    fn remake_solid_below(&mut self, level: usize, coord: BrickCoord) -> bool {
        let full = self.coarse[level - 1]
            .bricks
            .get(&coord)
            .is_some_and(Brick::is_full);
        let solid = full
            && coord
                .children()
                .iter()
                .all(|&c| self.solid_below(level - 1, c));
        let store = &mut self.coarse[level - 1];
        let was_solid = if solid {
            !store.solid_below.insert(coord)
        } else {
            store.solid_below.remove(&coord)
        };
        solid != was_solid
    }

    /// What the brick at `coord` of mip `level` reads as where none is held there.
    fn unheld(&self, level: usize, coord: BrickCoord) -> Brick {
        match &self.source {
            Some(source) => source.brick(level, coord),
            None => Brick::AIR,
        }
    }

    /// Coordinates of all bricks held, in ascending order.
    pub fn sorted_coords(&self) -> Vec<BrickCoord> {
        let mut coords: Vec<_> = self.bricks.keys().copied().collect();
        coords.sort_unstable();
        coords
    }

    /// The Voxel at `coord`: `None` for air, including where no brick is held.
    pub fn get(&self, coord: VoxelCoord) -> Option<Material> {
        self.brick(coord.brick())?.get(coord.local())
    }

    /// Sets the Voxel at `coord`, creating its brick if needed. Setting air where no brick is
    /// held allocates nothing.
    pub fn set(&mut self, coord: VoxelCoord, voxel: Option<Material>) {
        let brick = coord.brick();
        if voxel.is_none() && !self.bricks.contains_key(&brick) {
            return;
        }
        self.bricks
            .entry(brick)
            .or_insert(Brick::AIR)
            .set(coord.local(), voxel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lod::Interior;

    const ROCK: Material = Material(3);

    #[test]
    fn an_empty_store_reads_air_everywhere() {
        let store = BrickStore::new();
        assert_eq!(store.get(VoxelCoord([5, -5, 1_000_000])), None);
        assert!(store.is_empty());
    }

    #[test]
    fn set_then_get_by_voxel_coordinate_including_negatives() {
        let mut store = BrickStore::new();
        for v in [[0, 0, 0], [-1, -1, -1], [7, 8, -9], [-100, 200, -300]] {
            store.set(VoxelCoord(v), Some(ROCK));
            assert_eq!(store.get(VoxelCoord(v)), Some(ROCK), "{v:?}");
        }
        assert_eq!(store.get(VoxelCoord([1, 0, 0])), None);
        assert_eq!(store.len(), 4);
    }

    #[test]
    fn voxels_on_either_side_of_zero_live_in_different_bricks() {
        let mut store = BrickStore::new();
        store.set(VoxelCoord([-1, 0, 0]), Some(ROCK));
        store.set(VoxelCoord([0, 0, 0]), Some(Material(4)));
        assert_eq!(
            store.sorted_coords(),
            [BrickCoord([-1, 0, 0]), BrickCoord([0, 0, 0])]
        );
        assert_eq!(store.get(VoxelCoord([-1, 0, 0])), Some(ROCK));
        assert_eq!(store.get(VoxelCoord([0, 0, 0])), Some(Material(4)));
    }

    #[test]
    fn setting_air_in_empty_space_allocates_no_brick() {
        let mut store = BrickStore::new();
        store.set(VoxelCoord([9, 9, 9]), None);
        assert!(store.is_empty());
    }

    #[test]
    fn remove_makes_a_brick_read_as_air_again() {
        let mut store = BrickStore::new();
        store.set(VoxelCoord([1, 2, 3]), Some(ROCK));
        assert!(store.remove(BrickCoord([0, 0, 0])).is_some());
        assert_eq!(store.get(VoxelCoord([1, 2, 3])), None);
        assert!(store.remove(BrickCoord([0, 0, 0])).is_none());
    }

    #[test]
    fn content_hash_ignores_insertion_order() {
        let (mut a, mut b) = (BrickStore::new(), BrickStore::new());
        a.insert(BrickCoord([1, 0, 0]), Brick::solid(ROCK));
        a.insert(BrickCoord([-1, 0, 0]), Brick::solid(Material(4)));
        b.insert(BrickCoord([-1, 0, 0]), Brick::solid(Material(4)));
        b.insert(BrickCoord([1, 0, 0]), Brick::solid(ROCK));
        assert_eq!(a.content_hash(), b.content_hash());
    }

    #[test]
    fn content_hash_tells_a_brick_held_as_air_from_one_not_held() {
        let (mut held, mut absent) = (BrickStore::new(), BrickStore::new());
        for store in [&mut held, &mut absent] {
            store.insert(BrickCoord([1, 0, 0]), Brick::solid(ROCK));
        }
        held.insert(BrickCoord([9, 9, 9]), Brick::AIR);
        assert_ne!(held.content_hash(), absent.content_hash());
    }

    #[test]
    fn content_hash_sees_where_a_brick_is_and_what_is_in_it() {
        let hash = |coord, material| {
            let mut store = BrickStore::new();
            store.insert(BrickCoord(coord), Brick::solid(Material(material)));
            store.content_hash()
        };
        let base = hash([1, 0, 0], 3);
        assert_ne!(base, hash([0, 1, 0], 3));
        assert_ne!(base, hash([1, 0, 0], 4));
        assert_ne!(base, BrickStore::new().content_hash());
    }

    #[test]
    fn sorted_coords_ignore_insertion_order() {
        let mut store = BrickStore::new();
        for c in [[2, 0, 0], [-1, 5, 0], [0, 0, 1], [-1, 4, 7]] {
            store.insert(BrickCoord(c), Brick::solid(ROCK));
        }
        assert_eq!(
            store.sorted_coords(),
            [
                BrickCoord([-1, 4, 7]),
                BrickCoord([-1, 5, 0]),
                BrickCoord([0, 0, 1]),
                BrickCoord([2, 0, 0]),
            ]
        );
    }

    /// A 16-Voxel cube of solid ROCK: the eight level 0 bricks 0..=1 on each axis, which are
    /// the children of level 1 brick (0, 0, 0).
    fn block() -> BrickStore {
        let mut store = BrickStore::new();
        for i in 0..8 {
            store.insert(BrickCoord([i & 1, i >> 1 & 1, i >> 2]), Brick::solid(ROCK));
        }
        store
    }

    /// 4×4×4 solid bricks about the origin, in three Materials that vary from brick to brick.
    fn banded() -> BrickStore {
        let mut store = BrickStore::new();
        for i in 0..64 {
            let at = [i % 4 - 2, i / 4 % 4 - 2, i / 16 - 2];
            let material = Material(1 + at.iter().sum::<i32>().rem_euclid(3) as u16);
            store.insert(BrickCoord(at), Brick::solid(material));
        }
        store
    }

    /// The coarse bricks, as `(level, coordinate)`, that read differently in the two stores.
    fn differing(before: &BrickStore, after: &BrickStore) -> Vec<(usize, BrickCoord)> {
        let mut differing = Vec::new();
        for level in 1..before.level_count() {
            let mut coords = before.level(level).unwrap().sorted_coords();
            coords.extend(after.level(level).unwrap().sorted_coords());
            coords.sort();
            coords.dedup();
            differing.extend(
                coords
                    .into_iter()
                    .filter(|&c| before.brick_at(level, c) != after.brick_at(level, c))
                    .map(|c| (level, c)),
            );
        }
        differing
    }

    /// The pyramid that `build_lod` makes of `store`'s level 0 bricks and source.
    fn rebuilt(store: &BrickStore) -> BrickStore {
        let mut again = store.clone();
        again.build_lod(store.level_count() - 1);
        again
    }

    #[test]
    fn a_store_without_a_pyramid_has_one_level_and_refreshing_it_does_nothing() {
        let mut store = block();
        assert_eq!(store.level_count(), 1);
        assert!(store.level(1).is_none());
        assert_eq!(
            store.refresh_lod([BrickCoord([0, 0, 0])]),
            LodRefresh::default()
        );
        assert_eq!(store.lod_hash(), BrickStore::new().lod_hash());
    }

    #[test]
    fn level_0_of_a_pyramid_is_the_store_itself() {
        let mut store = block();
        store.build_lod(3);
        assert_eq!(store.level_count(), 4);
        assert_eq!(store.level(0).unwrap().len(), 8);
        assert!(store.level(4).is_none());
    }

    #[test]
    fn a_pyramid_does_not_change_the_hash_of_level_0() {
        let plain = block();
        let mut pyramid = block();
        pyramid.build_lod(4);
        assert_eq!(pyramid.content_hash(), plain.content_hash());
        assert_ne!(pyramid.lod_hash(), plain.lod_hash());
    }

    #[test]
    fn a_cube_of_stone_shrinks_by_half_at_each_level() {
        let mut store = block();
        store.build_lod(4);
        let level = |n| store.level(n).unwrap();
        // Level 1 brick 0 is the eight bricks as one solid brick.
        assert_eq!(level(1).sorted_coords(), [BrickCoord([0; 3])]);
        assert!(level(1).brick(BrickCoord([0; 3])).unwrap().is_uniform());
        // Each level up, the cube is half as many Voxels wide: 8, 4, 2 then 1.
        for (n, width) in [(1, 8), (2, 4), (3, 2), (4, 1)] {
            for v in 0..9 {
                let inside = v < width;
                let want = inside.then_some(ROCK);
                assert_eq!(
                    store.voxel_at(n, VoxelCoord([v, v, v])),
                    want,
                    "level {n} Voxel {v}"
                );
                assert_eq!(
                    store.voxel_at(n, VoxelCoord([v, 0, width])),
                    None,
                    "level {n} Voxel {v}"
                );
            }
        }
        assert_eq!(level(4).len(), 1);
    }

    #[test]
    fn a_lone_voxel_is_too_few_to_show_in_the_levels_above() {
        let mut store = BrickStore::new();
        store.set(VoxelCoord([3, 3, 3]), Some(ROCK));
        store.build_lod(3);
        // The coarse bricks are held, as they cover a held brick, but read as air.
        for n in 1..=3 {
            assert!(store.brick_at(n, BrickCoord([0; 3])).is_air(), "level {n}");
        }
    }

    #[test]
    fn building_again_replaces_the_old_pyramid() {
        let mut store = block();
        store.build_lod(4);
        let four = store.lod_hash();
        store.build_lod(1);
        assert_eq!(store.level_count(), 2);
        let mut fresh = block();
        fresh.build_lod(1);
        assert_eq!(store.lod_hash(), fresh.lod_hash());
        assert_ne!(store.lod_hash(), four);
        // What is left of level 1 is its content, and not just the right number of levels.
        assert!(store.brick_at(1, BrickCoord([0; 3])).is_uniform());
        assert_eq!(store.voxel_at(1, VoxelCoord([7, 7, 7])), Some(ROCK));
        store.build_lod(0);
        assert_eq!(store.level_count(), 1);
        assert_eq!(store.lod_hash(), BrickStore::new().lod_hash());
    }

    #[test]
    fn refreshing_remakes_the_ancestors_of_a_changed_brick_and_reports_what_differs() {
        let mut store = block();
        store.build_lod(4);
        let before = store.clone();
        store.insert(BrickCoord([1, 0, 1]), Brick::AIR);
        let refresh = store.refresh_lod([BrickCoord([1, 0, 1])]);
        // Four per level: the ancestor, and the ones over the bricks beside it that are not in
        // its parent. The hole is seen up to level 3, so level 4 is remade and then it stops.
        assert_eq!(refresh.remade, 4 * 4);
        // The hole shows up to level 3, and at level 4 it is one child in eight: still rock.
        assert_eq!(
            refresh.changed,
            [
                (1, BrickCoord([0; 3])),
                (2, BrickCoord([0; 3])),
                (3, BrickCoord([0; 3]))
            ]
        );
        assert_eq!(refresh.changed, differing(&before, &store));
        assert_eq!(store.lod_hash(), rebuilt(&store).lod_hash());
    }

    #[test]
    fn refreshing_a_brick_that_did_not_change_reports_nothing() {
        let mut store = block();
        store.build_lod(4);
        let refresh = store.refresh_lod([BrickCoord([1, 0, 1])]);
        assert_eq!(refresh.remade, 4);
        assert!(refresh.changed.is_empty());
    }

    #[test]
    fn bricks_that_share_ancestors_or_touch_ones_that_do_are_remade_once() {
        let mut store = block();
        store.build_lod(4);
        // None of these change anything, so only level 1 is remade: the parents of the bricks
        // and of the bricks touching them. Bricks -1 and 0 never share a parent, and each of the
        // two touches the other, so the parents are those of -2 to 1 along x, and the y and z
        // of the two bricks and of the ones beside them.
        let refresh = store.refresh_lod([BrickCoord([-1, 0, 0]), BrickCoord([0, 0, 0])]);
        assert_eq!(refresh.remade, 6);
        // Four children of one parent, its lower layer: that parent and the one beyond each of
        // its faces but the top, which the upper layer of children is in.
        let refresh =
            store.refresh_lod([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]].map(BrickCoord));
        assert_eq!(refresh.remade, 6);
        // Twenty-seven neighbours span two parents per axis: eight, and twelve more that only
        // the bricks touching the cube are in (a brick 2 along one axis, and not along two).
        let cube = (0..27).map(|i| BrickCoord([i % 3 - 1, i / 3 % 3 - 1, i / 9 - 1]));
        assert_eq!(store.refresh_lod(cube).remade, 20);
    }

    #[test]
    fn those_ancestors_are_all_that_differs_when_the_bricks_really_changed() {
        let mut store = block();
        for far in [[40, 0, 0], [-33, -33, 17], [2, 2, 2]] {
            store.insert(BrickCoord(far), Brick::solid(Material(5)));
        }
        store.build_lod(4);
        let before = store.clone();
        // The eight bricks around the origin, where -1 and 0 never share an ancestor.
        let around: Vec<_> = (0..8)
            .map(|i| BrickCoord([i & 1, i >> 1 & 1, i >> 2].map(|b| b - 1)))
            .collect();
        for &coord in &around {
            store.insert(coord, Brick::AIR);
        }
        store.insert(BrickCoord([0, 0, 0]), Brick::solid(Material(4)));
        let refresh = store.refresh_lod(around.iter().copied());
        assert_eq!(refresh.remade, 8 * 4);

        let differing = differing(&before, &store);
        assert!(!differing.is_empty());
        assert_eq!(refresh.changed, differing);
        // Everything that differs is over the bricks changed: coarse bricks -1 and 0 on each axis.
        let over = |&(_, c): &(usize, BrickCoord)| c.0.iter().all(|&a| a == 0 || a == -1);
        assert!(differing.iter().all(over), "{differing:?}");
    }

    #[test]
    fn a_refreshed_pyramid_is_the_one_a_rebuild_would_make() {
        let mut store = banded();
        store.build_lod(5);
        for centre in [
            [0, 0, 0],
            [1, 1, 1],
            [-8, -8, -8],
            [-1, 0, 7],
            [-2, -3, -4],
            [9, 8, 7],
        ] {
            let mut changed = BTreeSet::new();
            for d in 0..27 {
                let at = [0, 1, 2].map(|a| centre[a] + [d % 3, d / 3 % 3, d / 9][a] - 1);
                store.set(VoxelCoord(at), None);
                changed.insert(VoxelCoord(at).brick());
            }
            store.refresh_lod(changed);
            assert_eq!(
                store.lod_hash(),
                rebuilt(&store).lod_hash(),
                "after {centre:?}"
            );
        }
    }

    const SOIL: Material = Material(9);
    const BEDROCK: Material = Material(2);

    /// Ground up to z = 16 over the 16 Voxel wide square about (8, 8): bedrock with a skin of
    /// soil at z = 15. Its bricks are the four at z = 1 and four at z = 0, which are the
    /// children of level 1 brick (0, 0, 0).
    fn ground() -> BrickStore {
        let mut store = BrickStore::new();
        for i in 0..8 {
            let at = BrickCoord([i & 1, i >> 1 & 1, i >> 2]);
            let mut brick = Brick::solid(BEDROCK);
            if at.0[2] == 1 {
                for v in 0..64 {
                    brick.set([v % 8, v / 8, 7], Some(SOIL));
                }
            }
            store.insert(at, brick);
        }
        store
    }

    #[test]
    fn a_coarse_brick_is_remade_when_a_brick_next_to_its_children_changes() {
        let mut store = ground();
        store.build_lod(3);
        // Air over the soil, so that it is the surface.
        assert_eq!(store.voxel_at(1, VoxelCoord([3, 3, 7])), Some(SOIL));
        // Level 0 brick (0, 0, 2) is not a child of level 1 brick (0, 0, 0), but it is above
        // two of them. With bedrock on it the soil is no longer the surface there.
        store.insert(BrickCoord([0, 0, 2]), Brick::solid(BEDROCK));
        let refresh = store.refresh_lod([BrickCoord([0, 0, 2])]);
        assert_eq!(store.voxel_at(1, VoxelCoord([3, 3, 7])), Some(BEDROCK));
        assert!(
            refresh.changed.contains(&(1, BrickCoord([0; 3]))),
            "{refresh:?}"
        );
        assert_eq!(store.lod_hash(), rebuilt(&store).lod_hash());
        // And the same on the way back.
        store.remove(BrickCoord([0, 0, 2]));
        store.refresh_lod([BrickCoord([0, 0, 2])]);
        assert_eq!(store.voxel_at(1, VoxelCoord([3, 3, 7])), Some(SOIL));
        assert_eq!(store.lod_hash(), rebuilt(&store).lod_hash());
    }

    /// Ground 128 Voxels across, with its surface at z = `top`: 4 Voxels of soil, and bedrock
    /// under them as far down as z = -48.
    fn slab(top: i32) -> BrickStore {
        let material = |z: i32| match z {
            z if z >= top => None,
            z if z >= top - 4 => Some(SOIL),
            _ => Some(BEDROCK),
        };
        let mut store = BrickStore::new();
        for bz in -6..=(top - 1).div_euclid(8) {
            let layer: Vec<_> = (0..8).map(|v| material(8 * bz + v)).collect();
            let mut brick = Brick::AIR;
            for i in 0..512 {
                brick.set([i % 8, i / 8 % 8, i / 64].map(|a| a as u8), layer[i / 64]);
            }
            for bx in -8..8 {
                for by in -8..8 {
                    store.insert(BrickCoord([bx, by, bz]), brick.clone());
                }
            }
        }
        store
    }

    #[test]
    fn soil_four_voxels_thick_over_rock_reads_as_soil_at_every_level_wherever_its_surface_is() {
        // The surface at every height from 18 to 34, which puts it on every boundary between
        // coarse Voxels from level 1 to level 4 and on those of bricks (24 and 32), and has it
        // between them otherwise. Bedrock has the lower ID, so a tie never favours the soil.
        for top in 18..=34 {
            let mut store = slab(top);
            store.build_lod(5);
            for level in 1..=5 {
                // The highest solid Voxel in the column over the middle of the slab.
                let surface = (-8..=(top >> level) + 1)
                    .rev()
                    .find_map(|z| store.voxel_at(level, VoxelCoord([0, 0, z])));
                assert_eq!(surface, Some(SOIL), "surface at {top}, level {level}");
            }
        }
    }

    #[test]
    fn coarse_bricks_are_held_exactly_over_held_bricks() {
        let mut store = block();
        store.build_lod(2);
        assert_eq!(store.level(1).unwrap().len(), 1);
        assert_eq!(store.level(2).unwrap().len(), 1);
        // Held as air, a brick still has its ancestors held, reading as air.
        for coord in store.sorted_coords() {
            store.insert(coord, Brick::AIR);
        }
        store.refresh_lod(store.sorted_coords());
        assert_eq!(store.level(2).unwrap().len(), 1);
        assert!(store.brick_at(1, BrickCoord([0; 3])).is_air());
        assert!(store.brick_at(2, BrickCoord([0; 3])).is_air());
        // Removed, they have none.
        for coord in store.sorted_coords() {
            store.remove(coord);
        }
        store.refresh_lod(block().sorted_coords());
        assert!(store.level(1).unwrap().is_empty());
        assert!(store.level(2).unwrap().is_empty());
        assert_eq!(store.lod_hash(), rebuilt(&store).lod_hash());
    }

    /// An interior of radius 100, and so with coarse bricks over its edge from level 4 up.
    const INTERIOR: Interior = Interior {
        radius: 100,
        material: ROCK,
    };

    #[test]
    fn a_source_decides_what_a_brick_that_is_not_held_reads_as() {
        let mut store = BrickStore::new();
        store.build_lod(3);
        assert!(store.brick_at(2, BrickCoord([0; 3])).is_air());
        store.set_source(INTERIOR);
        // Wholly inside the ball: level 0 brick 0, and level 2 brick 0 (Voxels 0..=31).
        assert_eq!(store.voxel_at(0, VoxelCoord([3, 3, 3])), Some(ROCK));
        assert_eq!(store.voxel_at(2, VoxelCoord([7, 7, 7])), Some(ROCK));
        // Not wholly inside: level 0 brick 20 (Voxels 160..), and level 3 brick 4 (Voxels 256..).
        assert_eq!(store.voxel_at(0, VoxelCoord([160, 0, 0])), None);
        assert_eq!(store.voxel_at(3, VoxelCoord([32, 0, 0])), None);
        // The held bricks are only those held: reading as rock does not hold it.
        assert_eq!(store.get(VoxelCoord([3, 3, 3])), None);
        assert!(store.is_empty() && store.level(1).unwrap().is_empty());
    }

    #[test]
    fn digging_a_brick_out_of_the_interior_shows_in_the_levels_over_it() {
        let mut store = BrickStore::new();
        store.set_source(INTERIOR);
        store.build_lod(2);

        // A level 0 brick dug out to air: an eighth of level 1 brick 0 and of level 2 brick 0,
        // with the rock of the interior around it.
        store.insert(BrickCoord([0; 3]), Brick::AIR);
        let refresh = store.refresh_lod([BrickCoord([0; 3])]);
        // Four a level: the corner of eight parents that the brick is at touches four.
        assert_eq!(refresh.remade, 2 * 4);
        assert_eq!(
            refresh.changed,
            [(1, BrickCoord([0; 3])), (2, BrickCoord([0; 3]))]
        );
        assert_eq!(store.voxel_at(1, VoxelCoord([3, 3, 3])), None);
        assert_eq!(store.voxel_at(1, VoxelCoord([4, 0, 0])), Some(ROCK));
        assert_eq!(store.voxel_at(2, VoxelCoord([1, 1, 1])), None);
        assert_eq!(store.voxel_at(2, VoxelCoord([2, 0, 0])), Some(ROCK));

        // Rock put back: the coarse bricks are held still, but read as they did at first.
        store.insert(BrickCoord([0; 3]), Brick::solid(ROCK));
        let refresh = store.refresh_lod([BrickCoord([0; 3])]);
        assert_eq!(refresh.changed.len(), 2);
        assert_eq!(store.voxel_at(1, VoxelCoord([3, 3, 3])), Some(ROCK));
        // Taken away, they are gone, and the pyramid is as bare as when it was built.
        store.remove(BrickCoord([0; 3]));
        store.refresh_lod([BrickCoord([0; 3])]);
        assert!(store.level(1).unwrap().is_empty() && store.level(2).unwrap().is_empty());
        let mut bare = BrickStore::new();
        bare.set_source(INTERIOR);
        bare.build_lod(2);
        assert_eq!(store.lod_hash(), bare.lod_hash());
    }

    #[test]
    fn taking_back_an_edit_leaves_the_pyramid_as_it_was_even_over_the_edge_of_the_interior() {
        let mut store = BrickStore::new();
        store.set_source(INTERIOR);
        store.build_lod(9);
        let bare = store.lod_hash();
        // Level 4 and up have coarse bricks that are only partly inside the ball. Digging at
        // the origin remakes them from the bricks around, which the source says are air.
        store.insert(BrickCoord([0; 3]), Brick::AIR);
        let dug = store.refresh_lod([BrickCoord([0; 3])]);
        assert_eq!(dug.remade, 9 * 4);
        assert_ne!(store.lod_hash(), bare);
        assert_eq!(store.lod_hash(), rebuilt(&store).lod_hash());
        store.remove(BrickCoord([0; 3]));
        let undone = store.refresh_lod([BrickCoord([0; 3])]);
        assert_eq!(undone.remade, 9 * 4);
        assert_eq!(store.lod_hash(), bare);
        assert_eq!(dug.changed.len(), undone.changed.len());
    }

    /// Rock out to `rock` Voxels from the origin. Bricks wholly within `core` are held only if
    /// `hold_core`.
    fn planet(rock: i32, core: i32, hold_core: bool) -> BrickStore {
        let mut store = BrickStore::new();
        for i in 0..12 * 12 * 12 {
            let coord = BrickCoord([i % 12 - 6, i / 12 % 12 - 6, i / 144 - 6]);
            if !hold_core && coord.within_ball(core) {
                continue;
            }
            let mut brick = Brick::AIR;
            for v in 0..512 {
                let local = [v & 7, v >> 3 & 7, v >> 6].map(|a| a as u8);
                let at = coord.voxel(local).0.map(i64::from);
                if at.iter().map(|a| a * a).sum::<i64>() <= i64::from(rock * rock) {
                    brick.set(local, Some(ROCK));
                }
            }
            if !brick.is_air() {
                store.insert(coord, brick);
            }
        }
        store
    }

    #[test]
    fn an_interior_with_its_crust_held_makes_the_pyramid_of_a_planet_that_holds_it_all() {
        // The core and the crust it needs, but not the bricks wholly inside the core.
        let mut implied = planet(36, 24, false);
        let core = Interior {
            radius: 24,
            material: ROCK,
        };
        implied.set_source(core);
        implied.build_lod(7);
        let mut whole = planet(36, 24, true);
        whole.build_lod(7);
        assert!(whole.len() > implied.len() + 10);

        for level in 1..=7 {
            let mut coords = whole.level(level).unwrap().sorted_coords();
            coords.extend(implied.level(level).unwrap().sorted_coords());
            coords.sort();
            coords.dedup();
            assert!(!coords.is_empty());
            for coord in coords {
                assert_eq!(
                    whole.brick_at(level, coord),
                    implied.brick_at(level, coord),
                    "level {level} {coord:?}"
                );
            }
        }
    }

    #[test]
    fn lod_hash_ignores_the_order_the_bricks_were_inserted_in() {
        let mut backwards = BrickStore::new();
        for coord in block().sorted_coords().into_iter().rev() {
            backwards.insert(coord, Brick::solid(ROCK));
        }
        let mut forwards = block();
        forwards.build_lod(3);
        backwards.build_lod(3);
        assert_eq!(forwards.lod_hash(), backwards.lod_hash());
    }

    #[test]
    fn lod_hash_sees_the_material_in_a_coarse_brick_and_the_number_of_levels() {
        let hash = |material, levels| {
            let mut store = BrickStore::new();
            store.insert(BrickCoord([5, 5, 5]), Brick::solid(Material(material)));
            store.build_lod(levels);
            store.lod_hash()
        };
        assert_ne!(hash(3, 3), hash(4, 3));
        assert_ne!(hash(3, 3), hash(3, 2));
    }

    /// A block of rock 4 bricks on a side, with a pyramid of two levels.
    fn rock_block() -> BrickStore {
        let mut store = BrickStore::new();
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    store.insert(BrickCoord([x, y, z]), Brick::solid(ROCK));
                }
            }
        }
        store.build_lod(2);
        store
    }

    #[test]
    fn a_coarse_brick_is_solid_below_only_if_every_voxel_under_it_is() {
        let mut store = rock_block();
        let top = BrickCoord([0, 0, 0]);
        assert!(
            store.solid_below(0, top) && store.solid_below(1, top) && store.solid_below(2, top)
        );
        // A hole of one Voxel does not change what the coarse bricks read as, which is a majority,
        // but they are no longer solid all the way down: not the ones over it.
        let before = store.brick_at(2, top).into_owned();
        let hole = VoxelCoord([13, 13, 13]).brick();
        store.set(VoxelCoord([13, 13, 13]), None);
        let refresh = store.refresh_lod([hole]);
        assert_eq!(
            *store.brick_at(2, top),
            before,
            "the level 2 brick reads as it did"
        );
        assert!(refresh.remade >= 2);
        assert!(!store.solid_below(0, hole));
        assert!(!store.solid_below(1, hole.parent()));
        assert!(!store.solid_below(2, top));
        assert!(store.solid_below(0, BrickCoord([2, 2, 2])));
        assert!(store.solid_below(1, BrickCoord([1, 1, 1])));
        // Refreshing keeps it as building the pyramid afresh makes it, with the hole.
        let mut fresh = store.clone();
        fresh.build_lod(2);
        assert_eq!(store.coarse[0].solid_below, fresh.coarse[0].solid_below);
        assert_eq!(store.coarse[1].solid_below, fresh.coarse[1].solid_below);
        // Filled again, it is.
        store.set(VoxelCoord([13, 13, 13]), Some(ROCK));
        store.refresh_lod([hole]);
        assert!(store.solid_below(2, top));
        let mut fresh = store.clone();
        fresh.build_lod(2);
        assert_eq!(store.coarse[1].solid_below, fresh.coarse[1].solid_below);
    }

    #[test]
    fn where_the_store_holds_nothing_the_source_says_what_is_solid_below() {
        let mut ball = BrickStore::new();
        ball.set_source(Interior {
            radius: 100,
            material: ROCK,
        });
        ball.build_lod(2);
        assert!(ball.solid_below(2, BrickCoord([0, 0, 0])));
        assert!(!ball.solid_below(2, BrickCoord([5, 5, 5])));
    }
}
