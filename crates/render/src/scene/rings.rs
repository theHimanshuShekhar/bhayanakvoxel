//! Rings: the way a [`Scene`] holds the Planet at every distance from the camera (ADR 0002).
//!
//! A scene of rings has a level for each level of the brick store's pyramid that the LOD bands
//! use (see [`crate::lod`]), and each level lists only the bricks that the bands select for it
//! ([`Selection`]): a sparse set of regions of its own, whatever its distance and however the
//! camera moves, not a box. A region lists just the bricks of it that are selected, so a region
//! half in the band of one level and half in the next is listed by both, with different bricks.
//!
//! The bands partition space, so every part of it is seen at one level. The walk in
//! `trace.wgsl` goes through the levels finest first and takes the nearest hit. It skips the
//! *hole* of a level: the regions inside the ball about the camera that no brick of the level
//! reaches (a cube in it, [`Scene::hole`]), so a ray that starts near the camera goes to the
//! level that has something without stepping through empty regions.
//!
//! A level is only walked inside the smallest box of regions that has all the regions it lists
//! (`Level::occupied_first` and `occupied_end`, kept up to date by every operation that changes
//! them). Levels that have nothing, and the sky above and below the crust, then cost a clip and
//! no steps.
//!
//! The buffers are allocated once, with room to spare ([`Room`] says how much when it is given;
//! otherwise it is what the bricks selected need and half as much again), so they keep their
//! size as the camera moves. [`Scene::recentre`] follows the camera: it asks the selection what
//! changed and packs again only the regions that gained or lost a brick. [`Scene::update`] takes
//! Edits in, from the bricks of level 0 that they changed, and rewrites only the bricks that
//! differ.
//!
//! The way this can fail is that the words run out ([`RingsFull`]). A scene that was sized by the
//! bricks it needs packs itself again, bigger, when that happens, and says so
//! ([`SceneUpdate::Rebuilt`]), which makes the GPU copy new buffers; one with a [`Room`] given
//! fails, and is not to be used again: its selection and its pending patch no longer match what
//! is on the GPU.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::ops::Range;

use brick_store::{Brick, BrickCoord, BrickStore, BrickView};

use super::select::{self, Selection};
use super::{
    Bound, Level, MAX_LEVELS, REGION_BRICKS, REGION_FIRST_REF, REGION_VOXELS, REGION_WORDS,
    SOLID_FLAG, Scene, SceneUpdate, bit_of, encode_mixed, region_of,
};
use crate::lod::Bands;

/// How many words the buffers of a scene of rings have, for each level, when the scene is given
/// room instead of being sized by the bricks it holds. The levels share what they have, except
/// for the regions, which each level has its own table for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Room {
    /// Regions each level can list.
    pub regions_per_ring: usize,
    /// Words of `brick_refs` for each level: one for every brick it lists that is not inside a
    /// region of one Material.
    pub refs_per_ring: usize,
    /// Words of `bricks` for each level. A brick of two Materials with a surface in it takes 34
    /// to 50 words, so the default is room for 40 to 60 thousand of them in each level.
    pub words_per_ring: usize,
}

impl Default for Room {
    fn default() -> Self {
        Self {
            regions_per_ring: 4096,
            refs_per_ring: 1 << 18,
            words_per_ring: 1 << 21,
        }
    }
}

/// The rings do not fit in their buffers: the Planet is denser than they were sized for, or the
/// words that are free are in pieces too small for a region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingsFull;

impl fmt::Display for RingsFull {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the rings do not fit in their buffers")
    }
}

impl std::error::Error for RingsFull {}

/// Hands out ranges of the words of one buffer, first fit, and takes them back.
#[derive(Debug)]
struct Pool {
    /// The free ranges, ascending, and never touching: ones that touch are one range.
    free: Vec<Range<usize>>,
}

impl Pool {
    fn new(capacity: usize) -> Self {
        Self {
            free: std::iter::once(0..capacity).collect(),
        }
    }

    /// The start of `len` words that are free, which are now taken, if there are that many in a
    /// row.
    fn take(&mut self, len: usize) -> Option<usize> {
        let at = self.free.iter().position(|r| r.len() >= len)?;
        let start = self.free[at].start;
        if self.free[at].len() == len {
            self.free.remove(at);
        } else {
            self.free[at].start += len;
        }
        Some(start)
    }

    /// Gives `len` words from `start` back, joining them to the free ranges they touch.
    fn give(&mut self, start: usize, len: usize) {
        let at = self.free.partition_point(|r| r.start < start);
        let after = self.free.get(at).is_some_and(|r| r.start == start + len);
        let before = at > 0 && self.free[at - 1].end == start;
        match (before, after) {
            (true, true) => {
                self.free[at - 1].end = self.free[at].end;
                self.free.remove(at);
            }
            (true, false) => self.free[at - 1].end = start + len,
            (false, true) => self.free[at].start = start,
            (false, false) => self.free.insert(at, start..start + len),
        }
    }
}

/// How a scene of rings is sized.
#[derive(Clone, Copy, Debug)]
enum Sizing {
    /// By a [`Room`] that was given: a scene that does not fit is an error.
    Given(Room),
    /// By what the bricks selected need, and this many times as much, so that the camera can go
    /// on to where there are more. A scene that does not fit is packed again, bigger.
    Fitted(f64),
}

/// What a scene of rings keeps besides the buffers.
#[derive(Debug)]
pub(super) struct Rings {
    sizing: Sizing,
    /// The records of `regions` that no region uses. There is one for every region that the
    /// levels can list, so a region always finds one.
    free: Vec<usize>,
    /// Regions each level can list, and how many it does.
    capacity: Vec<usize>,
    listed: Vec<usize>,
    refs: Pool,
    data: Pool,
    /// The words of a brick on their way into `bricks`.
    scratch: Vec<u32>,
    /// What has been written since the last patch was handed out.
    patch: Patch,
    selection: Selection,
    /// The ball the selection looked in for the Planet (the scene's own bound may be taken off).
    ball: Bound,
    /// The coarsest level with a brick, as the header last said.
    coarsest: Option<usize>,
}

/// A [`RingPatch`] in the making.
#[derive(Debug, Default)]
struct Patch {
    regions: BTreeSet<usize>,
    table: BTreeSet<usize>,
    brick_refs: Vec<Range<usize>>,
    bricks: Vec<Range<usize>>,
    levels: bool,
}

/// What changed in a scene of rings, for copying to a GPU copy of it. Everything else is as
/// before.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RingPatch {
    /// The records of `regions` that were written, by index.
    pub regions: Vec<usize>,
    /// The slots of `table` that were written, by index.
    pub table: Vec<usize>,
    /// The words of `brick_refs` and `bricks` that were written, as ascending ranges that do not
    /// touch.
    pub brick_refs: Vec<Range<usize>>,
    pub bricks: Vec<Range<usize>>,
    /// Whether [`Scene::levels`] changed: the boxes walked, the holes, or the ball that rays are
    /// clipped to.
    pub levels: bool,
}

impl RingPatch {
    /// Words that copying this patch to the GPU writes, apart from the header.
    pub fn words(&self) -> usize {
        let ranges = |r: &[Range<usize>]| r.iter().map(Range::len).sum::<usize>();
        self.regions.len() * REGION_WORDS
            + self.table.len() * 4
            + ranges(&self.brick_refs)
            + ranges(&self.bricks)
    }
}

/// `ranges` in ascending order, with those that overlap or touch made one.
fn merged(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.sort_by_key(|r| r.start);
    let mut out: Vec<Range<usize>> = Vec::new();
    for range in ranges.into_iter().filter(|r| !r.is_empty()) {
        match out.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => out.push(range),
        }
    }
    out
}

/// The bricks of `selection` or of `keys`, by the level and region they are in.
type ByRegion = BTreeMap<(usize, [i32; 3]), BTreeSet<BrickCoord>>;

fn group(keys: impl IntoIterator<Item = (usize, BrickCoord)>) -> ByRegion {
    let mut by_region = ByRegion::new();
    for (level, coord) in keys {
        by_region
            .entry((level, region_of(coord)))
            .or_default()
            .insert(coord);
    }
    by_region
}

/// How far the camera can go in one move, in `d0`, before the scene is made again instead of
/// patched. Patching costs about in proportion to the distance (8 ms for 100 Voxels, 90 ms for
/// 1000, 170 ms for 2000 on the test Planet), a new scene about 200 ms wherever it is, and less
/// the further out the camera is, and they meet at about two `d0` (2056 Voxels).
const JUMP: f64 = 2.0;

/// Regions of bricks read on the threads before they are packed.
const PREFETCH_REGIONS: usize = 256;

/// Regions that the bricks of one level can be in, at the most, wherever the camera is: those of
/// its band, which is about as many at every level (a ring twice as wide has bricks twice as wide;
/// see ADR 0002), three times over for the layers of the crust; or those of the ball the Planet
/// is in, if that is less.
fn shell_regions(bands: Bands, bound: Bound, level: usize) -> usize {
    let size = f64::from(REGION_VOXELS << level);
    let band = std::f64::consts::PI * (2.0 * bands.d0() / f64::from(REGION_VOXELS)).powi(2);
    let ball = 4.0 * std::f64::consts::PI * ((f64::from(bound.radius) / size) + 2.0).powi(2);
    (3.0 * band.min(ball)) as usize
}

/// Regions to leave room for in a table of a level that lists `regions`.
fn regions_for(regions: usize, headroom: f64) -> usize {
    (regions as f64 * headroom) as usize + 64
}

impl Scene {
    /// Rings for a Planet inside `bound`: the bricks of `store` that the LOD `bands` select for a
    /// camera at `camera`, a position in Voxels, each level a sparse set of regions of its own.
    /// The store must have its pyramid ([`BrickStore::build_lod`]); the levels are those of it, at
    /// most [`crate::lod::LEVELS`]. Every level is read through [`BrickStore::brick_at`], so the
    /// store's source stands in for the bricks it does not hold, as the core of the Planet does.
    ///
    /// The module's docs say how rings are laid out and walked. [`Scene::recentre`] keeps them
    /// following the camera and [`Scene::update`] takes Edits in. The buffers are the size that
    /// the bricks need, and grow by packing again if the camera goes where they need more.
    pub fn rings(
        store: &BrickStore,
        camera: [f64; 3],
        bands: Bands,
        bound: Bound,
    ) -> Result<Scene, RingsFull> {
        Scene::rings_sized(store, camera, bands, bound, Sizing::Fitted(2.0))
    }

    /// [`Scene::rings`] with buffers of the size `room` says. That is all the room there will
    /// be: a scene that needs more is an error ([`RingsFull`]).
    pub fn rings_in(
        store: &BrickStore,
        camera: [f64; 3],
        bands: Bands,
        bound: Bound,
        room: Room,
    ) -> Result<Scene, RingsFull> {
        Scene::rings_sized(store, camera, bands, bound, Sizing::Given(room))
    }

    fn rings_sized(
        store: &BrickStore,
        camera: [f64; 3],
        bands: Bands,
        bound: Bound,
        sizing: Sizing,
    ) -> Result<Scene, RingsFull> {
        let top = select::top_level(store);
        let roots = select::top_bricks(store, top, Some((bound.centre, bound.radius)));
        let (selection, _) = Selection::new(store, camera, bands, top, roots);
        Scene::pack_rings(store, selection, bound, sizing)
    }

    /// A scene of rings that shows `selection`, with buffers sized as `sizing` says.
    fn pack_rings(
        store: &BrickStore,
        selection: Selection,
        bound: Bound,
        sizing: Sizing,
    ) -> Result<Scene, RingsFull> {
        let by_region = group(selection.bricks());
        let levels = selection.top() + 1;
        let mut wanted = vec![0; levels];
        for &(level, _) in by_region.keys() {
            wanted[level] += 1;
        }
        let (capacity, refs, words) = match sizing {
            Sizing::Given(room) => (
                vec![room.regions_per_ring; levels],
                room.refs_per_ring * levels,
                room.words_per_ring * levels,
            ),
            // The words are not known until the bricks are packed: they are packed into buffers
            // that grow, and the room is made after.
            // A level that has few regions now may have many once the camera has moved (each level
            // is about as full as the others at the surface of a Planet), so none is sized by
            // less than half of what the busiest has, nor by less than what its band can hold
            // (see [`shell_regions`]).
            Sizing::Fitted(headroom) => (
                {
                    let most = wanted.iter().copied().max().unwrap_or(0);
                    let bands = selection.bands();
                    (0..levels)
                        .map(|level| {
                            let n = wanted[level].max(most / 2);
                            regions_for(n.max(shell_regions(bands, bound, level)), headroom)
                        })
                        .collect()
                },
                0,
                0,
            ),
        };
        let fitted = matches!(sizing, Sizing::Fitted(_));
        let mut scene = Scene::empty_rings(selection, bound, sizing, capacity, refs, words);
        // Reading the bricks is what takes the time (they may be generated), so a chunk of regions
        // is read on all the threads, and then packed.
        let regions: Vec<_> = by_region.into_iter().collect();
        for chunk in regions.chunks(PREFETCH_REGIONS) {
            let read = select::par_map(chunk, |((level, _), bricks)| {
                bricks
                    .iter()
                    .map(|&coord| (coord, store.brick_at(*level, coord).into_owned()))
                    .collect::<HashMap<_, _>>()
            });
            for (((level, key), bricks), read) in chunk.iter().zip(&read) {
                scene.repack_region(store, *level, *key, bricks, Some(read))?;
            }
        }
        if let (true, Sizing::Fitted(headroom)) = (fitted, sizing) {
            scene.make_room(headroom);
        }
        scene.sync_header();
        scene.take_patch();
        Ok(scene)
    }

    /// A scene of rings with nothing listed yet.
    fn empty_rings(
        selection: Selection,
        bound: Bound,
        sizing: Sizing,
        capacity: Vec<usize>,
        refs: usize,
        words: usize,
    ) -> Scene {
        let levels = capacity.len();
        let mut base = 0;
        let level_headers: Vec<Level> = capacity
            .iter()
            .map(|&regions| {
                // Half full at most, so searches for a region end quickly.
                let slots = (regions * 2).next_power_of_two().max(2);
                let level = Level {
                    table_base: base as u32,
                    table_mask: slots as u32 - 1,
                    ..Level::default()
                };
                base += slots;
                level
            })
            .collect();
        let records: usize = capacity.iter().sum();
        // Fitted rings pack into buffers that grow, with room taken from the pools as if there
        // were no end to them.
        let unbounded = matches!(sizing, Sizing::Fitted(_));
        let pool = |words: usize| Pool::new(if unbounded { usize::MAX / 4 } else { words });
        Scene {
            levels: level_headers,
            table: vec![0; base * 4],
            regions: vec![0; records * REGION_WORDS],
            brick_refs: vec![0; refs],
            bricks: vec![0; words],
            bound: Some(bound),
            rings: Some(Rings {
                sizing,
                free: (0..records).rev().collect(),
                capacity,
                listed: vec![0; levels],
                refs: pool(refs),
                data: pool(words),
                scratch: Vec::new(),
                patch: Patch::default(),
                selection,
                ball: bound,
                coarsest: None,
            }),
            ..Scene::default()
        }
    }

    /// Makes the buffers of a scene that was packed into growing ones the size they will stay:
    /// what is used, and `headroom` times as much.
    fn make_room(&mut self, headroom: f64) {
        let rings = self.rings.as_mut().expect("a scene of rings");
        let room = |used: usize| (used as f64 * headroom) as usize + 4096;
        let (refs, words) = (room(self.brick_refs.len()), room(self.bricks.len()));
        rings.refs = Pool::new(refs);
        rings.refs.take(self.brick_refs.len());
        rings.data = Pool::new(words);
        rings.data.take(self.bricks.len());
        self.brick_refs.resize(refs, 0);
        self.bricks.resize(words, 0);
    }

    /// The scene packed again from scratch: what `selection` is, with the words in order. The
    /// buffers are sized as they were, or, for a scene that was sized by its bricks, again by
    /// them and `headroom` times as much.
    fn repacked(&self, store: &BrickStore, headroom: Option<f64>) -> Result<Scene, RingsFull> {
        let rings = self.rings.as_ref().expect("a scene of rings");
        let sizing = match (rings.sizing, headroom) {
            (Sizing::Fitted(_), Some(headroom)) => Sizing::Fitted(headroom),
            (sizing, _) => sizing,
        };
        let bound = self.bound.is_some().then_some(rings.ball);
        let mut scene = Scene::pack_rings(store, rings.selection.clone(), rings.ball, sizing)?;
        scene.bound = bound;
        Ok(scene)
    }

    /// A scene of rings packed again from scratch from `store`, with the same bricks selected and
    /// room: what the scene should show, with the words in order.
    pub fn rebuilt_rings(&self, store: &BrickStore) -> Result<Scene, RingsFull> {
        self.repacked(store, None)
    }

    /// Whether this is a scene of rings (see [`Scene::rings`]).
    pub fn is_rings(&self) -> bool {
        self.rings.is_some()
    }

    /// The LOD bands the rings are selected by, if this is a scene of rings.
    pub fn bands(&self) -> Option<Bands> {
        self.rings.as_ref().map(|r| r.selection.bands())
    }

    /// Selects the bricks again with other `bands` (a new bias, or another image height), from
    /// the camera where it is, and packs the scene again. Reported as [`SceneUpdate::Rebuilt`].
    pub fn set_bands(
        &mut self,
        store: &BrickStore,
        bands: Bands,
    ) -> Result<Option<SceneUpdate>, RingsFull> {
        let Some(rings) = &self.rings else {
            return Ok(None);
        };
        if rings.selection.bands() == bands {
            return Ok(None);
        }
        let camera = rings.selection.camera();
        self.rebuilt_at(store, camera, bands).map(Some)
    }

    /// The scene made again for `camera` and `bands`, in place, from nothing: what a jump of the
    /// camera or another bias asks for. Reported as [`SceneUpdate::Rebuilt`].
    fn rebuilt_at(
        &mut self,
        store: &BrickStore,
        camera: [f64; 3],
        bands: Bands,
    ) -> Result<SceneUpdate, RingsFull> {
        let rings = self.rings.as_ref().expect("a scene of rings");
        let (ball, sizing, clipped) = (rings.ball, rings.sizing, self.bound.is_some());
        *self = Scene::rings_sized(store, camera, bands, ball, sizing)?;
        self.bound = clipped.then_some(ball);
        Ok(SceneUpdate::Rebuilt)
    }

    /// Bricks drawn at `level` (see [`Selection::count`]): the part of the scene that the level
    /// is. Zero for a scene that is not rings.
    pub fn level_bricks(&self, level: usize) -> usize {
        self.rings
            .as_ref()
            .map_or(0, |rings| rings.selection.count(level))
    }

    /// Follows the camera to `camera`, a position in Voxels: the bricks that the bands select
    /// change as it moves, and only the regions that lose or gain one are packed again. Returns
    /// what changed, or `None` if nothing did (and if this is not a scene of rings).
    ///
    /// If the regions do not fit, the scene is packed again from scratch, which is reported as
    /// [`SceneUpdate::Rebuilt`]. It is an error only if they do not fit even then, which a scene
    /// that was given its room can be, and then the scene is half moved and not to be used again.
    pub fn recentre(
        &mut self,
        store: &BrickStore,
        camera: [f64; 3],
    ) -> Result<Option<SceneUpdate>, RingsFull> {
        let Some(rings) = &mut self.rings else {
            return Ok(None);
        };
        // A camera that has gone far has most of the bricks to change, which patching does
        // slower than making the scene again: past about two d0 (measured on the test Planet).
        let bands = rings.selection.bands();
        let from = rings.selection.camera();
        let jumped = (0..3)
            .map(|a| (camera[a] - from[a]).powi(2))
            .sum::<f64>()
            .sqrt();
        if jumped > JUMP * bands.d0() {
            return self.rebuilt_at(store, camera, bands).map(Some);
        }
        let diff = rings.selection.update(store, camera);
        let attempt = self
            .apply(
                store,
                group(diff.added.iter().chain(&diff.removed).copied()),
            )
            .map(|patch| {
                let nothing = patch.words() == 0 && !patch.levels;
                (!nothing).then_some(patch)
            });
        match attempt {
            Ok(patch) => Ok(patch.map(SceneUpdate::Rings)),
            Err(RingsFull) => self.pack_again(store).map(Some),
        }
    }

    /// Packs the scene again from scratch, bigger if it was sized by its bricks.
    fn pack_again(&mut self, store: &BrickStore) -> Result<SceneUpdate, RingsFull> {
        // Four times what is needed, not two: a flight from orbit to the ground overflows again and
        // again as the bricks multiply, and each overflow is a new scene and a new upload.
        *self = self.repacked(store, Some(4.0))?;
        Ok(SceneUpdate::Rebuilt)
    }

    /// Packs again the regions of `by_region`, in the bricks it lists, and brings the header up
    /// to date, and hands out the patch.
    fn apply(&mut self, store: &BrickStore, by_region: ByRegion) -> Result<RingPatch, RingsFull> {
        for ((level, key), bricks) in by_region {
            self.repack_region(store, level, key, &bricks, None)?;
        }
        self.sync_header();
        Ok(self.take_patch())
    }

    /// [`Scene::update`] for rings. The selection takes in what the Edit did to what is drawn
    /// (a brick may have become air, or solid with no air beside it, or the reverse), and the
    /// bricks over each changed brick that are drawn are packed again at every level, wherever
    /// they are; the store has remade them already (see [`BrickStore::refresh_lod`]). A brick
    /// whose words are what they were is not written again.
    ///
    /// If the words run out the scene is packed again from scratch ([`SceneUpdate::Rebuilt`]),
    /// which frees what the Edits left; it is an error only if they do not fit even then.
    pub(super) fn update_rings(
        &mut self,
        store: &BrickStore,
        changed: &[BrickCoord],
    ) -> Result<SceneUpdate, RingsFull> {
        let rings = self.rings.as_mut().expect("a scene of rings");
        let diff = rings.selection.refresh(store, changed);
        let mut touched = group(diff.added.iter().chain(&diff.removed).copied());
        // The bricks over the changed ones, and over the ones beside them, which read the
        // Voxels across their border (ADR 0002), that are drawn.
        let mut around: BTreeSet<BrickCoord> = BTreeSet::new();
        for brick in changed {
            around.insert(*brick);
            for axis in 0..3 {
                for step in [-1, 1] {
                    let mut next = brick.0;
                    next[axis] += step;
                    around.insert(BrickCoord(next));
                }
            }
        }
        for level in 0..self.levels.len() {
            for brick in &around {
                let over = BrickCoord(brick.0.map(|c| c >> level));
                if self
                    .rings
                    .as_ref()
                    .is_some_and(|r| r.selection.draws(level, over))
                {
                    touched
                        .entry((level, region_of(over)))
                        .or_default()
                        .insert(over);
                }
            }
        }
        match self.apply(store, touched) {
            Ok(patch) => Ok(SceneUpdate::Rings(patch)),
            Err(RingsFull) => self.pack_again(store),
        }
    }

    /// Brings the regions of `level` of `key` up to date with the bricks that are selected: the
    /// bricks `touch` are written again (if their words differ) or dropped, and the other bricks
    /// of the region are as they were. A region left with no bricks is unlisted, and one that has
    /// some and was not listed is.
    fn repack_region(
        &mut self,
        store: &BrickStore,
        level: usize,
        key: [i32; 3],
        touch: &BTreeSet<BrickCoord>,
        read: Option<&HashMap<BrickCoord, Brick>>,
    ) -> Result<(), RingsFull> {
        let found = self.find_region(level, key);
        // What each touched brick is to be: its entry in `brick_refs`, or none.
        let mut wanted: Vec<(usize, Option<u32>)> = Vec::with_capacity(touch.len());
        let old_entries = found.map(|index| self.region_entries(index));
        let old_at = |bit: usize| old_entries.as_ref().and_then(|e| e[bit]);
        for &coord in touch {
            let bit = bit_of(coord);
            let drawn = self
                .rings
                .as_ref()
                .is_some_and(|r| r.selection.draws(level, coord));
            let entry = if drawn {
                let brick = match read.and_then(|r| r.get(&coord)) {
                    Some(brick) => Cow::Borrowed(brick),
                    None => store.brick_at(level, coord),
                };
                match brick.view() {
                    BrickView::Air => None,
                    view => Some(self.put_brick(view, old_at(bit))?),
                }
            } else {
                None
            };
            wanted.push((bit, entry));
        }
        // The usual case: the same bricks are in the region, and some of them read differently.
        // Then the list of entries stays where it is, and only those words are written.
        if let (Some(index), Some(old)) = (found, &old_entries) {
            let uniform = self.regions[index * REGION_WORDS + REGION_FIRST_REF] & SOLID_FLAG != 0;
            if !uniform
                && wanted
                    .iter()
                    .all(|&(bit, entry)| entry.is_some() == old[bit].is_some())
            {
                let first_ref = self.regions[index * REGION_WORDS + REGION_FIRST_REF] as usize;
                let rank = |bit: usize| old[..bit].iter().filter(|e| e.is_some()).count();
                for &(bit, entry) in &wanted {
                    let (Some(entry), Some(before)) = (entry, old[bit]) else {
                        continue;
                    };
                    if entry != before {
                        self.release_brick(before);
                        let at = first_ref + rank(bit);
                        self.brick_refs[at] = entry;
                        self.patch().brick_refs.push(at..at + 1);
                    }
                }
                return Ok(());
            }
        }
        let old: Vec<Option<u32>> = old_entries.unwrap_or_else(|| vec![None; 512]);
        let mut new = old.clone();
        for &(bit, entry) in &wanted {
            new[bit] = entry;
        }
        if new == old {
            return Ok(());
        }
        // What the region no longer refers to is free, and so is its list of entries.
        for (&before, &after) in old.iter().zip(&new) {
            if let Some(entry) = before.filter(|&e| Some(e) != after) {
                self.release_brick(entry);
            }
        }
        if let Some(index) = found {
            self.release_refs(index);
        }
        let bricks: Vec<(usize, u32)> = (0..512).filter_map(|bit| Some((bit, new[bit]?))).collect();
        if bricks.is_empty() {
            if let Some(index) = found {
                self.unlist_region(level, key, index);
            }
            return Ok(());
        }
        let index = match found {
            Some(index) => index,
            None => {
                let rings = self.rings.as_mut().expect("a scene of rings");
                if rings.listed[level] >= rings.capacity[level] {
                    return Err(RingsFull);
                }
                rings.listed[level] += 1;
                rings.free.pop().ok_or(RingsFull)?
            }
        };
        let first = bricks[0].1;
        let at = index * REGION_WORDS;
        if bricks.len() == 512 && first & SOLID_FLAG != 0 && bricks.iter().all(|b| b.1 == first) {
            self.regions[at..at + REGION_FIRST_REF].fill(u32::MAX);
            self.regions[at + REGION_FIRST_REF] = first;
        } else {
            self.write_entries(index, &bricks)?;
        }
        let new_slot = found
            .is_none()
            .then(|| self.insert_region(level, key, index as u32));
        let patch = self.patch();
        patch.regions.insert(index);
        patch.table.extend(new_slot);
        Ok(())
    }

    /// The `brick_refs` entry of each of the 512 bricks of the region record at `index`, in one
    /// pass over its mask.
    fn region_entries(&self, index: usize) -> Vec<Option<u32>> {
        let record = &self.regions[index * REGION_WORDS..][..REGION_WORDS];
        let first_ref = record[REGION_FIRST_REF];
        if first_ref & SOLID_FLAG != 0 {
            return vec![Some(first_ref); 512];
        }
        let mut rank = 0;
        (0..512)
            .map(|bit| {
                (record[bit / 32] >> (bit % 32) & 1 != 0).then(|| {
                    rank += 1;
                    self.brick_refs[(first_ref + rank - 1) as usize]
                })
            })
            .collect()
    }

    /// Takes the region at `key` of mip `level` out of the table, and its record `index` back.
    fn unlist_region(&mut self, level: usize, key: [i32; 3], index: usize) {
        let mut slots = BTreeSet::new();
        self.remove_region(level, key, &mut slots);
        let rings = self.rings.as_mut().expect("a scene of rings");
        rings.listed[level] -= 1;
        rings.patch.table.extend(slots);
        rings.patch.regions.remove(&index);
        rings.free.push(index);
    }

    /// Writes the mask and the list of entries of the region record at `index`, for `bricks`
    /// (mask bit and entry, by mask bit). The list takes words from `brick_refs`.
    fn write_entries(&mut self, index: usize, bricks: &[(usize, u32)]) -> Result<(), RingsFull> {
        let rings = self.rings.as_mut().expect("a scene of rings");
        let first_ref = rings.refs.take(bricks.len()).ok_or(RingsFull)?;
        rings
            .patch
            .brick_refs
            .push(first_ref..first_ref + bricks.len());
        if self.brick_refs.len() < first_ref + bricks.len() {
            self.brick_refs.resize(first_ref + bricks.len(), 0);
        }
        let mut mask = [0u32; 16];
        for (i, &(bit, entry)) in bricks.iter().enumerate() {
            mask[bit / 32] |= 1 << (bit % 32);
            self.brick_refs[first_ref + i] = entry;
        }
        let at = index * REGION_WORDS;
        self.regions[at..at + REGION_FIRST_REF].copy_from_slice(&mask);
        self.regions[at + REGION_FIRST_REF] = first_ref as u32;
        Ok(())
    }

    /// Writes `view`, which is not air, and returns its entry for `brick_refs`. A mixed brick
    /// takes words from `bricks`, unless `old` is the entry of a mixed brick with the same words,
    /// which is kept.
    fn put_brick(&mut self, view: BrickView<'_>, old: Option<u32>) -> Result<u32, RingsFull> {
        let BrickView::Mixed {
            occupancy,
            palette,
            bits,
            indices,
        } = view
        else {
            return match view {
                BrickView::Solid(material) => Ok(SOLID_FLAG | u32::from(material.0)),
                _ => unreachable!("air bricks are skipped"),
            };
        };
        let old = old.filter(|&entry| entry & SOLID_FLAG == 0);
        let old_len = old.map(|entry| self.mixed_words(entry));
        let rings = self.rings.as_mut().expect("a scene of rings");
        rings.scratch.clear();
        encode_mixed(&mut rings.scratch, occupancy, palette, bits, indices);
        let len = rings.scratch.len();
        if let (Some(old), Some(old_len)) = (old, old_len)
            && old_len == len
            && self.bricks[old as usize..][..len] == rings.scratch[..]
        {
            return Ok(old);
        }
        let at = rings.data.take(len).ok_or(RingsFull)?;
        rings.patch.bricks.push(at..at + len);
        if self.bricks.len() < at + len {
            self.bricks.resize(at + len, 0);
        }
        self.bricks[at..at + len].copy_from_slice(&rings.scratch);
        Ok(at as u32)
    }

    /// Gives back the words of `bricks` that the brick with `entry` has.
    fn release_brick(&mut self, entry: u32) {
        let words = self.mixed_words(entry);
        if words > 0 {
            let rings = self.rings.as_mut().expect("a scene of rings");
            rings.data.give(entry as usize, words);
        }
    }

    /// Gives back the words of `brick_refs` that the region record at `index` lists its bricks
    /// in.
    fn release_refs(&mut self, index: usize) {
        let refs = self.region_refs(index);
        if refs > 0 {
            let first_ref = self.regions[index * REGION_WORDS + REGION_FIRST_REF] as usize;
            let rings = self.rings.as_mut().expect("a scene of rings");
            rings.refs.give(first_ref, refs);
        }
    }

    /// The regions of the box of mip `level` that no brick of it reaches, as the first region
    /// and the one after the last, in regions of `level`: a cube about the camera that is inside
    /// the ball that no brick of the level is in. None for level 0, and for a scene that is not
    /// rings. The walk skips them.
    pub fn hole(&self, level: usize) -> ([i32; 3], [i32; 3]) {
        self.levels
            .get(level)
            .map_or(([0; 3], [0; 3]), |l| (l.hole_first, l.hole_end))
    }

    /// The level of the coarsest brick drawn, if the scene is rings with any.
    pub(super) fn coarsest_drawn(&self) -> Option<usize> {
        self.rings.as_ref().and_then(|r| r.selection.coarsest())
    }

    /// Brings the parts of the header that follow the camera and the bricks up to date: the
    /// holes, and the ball that rays are clipped to (which depends on the coarsest level with a
    /// brick). Marks the patch if either changed.
    fn sync_header(&mut self) {
        let rings = self.rings.as_mut().expect("a scene of rings");
        let camera = rings.selection.camera();
        let bands = rings.selection.bands();
        // The ball that no brick of a level is in: as far as its band says, or as far as the
        // nearest brick that is drawn whole though it is nearer than that.
        let stuck = rings.selection.nearest_stuck();
        let radii: Vec<f64> = (0..self.levels.len())
            .map(|level| stuck[level].unwrap_or(bands.clear_radius(level)))
            .collect();
        let mut changed = false;
        for (level, radius) in radii.into_iter().enumerate() {
            // The cube inside that ball, in whole regions of the level.
            let half = radius / 3f64.sqrt();
            let size = f64::from(REGION_VOXELS << level);
            let first: [i32; 3] =
                std::array::from_fn(|a| ((camera[a] - half) / size).ceil() as i32);
            let end: [i32; 3] = std::array::from_fn(|a| ((camera[a] + half) / size).floor() as i32);
            let (first, end) = if (0..3).all(|a| end[a] > first[a]) {
                (first, end)
            } else {
                ([0; 3], [0; 3])
            };
            let l = &mut self.levels[level];
            changed |= (l.hole_first, l.hole_end) != (first, end);
            (l.hole_first, l.hole_end) = (first, end);
        }
        let coarsest = self.coarsest_drawn();
        let rings = self.rings.as_mut().expect("a scene of rings");
        changed |= rings.coarsest != coarsest;
        rings.coarsest = coarsest;
        if changed {
            rings.patch.levels = true;
        }
    }

    /// Bytes of the scene's buffers that hold something. The buffers of rings are allocated
    /// whole, so for them this is how full they are, and the size of the GPU copy is that of the
    /// buffers.
    pub fn used_bytes(&self) -> u64 {
        let words = match &self.rings {
            Some(rings) => {
                let free = |pool: &Pool| {
                    pool.free
                        .iter()
                        .map(|r| r.end.min(usize::MAX / 8).saturating_sub(r.start))
                        .sum::<usize>()
                };
                let (refs, data) = (
                    free(&rings.refs).min(self.brick_refs.len()),
                    free(&rings.data).min(self.bricks.len()),
                );
                self.table.len() + self.regions.len() + self.brick_refs.len() + self.bricks.len()
                    - rings.free.len() * REGION_WORDS
                    - refs
                    - data
            }
            None => {
                self.table.len() + self.regions.len() + self.brick_refs.len() + self.bricks.len()
                    - self.garbage
            }
        };
        words as u64 * 4
    }

    /// What has been written since the last patch.
    fn patch(&mut self) -> &mut Patch {
        &mut self.rings.as_mut().expect("a scene of rings").patch
    }

    /// The level whose part of `table` has `slot`.
    fn level_of_slot(&self, slot: usize) -> usize {
        self.levels
            .partition_point(|l| l.table_base as usize <= slot)
            .saturating_sub(1)
    }

    /// Fits the occupied box of each of `levels` to the regions it lists now, and says whether
    /// any changed.
    fn occupy(&mut self, levels: &BTreeSet<usize>) -> bool {
        let mut changed = false;
        for &level in levels {
            let Level {
                table_base,
                table_mask,
                ..
            } = self.levels[level];
            let (mut first, mut end) = ([i32::MAX; 3], [i32::MIN; 3]);
            for slot in table_base..=table_base + table_mask {
                let entry = &self.table[slot as usize * 4..][..4];
                if entry[3] != 0 {
                    for a in 0..3 {
                        first[a] = first[a].min(entry[a] as i32);
                        end[a] = end[a].max(entry[a] as i32 + 1);
                    }
                }
            }
            let (first, end) = if first[0] == i32::MAX {
                ([0; 3], [0; 3])
            } else {
                (first, end)
            };
            let l = &mut self.levels[level];
            changed |= (l.occupied_first, l.occupied_end) != (first, end);
            (l.occupied_first, l.occupied_end) = (first, end);
            l.origin = first.map(|f| f << (6 + level));
            l.dims = std::array::from_fn(|a| (end[a] - first[a]) as u32);
        }
        changed
    }

    /// Hands out what has been written since the last call.
    fn take_patch(&mut self) -> RingPatch {
        let mut patch = std::mem::take(self.patch());
        if !patch.table.is_empty() {
            let touched: BTreeSet<usize> = patch
                .table
                .iter()
                .map(|&slot| self.level_of_slot(slot))
                .collect();
            patch.levels |= self.occupy(&touched);
        }
        RingPatch {
            regions: patch.regions.into_iter().collect(),
            table: patch.table.into_iter().collect(),
            brick_refs: merged(patch.brick_refs),
            bricks: merged(patch.bricks),
            levels: patch.levels,
        }
    }
}

const _: () = assert!(MAX_LEVELS >= crate::lod::LEVELS);
const _: () = assert!(REGION_BRICKS == 8);

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use brick_store::{Material, VoxelCoord};
    use planet_gen::generate_region;

    use super::*;

    const LEVELS: usize = 5;
    /// Above the test Planet's +y pole.
    const ABOVE: [f64; 3] = [0.0, 1560.0, 0.0];

    /// Bands on a scale that the ground patch has several levels in: level 1 from 128 Voxels.
    fn bands() -> Bands {
        Bands::with_d0(64.0)
    }

    fn bound() -> Bound {
        Bound {
            centre: [0; 3],
            radius: planet_gen::MAX_RADIUS,
        }
    }

    /// The ground near the +y pole, Voxels x -440..=447, y 1360..=1575, z -360..=263, with its
    /// pyramid.
    fn ground() -> &'static BrickStore {
        static GROUND: OnceLock<BrickStore> = OnceLock::new();
        GROUND.get_or_init(|| {
            let mut store = generate_region(
                0xB4A7_A9AC,
                BrickCoord([-55, 170, -45]),
                BrickCoord([55, 196, 32]),
            );
            store.build_lod(LEVELS - 1);
            store
        })
    }

    fn rings_at(camera: [f64; 3]) -> Scene {
        Scene::rings(ground(), camera, bands(), bound()).unwrap()
    }

    /// What a scene shows of every region it lists, by level and region: its Material if it is
    /// all one, else the words of each brick by mask bit. It does not depend on where in the
    /// buffers anything is.
    #[allow(clippy::type_complexity)]
    fn shown(scene: &Scene) -> BTreeMap<(usize, [i32; 3]), (Option<u32>, Vec<(usize, Vec<u32>)>)> {
        let mut all = BTreeMap::new();
        for (level, l) in scene.levels.iter().enumerate() {
            for slot in l.table_base..=l.table_base + l.table_mask {
                let entry = &scene.table[slot as usize * 4..][..4];
                if entry[3] == 0 {
                    continue;
                }
                let key = [entry[0], entry[1], entry[2]].map(|k| k as i32);
                let index = entry[3] as usize - 1;
                let first_ref = scene.regions[index * REGION_WORDS + REGION_FIRST_REF];
                let region = if first_ref & SOLID_FLAG != 0 {
                    (Some(first_ref & 0xffff), Vec::new())
                } else {
                    let bricks = (0..512)
                        .filter_map(|bit| {
                            let entry = scene.entry(index, bit)?;
                            let words = match scene.mixed_words(entry) {
                                0 => vec![entry],
                                n => scene.bricks[entry as usize..][..n].to_vec(),
                            };
                            Some((bit, words))
                        })
                        .collect();
                    (None, bricks)
                };
                all.insert((level, key), region);
            }
        }
        all
    }

    /// The buffers of a scene, as a GPU copy of it holds them.
    #[derive(Clone, PartialEq, Debug)]
    struct Copy {
        table: Vec<u32>,
        regions: Vec<u32>,
        brick_refs: Vec<u32>,
        bricks: Vec<u32>,
    }

    impl Copy {
        fn of(scene: &Scene) -> Self {
            Self {
                table: scene.table.clone(),
                regions: scene.regions.clone(),
                brick_refs: scene.brick_refs.clone(),
                bricks: scene.bricks.clone(),
            }
        }

        /// What a GPU copy holds after `patch` is written to it from `scene`, as the tracer does.
        fn patched(mut self, patch: &RingPatch, scene: &Scene) -> Self {
            for &slot in &patch.table {
                self.table[slot * 4..][..4].copy_from_slice(&scene.table[slot * 4..][..4]);
            }
            for &index in &patch.regions {
                let at = index * REGION_WORDS;
                self.regions[at..at + REGION_WORDS]
                    .copy_from_slice(&scene.regions[at..][..REGION_WORDS]);
            }
            for range in &patch.brick_refs {
                self.brick_refs[range.clone()].copy_from_slice(&scene.brick_refs[range.clone()]);
            }
            for range in &patch.bricks {
                self.bricks[range.clone()].copy_from_slice(&scene.bricks[range.clone()]);
            }
            self
        }
    }

    fn patch_of(update: Option<SceneUpdate>) -> RingPatch {
        match update {
            Some(SceneUpdate::Rings(patch)) => patch,
            other => panic!("the rings were patched: {other:?}"),
        }
    }

    #[test]
    fn each_level_lists_the_bricks_the_bands_select_for_it_and_no_other() {
        let scene = rings_at(ABOVE);
        let selection = &scene.rings.as_ref().unwrap().selection;
        let mut listed = 0;
        for ((level, key), (uniform, bricks)) in shown(&scene) {
            assert!(level < scene.levels.len());
            // A uniform region is 512 bricks, and every one of them is drawn.
            let n = if uniform.is_some() { 512 } else { bricks.len() };
            listed += n;
            let first = key.map(|k| k * REGION_BRICKS);
            let drawn = (0..512)
                .filter(|&bit| {
                    let coord =
                        BrickCoord(std::array::from_fn(|a| first[a] + (bit >> (3 * a) & 7)));
                    selection.draws(level, coord)
                })
                .count();
            assert_eq!(drawn, n, "level {level} region {key:?}");
        }
        let drawn: usize = (0..LEVELS).map(|l| scene.level_bricks(l)).sum();
        assert_eq!(listed, drawn);
        assert!(scene.level_bricks(0) > 100 && scene.level_bricks(1) > 100);
        assert!(scene.levels[0].occupied_end[1] > scene.levels[0].occupied_first[1]);
    }

    #[test]
    fn the_hole_of_a_level_lists_no_region() {
        for camera in [ABOVE, [200.0, 1700.0, -100.0], [0.0, 3000.0, 0.0]] {
            let scene = rings_at(camera);
            assert_eq!(scene.hole(0), ([0; 3], [0; 3]));
            for level in 0..scene.levels.len() {
                let (first, end) = scene.hole(level);
                let size = f64::from(REGION_VOXELS << level);
                let radius = bands().clear_radius(level);
                for a in 0..3 {
                    // The hole is within the ball about the camera that no brick is in.
                    if end[a] > first[a] {
                        let half = f64::from(end[a] - first[a]) * size / 2.0;
                        assert!(half * 3f64.sqrt() <= radius + 2.0 * size, "{level}");
                    }
                }
                for ((lvl, key), _) in shown(&scene) {
                    let inside = (0..3).all(|a| key[a] >= first[a] && key[a] < end[a]);
                    assert!(!(lvl == level && inside), "level {level} lists {key:?}");
                }
            }
        }
    }

    #[test]
    fn rings_that_follow_the_camera_show_what_rings_built_where_it_ends_do() {
        let mut scene = rings_at([-420.0, 1560.0, -300.0]);
        let mut copy = Copy::of(&scene);
        let mut moves = 0;
        let mut at = [-420.0, 1560.0, -300.0];
        for step in 0..120 {
            at = [
                at[0] + 7.5 + f64::from(step % 5),
                1560.0 + 300.0 * (f64::from(step) / 15.0).sin(),
                at[2] + 4.0,
            ];
            match scene.recentre(ground(), at).unwrap() {
                None => {}
                Some(SceneUpdate::Rings(patch)) => {
                    moves += 1;
                    copy = copy.patched(&patch, &scene);
                }
                Some(SceneUpdate::Rebuilt) => copy = Copy::of(&scene),
                Some(other) => panic!("step {step}: {other:?}"),
            }
            assert_eq!(copy, Copy::of(&scene), "step {step}");
        }
        assert!(moves > 20, "{moves}");
        // What it shows is a selection that has followed the camera, which a new one differs from
        // only within the hysteresis: so compare with the same selection packed from scratch.
        assert_eq!(
            shown(&scene),
            shown(&scene.rebuilt_rings(ground()).unwrap())
        );
    }

    #[test]
    fn a_camera_that_does_not_move_changes_nothing_and_a_small_move_writes_little() {
        let mut scene = rings_at(ABOVE);
        assert_eq!(scene.recentre(ground(), ABOVE), Ok(None));
        let total = scene.used_bytes() as usize / 4;
        let mut written = 0;
        for i in 1..=5 {
            let at = [f64::from(i) * 2.0, 1560.0, 0.0];
            if let Some(update) = scene.recentre(ground(), at).unwrap() {
                written += patch_of(Some(update)).words();
            }
        }
        // 10 Voxels of walking: a small part of the scene is written.
        assert!(written * 4 < total, "{written} words of {total}");
    }

    /// Digs a ball of Voxels out of `store` and returns the bricks of level 0 it changed.
    fn dig(store: &mut BrickStore, centre: [i32; 3], radius: i32) -> Vec<BrickCoord> {
        let mut changed = BTreeSet::new();
        for x in -radius..=radius {
            for y in -radius..=radius {
                for z in -radius..=radius {
                    let voxel = VoxelCoord([centre[0] + x, centre[1] + y, centre[2] + z]);
                    if x * x + y * y + z * z <= radius * radius && store.get(voxel).is_some() {
                        changed.insert(voxel.brick());
                        store.set(voxel, None);
                    }
                }
            }
        }
        changed.into_iter().collect()
    }

    #[test]
    fn an_edit_shows_in_every_level_as_a_new_scene_would_and_writes_only_what_changed() {
        let mut store = ground().clone();
        let mut scene = Scene::rings(&store, ABOVE, bands(), bound()).unwrap();
        let mut copy = Copy::of(&scene);
        let changed = dig(&mut store, [20, 1510, -30], 14);
        assert!(!changed.is_empty());
        let refresh = store.refresh_lod(changed.iter().copied());
        assert!(refresh.changed.iter().any(|&(level, _)| level >= 2));

        let SceneUpdate::Rings(patch) = scene.update(&store, &changed).unwrap() else {
            panic!("an edit patches the rings");
        };
        assert_eq!(
            shown(&scene),
            shown(&Scene::rings(&store, ABOVE, bands(), bound()).unwrap())
        );
        copy = copy.patched(&patch, &scene);
        assert_eq!(copy, Copy::of(&scene));
        // A crater 28 Voxels across touches a handful of bricks in each level.
        assert!(patch.words() < 30_000, "{} words", patch.words());

        // The same Edit again finds every brick as it is, so writes none of them.
        let SceneUpdate::Rings(again) = scene.update(&store, &changed).unwrap() else {
            panic!("an edit patches the rings");
        };
        assert!(again.bricks.is_empty(), "{again:?}");
        assert_eq!(copy, Copy::of(&scene));
    }

    /// Words of `bricks` that the scene's bricks take.
    fn words_taken(scene: &Scene) -> usize {
        let free: usize = scene
            .rings
            .as_ref()
            .unwrap()
            .data
            .free
            .iter()
            .map(Range::len)
            .sum();
        scene.bricks.len() - free
    }

    #[test]
    fn a_scene_that_was_given_too_little_room_is_an_error_and_one_that_was_not_grows() {
        let scene = rings_at(ABOVE);
        let tiny = Room {
            words_per_ring: words_taken(&scene) / LEVELS / 2,
            ..Room::default()
        };
        assert!(Scene::rings_in(ground(), ABOVE, bands(), bound(), tiny).is_err());
        // A fitted scene that has to hold more than it was sized for packs itself again.
        let mut store = ground().clone();
        let mut small = Scene::rings(&store, ABOVE, bands(), bound()).unwrap();
        let spare: usize = small
            .rings
            .as_ref()
            .unwrap()
            .data
            .free
            .iter()
            .map(Range::len)
            .sum();
        // Take the spare words, so that the next brick that needs some does not fit.
        assert!(small.rings.as_mut().unwrap().data.take(spare).is_some());
        let changed = dig(&mut store, [20, 1510, -30], 14);
        store.refresh_lod(changed.iter().copied());
        let update = small.update(&store, &changed).unwrap();
        assert_eq!(update, SceneUpdate::Rebuilt);
        assert_eq!(
            shown(&small),
            shown(&Scene::rings(&store, ABOVE, bands(), bound()).unwrap())
        );
    }

    #[test]
    fn another_bias_selects_other_bricks() {
        let mut scene = rings_at(ABOVE);
        assert_eq!(scene.set_bands(ground(), bands()), Ok(None));
        let before: usize = (0..LEVELS).map(|l| scene.level_bricks(l)).sum();
        let loose = Bands::with_d0(32.0);
        assert_eq!(
            scene.set_bands(ground(), loose),
            Ok(Some(SceneUpdate::Rebuilt))
        );
        assert_eq!(scene.bands(), Some(loose));
        let after: usize = (0..LEVELS).map(|l| scene.level_bricks(l)).sum();
        // Half the distances for every level: fewer, coarser bricks.
        assert!(after < before, "{after} against {before}");
    }

    #[test]
    fn the_clip_radius_allows_for_the_coarsest_voxel_drawn() {
        let scene = rings_at(ABOVE);
        let coarsest = scene.coarsest_drawn().unwrap();
        let radius = scene.clip_radius().unwrap();
        let want = planet_gen::MAX_RADIUS as f32
            + (1u32 << coarsest) as f32 * 3f32.sqrt()
            + crate::scene::BOUND_SLACK;
        assert_eq!(radius, want);
    }

    #[test]
    fn removing_regions_leaves_the_others_where_a_search_finds_them() {
        // Enough regions in one table that many of them collided.
        let mut store = BrickStore::new();
        let keys: Vec<[i32; 3]> = (0..1500).map(|i| [i % 37, i / 37 % 5, i / 185]).collect();
        for key in &keys {
            store.insert(
                BrickCoord(key.map(|k| k * 8)),
                brick_store::Brick::solid(Material(1)),
            );
        }
        let mut scene = Scene::new(&store, None);
        let mut touched = BTreeSet::new();
        for key in keys.iter().step_by(3) {
            assert!(scene.remove_region(0, *key, &mut touched).is_some());
        }
        assert!(!touched.is_empty());
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(scene.find_region(0, *key).is_some(), i % 3 != 0, "{key:?}");
        }
        assert_eq!(scene.remove_region(0, keys[0], &mut touched), None);
    }

    #[test]
    fn a_pool_hands_out_words_in_a_row_and_joins_what_comes_back() {
        let mut pool = Pool::new(100);
        let (a, b, c) = (pool.take(30), pool.take(30), pool.take(30));
        assert_eq!((a, b, c), (Some(0), Some(30), Some(60)));
        assert_eq!(pool.take(20), None, "only ten words are left");
        pool.give(30, 30);
        assert_eq!(pool.take(31), None, "the hole is thirty long");
        pool.give(0, 30);
        pool.give(60, 30);
        assert_eq!(pool.free, std::iter::once(0..100).collect::<Vec<_>>());
        assert_eq!(pool.take(100), Some(0));
    }

    #[test]
    fn ranges_that_touch_or_overlap_are_written_as_one() {
        assert_eq!(
            merged(vec![10..20, 0..5, 5..8, 18..25, 30..30]),
            [0..8, 10..25]
        );
    }

    /// A brick that is whole because there is nothing under it to split it into is found by a ray
    /// even when it is nearer than its level is used from, where its level's hole would be.
    #[test]
    fn a_brick_that_cannot_be_split_is_not_in_the_hole_of_its_level() {
        #[derive(Debug)]
        struct Odd;
        impl brick_store::BrickSource for Odd {
            fn brick(&self, level: usize, coord: BrickCoord) -> brick_store::Brick {
                if level == 1 && coord == BrickCoord([0, 0, 0]) {
                    brick_store::Brick::solid(Material(1))
                } else {
                    brick_store::Brick::AIR
                }
            }
        }
        let mut store = BrickStore::new();
        store.set_source(Odd);
        store.build_lod(1);
        let bound = Bound {
            centre: [0; 3],
            radius: 100,
        };
        // 40 Voxels above the brick (which is 16 across), and level 1 is used from 2000 away.
        let camera = [8.5, 40.5, 8.5];
        let scene = Scene::rings(&store, camera, Bands::with_d0(1000.0), bound).unwrap();
        assert_eq!(scene.level_bricks(1), 1);
        let (first, end) = scene.hole(1);
        assert!(
            (0..3).any(|a| end[a] <= first[a]) || first[1] > 0 || end[1] <= 0,
            "{first:?} {end:?}"
        );
        let (cell, origin) = crate::camera::split(camera);
        let ray = crate::reference::Ray {
            cell,
            origin,
            dir: [0.0, -1.0, 0.0],
        };
        let hit = crate::reference::trace(&scene, &ray).expect("the brick is hit");
        assert_eq!(hit.voxel[1], 15);
    }

    #[test]
    fn a_camera_that_jumps_gets_a_new_scene_and_one_that_walks_gets_a_patch() {
        let mut scene = rings_at(ABOVE);
        // d0 is 64 Voxels here: a jump is more than 96.
        let near = [60.0, 1560.0, 0.0];
        assert!(matches!(
            scene.recentre(ground(), near),
            Ok(None | Some(SceneUpdate::Rings(_)))
        ));
        let far = [400.0, 1600.0, 0.0];
        assert_eq!(
            scene.recentre(ground(), far),
            Ok(Some(SceneUpdate::Rebuilt))
        );
        assert_eq!(
            shown(&scene),
            shown(&Scene::rings(ground(), far, bands(), bound()).unwrap())
        );
        // The ball that rays are clipped to is as it was.
        assert_eq!(scene.bound, Some(bound()));
        // And it carries on from there.
        assert!(matches!(
            scene.recentre(ground(), [410.0, 1600.0, 0.0]),
            Ok(None | Some(SceneUpdate::Rings(_)))
        ));
    }
}
