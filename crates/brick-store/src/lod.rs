//! Downsampling: how a coarse brick is made from the eight finer bricks it covers (ADR 0002),
//! by majority, with the Voxels exposed to air counting first. Integer-only, so every machine
//! builds the same coarse brick. Also what a brick reads as where none is held.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::fmt::Debug;

use crate::brick::{BRICK_VOXELS, Brick, BrickView, Material};
use crate::coord::BrickCoord;

/// What a brick reads as where the store holds none, at any mip level. Without one, that is
/// air. The store asks it for the level 0 bricks that it does not hold, and for the children
/// that a coarse brick is made from when they are not held either.
///
/// A coarse brick reflects the real Planet where the source is exact for the bricks it is made
/// from, which means one of:
/// - the source returns, at every level, what downsampling the real Planet gives there; or
/// - every coarse brick that the source gets wrong has a level 0 brick under it that is held,
///   so that the store makes it from held bricks and never reads the source for it.
///
/// Whether or not it is true, the pyramid is a function of the bricks held at level 0 and of
/// the source alone, so refreshing it after an Edit gives what building it afresh does. That
/// holds with the source read for the bricks beside the ones a coarse brick is made from, too:
/// whether a Voxel is exposed to air depends on them.
pub trait BrickSource: Debug + Send + Sync {
    /// The brick at `coord` of mip `level`, where none is held.
    fn brick(&self, level: usize, coord: BrickCoord) -> Brick;

    /// Whether every Voxel of level 0 under the brick at `coord` of `level` is solid, which the
    /// source knows without reading them: an answer of `true` must be exact, and `false` means
    /// "not known". A coarse brick that reads as full may still have air under it (a coarse Voxel
    /// is solid when half its children are), so a renderer that leaves out what no ray can reach
    /// asks this before it leaves a brick out for good.
    fn solid_below(&self, _level: usize, _coord: BrickCoord) -> bool {
        false
    }
}

/// A ball at the Planet's centre made of one Material, so that none of its bricks need be held.
/// A brick wholly inside `radius` Voxels of the centre reads as `material`, and any other as
/// air, at every level.
///
/// That is exact for bricks wholly inside the ball, and for bricks wholly outside it if the
/// Planet has nothing there that is not held. It is not exact for a brick over the ball's edge,
/// which is part rock and part not. So the Planet must hold every level 0 brick that is not
/// wholly inside the ball and not air: its crust. `planet_gen::generate_planet` does. Then each
/// coarse brick over the ball's edge has held bricks under it, and is made from them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interior {
    pub radius: i32,
    pub material: Material,
}

impl BrickSource for Interior {
    fn brick(&self, level: usize, coord: BrickCoord) -> Brick {
        if coord.within_ball_at(level, self.radius) {
            Brick::solid(self.material)
        } else {
            Brick::AIR
        }
    }

    fn solid_below(&self, level: usize, coord: BrickCoord) -> bool {
        coord.within_ball_at(level, self.radius)
    }
}

/// The coarse brick over eight finer ones, given in [`crate::BrickCoord::children`] order.
/// `outside` reads the bricks around them: the brick at an offset from the first child, with
/// one coordinate in `-1..=2` outside the eight.
///
/// Each coarse Voxel covers 2×2×2 Voxels of the finer bricks. It is solid when at least 4 of
/// those 8 are solid. Then it takes the most common Material among the solid ones that are
/// exposed, if any are, and the most common among all the solid ones if none are. A Voxel is
/// exposed when air is next to one of its six faces, in the same block or not: the air that
/// makes a block of solid Voxels the surface can be in the next block, even in the next brick.
/// If several Materials are equally common, the lowest Material ID wins.
pub(crate) fn downsample<'a>(
    children: [&'a Brick; 8],
    outside: impl Fn([i32; 3]) -> Cow<'a, Brick>,
) -> Brick {
    // Whole bricks of air, or of one Material, are by far the commonest case up the pyramid.
    if children.iter().all(|c| c.is_air()) {
        return Brick::AIR;
    }
    if let BrickView::Solid(material) = children[0].view()
        && children[1..]
            .iter()
            .all(|c| matches!(c.view(), BrickView::Solid(m) if m == material))
    {
        return Brick::solid(material);
    }

    // Made when the first Voxel needs the air beside it.
    let mut around = None;
    let mut coarse = Brick::AIR;
    for i in 0..BRICK_VOXELS {
        let at = [i & 7, i >> 3 & 7, i >> 6];
        // The 8 Voxels are in one child, `& 7` inside it.
        let child = children[at[0] >> 2 | (at[1] >> 2) << 1 | (at[2] >> 2) << 2];
        match child.view() {
            BrickView::Air => continue,
            BrickView::Solid(material) => {
                coarse.set(at.map(|a| a as u8), Some(material));
                continue;
            }
            BrickView::Mixed { .. } => {}
        }
        let voxels: [Option<Material>; 8] = std::array::from_fn(|j| {
            child.get([0, 1, 2].map(|a| ((2 * at[a] + (j >> a & 1)) & 7) as u8))
        });
        let mut all = Tally::new();
        for material in voxels.into_iter().flatten() {
            all.add(material);
        }
        if all.count < 4 {
            continue;
        }
        // Only a block of more than one Material has anything to choose, and so anything to
        // look at the air for.
        let mut winner = all.winner();
        if all.len > 1 {
            let mut exposed = Tally::new();
            for (j, voxel) in voxels.into_iter().enumerate() {
                // The Voxel's position in the 16×16×16 Voxels of the eight bricks.
                let found = [0, 1, 2].map(|a| (2 * at[a] + (j >> a & 1)) as i32);
                if let Some(material) = voxel
                    && around
                        .get_or_insert_with(|| Around::new(children, &outside))
                        .exposed(found)
                {
                    exposed.add(material);
                }
            }
            if exposed.count > 0 {
                winner = exposed.winner();
            }
        }
        coarse.set(at.map(|a| a as u8), Some(winner));
    }
    coarse
}

/// The 16×16×16 Voxels of eight bricks, and the Voxels next to them in the bricks around, which
/// are read only when they are needed.
struct Around<'a, F> {
    children: [&'a Brick; 8],
    outside: F,
    /// The bricks read, by offset from the first child: -1 to 2 along an axis.
    read: [Option<Cow<'a, Brick>>; 64],
}

impl<'a, F: Fn([i32; 3]) -> Cow<'a, Brick>> Around<'a, F> {
    fn new(children: [&'a Brick; 8], outside: F) -> Self {
        Self {
            children,
            outside,
            read: std::array::from_fn(|_| None),
        }
    }

    /// The Voxel at `at`, counted in Voxels from the first child's minimum corner, which may be
    /// one past the 16 of the eight bricks along one axis, or one before them.
    #[inline]
    fn voxel(&mut self, at: [i32; 3]) -> Option<Material> {
        // Negative coordinates are huge as unsigned ones.
        if at.iter().all(|&v| (v as u32) < 16) {
            let child = (at[0] >> 3 | (at[1] >> 3) << 1 | (at[2] >> 3) << 2) as usize;
            return self.children[child].get(at.map(|v| (v & 7) as u8));
        }
        self.beside(at)
    }

    /// The Voxel at `at`, which is in a brick that is not one of the eight.
    #[cold]
    fn beside(&mut self, at: [i32; 3]) -> Option<Material> {
        let brick = at.map(|v| v.div_euclid(8));
        let local = at.map(|v| v.rem_euclid(8) as u8);
        let slot = (brick[0] + 1 + 4 * (brick[1] + 1) + 16 * (brick[2] + 1)) as usize;
        let outside = &self.outside;
        self.read[slot]
            .get_or_insert_with(|| outside(brick))
            .get(local)
    }

    /// Whether there is air next to a face of the Voxel at `at`.
    fn exposed(&mut self, at: [i32; 3]) -> bool {
        (0..6).any(|face| {
            let mut next = at;
            next[face / 2] += if face % 2 == 0 { -1 } else { 1 };
            self.voxel(next).is_none()
        })
    }
}

/// How many of one coarse Voxel's children are made of each Material.
struct Tally {
    counts: [(Material, u8); 8],
    /// How many Materials are tallied.
    len: usize,
    /// How many children are tallied, which is how many are solid.
    count: u8,
}

impl Tally {
    fn new() -> Self {
        Self {
            counts: [(Material(0), 0); 8],
            len: 0,
            count: 0,
        }
    }

    fn add(&mut self, material: Material) {
        self.count += 1;
        match self.counts[..self.len]
            .iter_mut()
            .find(|(m, _)| *m == material)
        {
            Some((_, count)) => *count += 1,
            None => {
                self.counts[self.len] = (material, 1);
                self.len += 1;
            }
        }
    }

    /// The most common Material, the lowest ID of those equally common. Panics if there is none.
    fn winner(&self) -> Material {
        self.counts[..self.len]
            .iter()
            .max_by_key(|&&(m, n)| (n, Reverse(m)))
            .map(|&(m, _)| m)
            .expect("a Material was tallied")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::LocalCoord;

    const ROCK: Material = Material(7);
    const SAND: Material = Material(3);

    fn local(i: usize) -> LocalCoord {
        [(i % 8) as u8, (i / 8 % 8) as u8, (i / 64) as u8]
    }

    /// Downsamples `children` with air all around them.
    fn downsample_in_air(children: [&Brick; 8]) -> Brick {
        downsample(children, |_| Cow::Owned(Brick::AIR))
    }

    /// Downsamples eight copies of `brick`.
    fn of_eight(brick: &Brick) -> Brick {
        downsample_in_air([brick; 8])
    }

    /// The 2×2×2 Voxels that make coarse Voxel (0, 0, 0).
    fn cell() -> [LocalCoord; 8] {
        std::array::from_fn(|i| [(i & 1) as u8, (i >> 1 & 1) as u8, (i >> 2) as u8])
    }

    /// Coarse Voxel (0, 0, 0) when its cell holds `voxels` and the other children are air.
    fn coarse_origin(voxels: &[Option<Material>; 8]) -> Option<Material> {
        let mut brick = Brick::AIR;
        for (at, &voxel) in cell().into_iter().zip(voxels) {
            brick.set(at, voxel);
        }
        let air = Brick::AIR;
        downsample_in_air([&brick, &air, &air, &air, &air, &air, &air, &air]).get([0, 0, 0])
    }

    /// The Materials the exposed-surface tests use. BEDROCK has the lowest ID, so it wins every
    /// tie that is decided by ID, and no majority rule can favour SOIL by accident.
    const BEDROCK: Material = Material(2);
    const SOIL: Material = Material(9);

    /// The coarse brick over the eight bricks at (0..2)³ of a world given Voxel by Voxel, made
    /// the way the store makes it: the bricks around them are read as the world has them.
    fn coarse_of(world: impl Fn([i32; 3]) -> Option<Material>) -> Brick {
        let brick_at = |at: [i32; 3]| {
            let mut brick = Brick::AIR;
            for i in 0..BRICK_VOXELS {
                let local = local(i);
                let voxel = BrickCoord(at).voxel(local).0;
                brick.set(local, world(voxel));
            }
            Cow::Owned(brick)
        };
        let children = BrickCoord([0; 3]).children().map(|c| brick_at(c.0));
        downsample(children.each_ref().map(|c| &**c), brick_at)
    }

    /// Ground that is `top` Voxels up: SOIL for the `thin` Voxels under its surface, BEDROCK
    /// below that, and air above, whatever x and y are.
    fn layered(top: i32, thin: i32) -> impl Fn([i32; 3]) -> Option<Material> {
        move |[_, _, z]| match z {
            z if z >= top => None,
            z if z >= top - thin => Some(SOIL),
            _ => Some(BEDROCK),
        }
    }

    #[test]
    fn a_solid_voxel_exposed_to_air_decides_the_material_before_the_buried_ones() {
        // Voxels 2 and 3 in z are the top of the ground in coarse Voxel (0, 0, 1): its upper
        // layer is soil with air above it, its lower layer bedrock. Four of each, and bedrock
        // has the lower ID, so counting alone would say bedrock.
        let coarse = coarse_of(layered(4, 1));
        assert_eq!(coarse.get([0, 0, 1]), Some(SOIL));
        // The Voxel under it is all bedrock, and so are the others below.
        assert_eq!(coarse.get([0, 0, 0]), Some(BEDROCK));
    }

    #[test]
    fn a_voxel_with_no_air_next_to_it_keeps_the_most_common_material() {
        // Five Voxels of soil and three of bedrock, all of them buried: a solid block with
        // solid Voxels all around it.
        let coarse = coarse_of(|[x, y, z]| {
            let inside = (0..2).contains(&x) && (0..2).contains(&y) && (0..2).contains(&z);
            Some(match (inside, x + 2 * y + 4 * z) {
                (true, i) if i < 3 => BEDROCK,
                _ => SOIL,
            })
        });
        assert_eq!(coarse.get([0, 0, 0]), Some(SOIL));
        // And with the Materials the other way round the lowest ID does not matter: the
        // majority is bedrock.
        let coarse = coarse_of(|[x, y, z]| {
            let inside = (0..2).contains(&x) && (0..2).contains(&y) && (0..2).contains(&z);
            Some(match (inside, x + 2 * y + 4 * z) {
                (true, i) if i < 3 => SOIL,
                _ => BEDROCK,
            })
        });
        assert_eq!(coarse.get([0, 0, 0]), Some(BEDROCK));
        // Buried ties still go to the lowest ID.
        let coarse = coarse_of(|[x, y, z]| {
            let inside = (0..2).contains(&x) && (0..2).contains(&y) && (0..2).contains(&z);
            Some(if inside && z == 0 { SOIL } else { BEDROCK })
        });
        assert_eq!(coarse.get([0, 0, 0]), Some(BEDROCK));
    }

    #[test]
    fn a_block_that_is_all_solid_is_surface_when_the_air_is_in_the_coarse_voxel_next_to_it() {
        // Ground up to z = 6, so coarse Voxel (0, 0, 2) is all solid: soil in its top layer
        // and bedrock under it. The air is in (0, 0, 3), one coarse Voxel up.
        assert_eq!(coarse_of(layered(6, 1)).get([0, 0, 2]), Some(SOIL));
        assert_eq!(coarse_of(layered(6, 1)).get([0, 0, 3]), None);
    }

    #[test]
    fn the_air_can_be_in_the_next_brick_over() {
        // Ground up to z = 16, the top of the eight bricks. Coarse Voxel (0, 0, 7) is the last
        // of its brick, and its air is the finer brick above them, which only the surroundings
        // know. Its x and y neighbours are in the next brick over too.
        assert_eq!(coarse_of(layered(16, 1)).get([0, 0, 7]), Some(SOIL));
        assert_eq!(coarse_of(layered(16, 1)).get([7, 7, 7]), Some(SOIL));
        // Without that air it is buried, and the tie goes to bedrock.
        assert_eq!(coarse_of(layered(17, 1)).get([0, 0, 7]), Some(BEDROCK));
    }

    #[test]
    fn a_wall_is_exposed_on_its_side_whatever_its_depth() {
        // Ground for x < 16, in layers of x: a skin of soil at x = 15 over bedrock, and air
        // beyond x = 16 which is only in the next brick over.
        let coarse = coarse_of(|[x, _, _]| match x {
            x if x >= 16 => None,
            15 => Some(SOIL),
            _ => Some(BEDROCK),
        });
        assert_eq!(coarse.get([7, 3, 3]), Some(SOIL));
        assert_eq!(coarse.get([6, 3, 3]), Some(BEDROCK));
    }

    #[test]
    fn a_voxel_is_solid_by_the_same_count_whether_or_not_it_is_exposed() {
        // Three solid Voxels of a block are still too few, and 4 enough, with the surface rule.
        let world = |solid: i32| {
            coarse_of(move |[x, y, z]| {
                let at = x + 2 * y + 4 * z;
                (x < 2 && y < 2 && z < 2 && at < solid).then_some(SOIL)
            })
            .get([0, 0, 0])
        };
        assert_eq!(world(3), None);
        assert_eq!(world(4), Some(SOIL));
    }

    #[test]
    fn eight_air_bricks_make_an_air_brick() {
        assert!(of_eight(&Brick::AIR).is_air());
    }

    #[test]
    fn eight_bricks_of_one_material_make_a_uniform_brick_of_it() {
        let coarse = of_eight(&Brick::solid(ROCK));
        assert!(coarse.is_uniform());
        assert!((0..BRICK_VOXELS).all(|i| coarse.get(local(i)) == Some(ROCK)));
    }

    #[test]
    fn eight_solid_bricks_of_different_materials_make_a_mixed_brick() {
        let (rock, sand) = (Brick::solid(ROCK), Brick::solid(SAND));
        let coarse = downsample_in_air([&rock, &sand, &rock, &rock, &rock, &rock, &rock, &rock]);
        assert!(!coarse.is_uniform());
        // The second child is the one at x + 1, which fills x in 4..8.
        assert_eq!(coarse.get([3, 0, 0]), Some(ROCK));
        assert_eq!(coarse.get([4, 0, 0]), Some(SAND));
        assert_eq!(coarse.get([7, 3, 3]), Some(SAND));
        assert_eq!(coarse.get([4, 4, 0]), Some(ROCK));
    }

    #[test]
    fn a_child_decides_the_octant_of_the_coarse_brick_it_fills() {
        let (rock, air) = (Brick::solid(ROCK), Brick::AIR);
        // Child 5 is the one at x + 1 and z + 1.
        let coarse = downsample_in_air([&air, &air, &air, &air, &air, &rock, &air, &air]);
        for i in 0..BRICK_VOXELS {
            let [x, y, z] = local(i);
            let in_octant = x >= 4 && y < 4 && z >= 4;
            assert_eq!(coarse.get([x, y, z]).is_some(), in_octant, "{x} {y} {z}");
        }
    }

    #[test]
    fn a_coarse_voxel_reads_the_cell_at_double_its_position() {
        // Coarse Voxel (1, 2, 3) of the first child reads Voxels 2..=3, 4..=5 and 6..=7.
        let mut block = Brick::AIR;
        for at in [[2, 4, 6], [3, 4, 6], [2, 5, 6], [3, 5, 6]] {
            block.set(at, Some(ROCK));
        }
        let air = Brick::AIR;
        let coarse = downsample_in_air([&block, &air, &air, &air, &air, &air, &air, &air]);
        assert_eq!(coarse.get([1, 2, 3]), Some(ROCK));
        assert_eq!(coarse.get([0, 2, 3]), None);
        assert_eq!(coarse.get([1, 1, 3]), None);
        assert_eq!(coarse.get([1, 2, 2]), None);
    }

    #[test]
    fn four_solid_voxels_are_enough_and_three_are_not() {
        let mut voxels = [None; 8];
        for voxel in &mut voxels[..3] {
            *voxel = Some(ROCK);
        }
        assert_eq!(coarse_origin(&voxels), None, "3 of 8");
        voxels[3] = Some(ROCK);
        assert_eq!(coarse_origin(&voxels), Some(ROCK), "4 of 8");
        voxels[4] = Some(ROCK);
        assert_eq!(coarse_origin(&voxels), Some(ROCK), "5 of 8");
    }

    #[test]
    fn the_most_common_solid_material_wins_and_air_does_not_count_against_it() {
        let (r, s, n) = (Some(ROCK), Some(SAND), None);
        // Seven solid: four sand, three rock.
        assert_eq!(coarse_origin(&[s, s, s, s, r, r, r, n]), Some(SAND));
        // Four solid: three sand, one rock, though air is as common as all of them together.
        assert_eq!(coarse_origin(&[s, s, s, r, n, n, n, n]), Some(SAND));
        // Five solid: rock outnumbers sand, so the lower ID of sand does not help it.
        assert_eq!(coarse_origin(&[r, r, r, s, s, n, n, n]), Some(ROCK));
        // Three solid is a minority however they are split.
        assert_eq!(coarse_origin(&[s, s, r, n, n, n, n, n]), None);
    }

    #[test]
    fn a_tie_between_materials_goes_to_the_lowest_material_id() {
        let (r, s, n) = (Some(ROCK), Some(SAND), None);
        assert!(SAND < ROCK);
        assert_eq!(coarse_origin(&[r, r, r, r, s, s, s, s]), Some(SAND));
        assert_eq!(coarse_origin(&[s, s, s, s, r, r, r, r]), Some(SAND));
        // A tie among exactly the four solid Voxels needed.
        assert_eq!(coarse_origin(&[r, r, s, s, n, n, n, n]), Some(SAND));
    }

    #[test]
    fn the_result_does_not_depend_on_the_order_the_materials_were_written_in() {
        let (mut forwards, mut backwards) = (Brick::AIR, Brick::AIR);
        for (i, &at) in cell().iter().enumerate() {
            forwards.set(at, Some(Material(i as u16 % 3)));
        }
        for (i, &at) in cell().iter().enumerate().rev() {
            backwards.set(at, Some(Material(i as u16 % 3)));
        }
        assert_eq!(
            of_eight(&forwards).content_hash(),
            of_eight(&backwards).content_hash()
        );
    }
}
