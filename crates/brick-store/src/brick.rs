//! One 8×8×8 brick: a 512-bit occupancy mask plus palette-compressed Material IDs (ADR 0002).

use crate::coord::{BRICK_SIZE, LocalCoord};

/// Voxels per brick.
pub const BRICK_VOXELS: usize = (BRICK_SIZE * BRICK_SIZE * BRICK_SIZE) as usize;

/// The kind of matter a solid Voxel is made of. An air Voxel is `None` wherever a Voxel is read
/// or written, so no Material ID is reserved for air.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Material(pub u16);

/// Sparse 8×8×8 block of Voxels.
///
/// A brick that is all air or all one Material is stored as a flag only. Anything else keeps an
/// occupancy mask and a palette of the Materials used, with a packed palette index per Voxel
/// (1, 2, 4 or 8 bits, growing with the palette).
#[derive(Clone, Debug)]
pub struct Brick(Repr);

#[derive(Clone, Debug)]
enum Repr {
    Air,
    Solid(Material),
    Mixed(Box<Mixed>),
}

#[derive(Clone, Debug)]
struct Mixed {
    /// Bit `i` is set when Voxel `i` is solid.
    occupancy: [u64; 8],
    /// Materials used in this brick, in first-use order. Entries are never removed, so after
    /// Voxels are overwritten some may be unused until the brick collapses to a flag.
    palette: Vec<Material>,
    /// Bits per palette index.
    bits: u32,
    /// One packed palette index per Voxel. Indices of air Voxels are meaningless.
    indices: Vec<u8>,
}

/// Bricks are equal when every Voxel is, whatever their palette layout. A mixed brick is never
/// uniform, since `set` collapses one that is, so a mixed brick is never equal to a flag.
impl PartialEq for Brick {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Repr::Air, Repr::Air) => true,
            (Repr::Solid(a), Repr::Solid(b)) => a == b,
            (Repr::Mixed(a), Repr::Mixed(b)) => {
                a.occupancy == b.occupancy && (0..BRICK_VOXELS).all(|i| a.get(i) == b.get(i))
            }
            _ => false,
        }
    }
}

impl Eq for Brick {}

/// Read-only view of how a brick is stored, for uploading it elsewhere (e.g. to the GPU).
#[derive(Clone, Copy, Debug)]
pub enum BrickView<'a> {
    Air,
    Solid(Material),
    Mixed {
        /// Bit `i % 64` of word `i / 64` is set when Voxel `i` is solid. Voxel `i` is
        /// `x | y << 3 | z << 6`.
        occupancy: &'a [u64; 8],
        /// Materials that the packed indices point into.
        palette: &'a [Material],
        /// Bits per palette index: 1, 2, 4, 8 or 16.
        bits: u32,
        /// One `bits`-wide palette index per Voxel, packed little-endian from the lowest bit.
        /// Indices of air Voxels are meaningless.
        indices: &'a [u8],
    },
}

/// Position of a Voxel in occupancy and index order: x fastest, then y, then z.
fn voxel_index(local: LocalCoord) -> usize {
    let [x, y, z] = local.map(usize::from);
    debug_assert!(x < 8 && y < 8 && z < 8);
    x | y << 3 | z << 6
}

/// Smallest index width that can address `palette_len` entries. 16 bits only occurs when a
/// brick holds more than 256 distinct Materials.
fn bits_for(palette_len: usize) -> u32 {
    match palette_len {
        0..=2 => 1,
        3..=4 => 2,
        5..=16 => 4,
        17..=256 => 8,
        _ => 16,
    }
}

impl Mixed {
    fn empty() -> Self {
        Self {
            occupancy: [0; 8],
            palette: Vec::new(),
            bits: 1,
            indices: vec![0; BRICK_VOXELS / 8],
        }
    }

    fn filled(material: Material) -> Self {
        Self {
            occupancy: [u64::MAX; 8],
            palette: vec![material],
            ..Self::empty()
        }
    }

    fn is_solid(&self, i: usize) -> bool {
        self.occupancy[i / 64] >> (i % 64) & 1 == 1
    }

    fn read_index(&self, i: usize) -> usize {
        match self.bits {
            16 => usize::from(u16::from_le_bytes([
                self.indices[2 * i],
                self.indices[2 * i + 1],
            ])),
            8 => usize::from(self.indices[i]),
            bits => {
                let bit = i * bits as usize;
                usize::from(self.indices[bit / 8] >> (bit % 8)) & ((1 << bits) - 1)
            }
        }
    }

    fn write_index(&mut self, i: usize, index: usize) {
        match self.bits {
            16 => {
                let [lo, hi] = (index as u16).to_le_bytes();
                self.indices[2 * i] = lo;
                self.indices[2 * i + 1] = hi;
            }
            8 => self.indices[i] = index as u8,
            bits => {
                let bit = i * bits as usize;
                let mask = ((1u8 << bits) - 1) << (bit % 8);
                let byte = &mut self.indices[bit / 8];
                *byte = *byte & !mask | (index as u8) << (bit % 8);
            }
        }
    }

    /// Repacks every index at `bits` wide.
    fn widen(&mut self, bits: u32) {
        let old = self.clone();
        self.bits = bits;
        self.indices = vec![0; BRICK_VOXELS * bits as usize / 8];
        for i in (0..BRICK_VOXELS).filter(|&i| old.is_solid(i)) {
            self.write_index(i, old.read_index(i));
        }
    }

    fn get(&self, i: usize) -> Option<Material> {
        self.is_solid(i).then(|| self.palette[self.read_index(i)])
    }

    fn set(&mut self, i: usize, voxel: Option<Material>) {
        let Some(material) = voxel else {
            self.occupancy[i / 64] &= !(1 << (i % 64));
            return;
        };
        let index = self
            .palette
            .iter()
            .position(|&m| m == material)
            .unwrap_or_else(|| {
                self.palette.push(material);
                let bits = bits_for(self.palette.len());
                if bits != self.bits {
                    self.widen(bits);
                }
                self.palette.len() - 1
            });
        self.occupancy[i / 64] |= 1 << (i % 64);
        self.write_index(i, index);
    }

    /// The flag this brick can be stored as, if it is uniform.
    fn uniform(&self) -> Option<Repr> {
        if self.occupancy == [0; 8] {
            return Some(Repr::Air);
        }
        if self.occupancy != [u64::MAX; 8] {
            return None;
        }
        let first = self.get(0)?;
        (1..BRICK_VOXELS)
            .all(|i| self.get(i) == Some(first))
            .then_some(Repr::Solid(first))
    }
}

impl Brick {
    /// A brick of only air.
    pub const AIR: Brick = Brick(Repr::Air);

    /// A brick filled with one Material.
    pub fn solid(material: Material) -> Brick {
        Brick(Repr::Solid(material))
    }

    /// The Voxel at `local`: `None` for air.
    pub fn get(&self, local: LocalCoord) -> Option<Material> {
        match &self.0 {
            Repr::Air => None,
            Repr::Solid(material) => Some(*material),
            Repr::Mixed(mixed) => mixed.get(voxel_index(local)),
        }
    }

    /// Sets the Voxel at `local`. A brick that becomes uniform collapses back to a flag.
    pub fn set(&mut self, local: LocalCoord, voxel: Option<Material>) {
        if self.get(local) == voxel {
            return;
        }
        let mut mixed = match std::mem::replace(&mut self.0, Repr::Air) {
            Repr::Mixed(mixed) => mixed,
            Repr::Air => Box::new(Mixed::empty()),
            Repr::Solid(material) => Box::new(Mixed::filled(material)),
        };
        mixed.set(voxel_index(local), voxel);
        self.0 = mixed.uniform().unwrap_or(Repr::Mixed(mixed));
    }

    /// The brick's stored representation.
    pub fn view(&self) -> BrickView<'_> {
        match &self.0 {
            Repr::Air => BrickView::Air,
            Repr::Solid(material) => BrickView::Solid(*material),
            Repr::Mixed(mixed) => BrickView::Mixed {
                occupancy: &mixed.occupancy,
                palette: &mixed.palette,
                bits: mixed.bits,
                indices: &mixed.indices,
            },
        }
    }

    /// True when every Voxel of the brick is solid.
    pub fn is_full(&self) -> bool {
        match &self.0 {
            Repr::Air => false,
            Repr::Solid(_) => true,
            Repr::Mixed(mixed) => mixed.occupancy.iter().all(|&w| w == u64::MAX),
        }
    }

    /// True when the brick is all air.
    pub fn is_air(&self) -> bool {
        matches!(self.0, Repr::Air)
    }

    /// True when the brick is all air or all one Material, and so has no per-Voxel storage.
    pub fn is_uniform(&self) -> bool {
        !matches!(self.0, Repr::Mixed(_))
    }

    /// Bits per palette index: 0 for a uniform brick, otherwise 1, 2, 4, 8 or 16.
    pub fn index_bits(&self) -> u32 {
        match &self.0 {
            Repr::Mixed(mixed) => mixed.bits,
            _ => 0,
        }
    }

    /// Heap bytes held for per-Voxel data: 0 for a uniform brick.
    pub fn storage_bytes(&self) -> usize {
        match &self.0 {
            Repr::Mixed(mixed) => mixed.indices.len() + mixed.palette.len() * size_of::<Material>(),
            _ => 0,
        }
    }

    /// FNV-1a hash of the canonical content: occupancy and Material of every Voxel in index
    /// order (x fastest). It ignores palette layout, so equal content always hashes equal and
    /// storage refactors do not change it.
    pub fn content_hash(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat =
            |byte: u8| hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        for i in 0..BRICK_VOXELS {
            match self.get([(i & 7) as u8, (i >> 3 & 7) as u8, (i >> 6) as u8]) {
                None => [0, 0, 0],
                Some(Material(id)) => {
                    let [lo, hi] = id.to_le_bytes();
                    [1, lo, hi]
                }
            }
            .into_iter()
            .for_each(&mut eat);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROCK: Material = Material(7);

    fn local(i: usize) -> LocalCoord {
        [(i % 8) as u8, (i / 8 % 8) as u8, (i / 64) as u8]
    }

    #[test]
    fn a_new_brick_is_all_air() {
        let b = Brick::AIR;
        assert!((0..BRICK_VOXELS).all(|i| b.get(local(i)).is_none()));
    }

    #[test]
    fn a_solid_brick_reads_its_material_everywhere() {
        let b = Brick::solid(ROCK);
        assert!((0..BRICK_VOXELS).all(|i| b.get(local(i)) == Some(ROCK)));
    }

    #[test]
    fn set_then_get_round_trips_without_disturbing_neighbours() {
        let mut b = Brick::AIR;
        b.set([3, 4, 5], Some(ROCK));
        assert_eq!(b.get([3, 4, 5]), Some(ROCK));
        assert_eq!(b.get([4, 4, 5]), None);
        assert_eq!(b.get([3, 5, 5]), None);
        assert_eq!(b.get([3, 4, 6]), None);
        b.set([3, 4, 5], None);
        assert_eq!(b.get([3, 4, 5]), None);
    }

    #[test]
    fn digging_one_voxel_out_of_a_solid_brick_keeps_the_rest() {
        let mut b = Brick::solid(ROCK);
        b.set([7, 7, 7], None);
        assert_eq!(b.get([7, 7, 7]), None);
        assert_eq!(b.get([0, 0, 0]), Some(ROCK));
        assert_eq!(b.get([7, 7, 6]), Some(ROCK));
    }

    #[test]
    fn uniform_bricks_have_no_per_voxel_storage() {
        assert!(Brick::AIR.is_air() && !Brick::solid(ROCK).is_air());
        for b in [Brick::AIR, Brick::solid(ROCK)] {
            assert!(b.is_uniform());
            assert_eq!(b.storage_bytes(), 0);
            assert_eq!(b.index_bits(), 0);
        }
    }

    #[test]
    fn writing_the_value_already_there_keeps_a_brick_uniform() {
        let mut b = Brick::solid(ROCK);
        b.set([1, 2, 3], Some(ROCK));
        assert!(b.is_uniform());
        let mut b = Brick::AIR;
        b.set([1, 2, 3], None);
        assert!(b.is_uniform());
    }

    #[test]
    fn bricks_with_the_same_voxels_are_equal_whatever_order_they_were_written_in() {
        // Different palette order, and so different index layout, for the same content.
        let voxels = [
            ([0, 0, 0], 5),
            ([1, 0, 0], 6),
            ([2, 0, 0], 7),
            ([3, 3, 3], 5),
        ];
        let (mut forwards, mut backwards) = (Brick::AIR, Brick::AIR);
        for &(at, id) in &voxels {
            forwards.set(at, Some(Material(id)));
        }
        for &(at, id) in voxels.iter().rev() {
            backwards.set(at, Some(Material(id)));
        }
        let palette = |b: &Brick| match b.view() {
            BrickView::Mixed { palette, .. } => palette.to_vec(),
            _ => unreachable!("the brick is mixed"),
        };
        assert_ne!(palette(&forwards), palette(&backwards));
        assert_eq!(forwards, backwards);
        // Overwriting leaves an unused palette entry behind, which does not matter either.
        let mut overwritten = forwards.clone();
        overwritten.set([1, 0, 0], Some(Material(9)));
        overwritten.set([1, 0, 0], Some(Material(6)));
        assert_eq!(overwritten, forwards);
    }

    #[test]
    fn bricks_with_different_voxels_are_not_equal() {
        let mut base = Brick::AIR;
        base.set([1, 2, 3], Some(ROCK));
        let mut other_material = Brick::AIR;
        other_material.set([1, 2, 3], Some(Material(8)));
        let mut other_place = Brick::AIR;
        other_place.set([1, 2, 4], Some(ROCK));
        let mut extra = base.clone();
        extra.set([5, 5, 5], Some(ROCK));
        for other in [&other_material, &other_place, &extra, &Brick::AIR] {
            assert_ne!(&base, other);
        }
        // A flag is never equal to a mixed brick, nor air to a solid one.
        assert_ne!(Brick::AIR, Brick::solid(ROCK));
        assert_ne!(Brick::solid(ROCK), Brick::solid(Material(8)));
        assert_eq!(Brick::solid(ROCK), Brick::solid(ROCK));
        assert_ne!(Brick::solid(ROCK), base);
    }

    #[test]
    fn a_mixed_brick_is_not_uniform_and_has_storage() {
        let mut b = Brick::AIR;
        b.set([0, 0, 0], Some(ROCK));
        assert!(!b.is_uniform());
        assert!(b.storage_bytes() > 0);
    }

    #[test]
    fn clearing_the_last_solid_voxel_collapses_to_air() {
        let mut b = Brick::AIR;
        b.set([0, 0, 0], Some(ROCK));
        b.set([0, 0, 0], None);
        assert!(b.is_uniform());
        assert_eq!(b.storage_bytes(), 0);
    }

    #[test]
    fn filling_the_last_air_voxel_collapses_to_solid() {
        let mut b = Brick::solid(ROCK);
        b.set([2, 2, 2], None);
        assert!(!b.is_uniform());
        b.set([2, 2, 2], Some(ROCK));
        assert!(b.is_uniform());
        assert_eq!(b.storage_bytes(), 0);
        assert_eq!(b.get([5, 5, 5]), Some(ROCK));
    }

    #[test]
    fn a_full_brick_of_one_material_collapses_even_after_a_second_material_was_used() {
        let mut b = Brick::solid(ROCK);
        b.set([0, 0, 0], Some(Material(9)));
        b.set([0, 0, 0], Some(ROCK));
        assert!(b.is_uniform());
    }

    /// Fills the first `n` Voxels with `n` distinct Materials.
    fn with_distinct_materials(n: usize) -> Brick {
        let mut b = Brick::AIR;
        for i in 0..n {
            b.set(local(i), Some(Material(1000 + i as u16)));
        }
        b
    }

    #[test]
    fn palette_index_width_grows_with_the_palette() {
        for (distinct, bits) in [
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 4),
            (16, 4),
            (17, 8),
            (256, 8),
            (257, 16),
            (512, 16),
        ] {
            let b = with_distinct_materials(distinct);
            assert_eq!(b.index_bits(), bits, "{distinct} Materials");
        }
    }

    #[test]
    fn growing_the_palette_preserves_every_voxel() {
        let b = with_distinct_materials(512);
        for i in 0..512 {
            assert_eq!(
                b.get(local(i)),
                Some(Material(1000 + i as u16)),
                "voxel {i}"
            );
        }
    }

    #[test]
    fn mixed_air_and_materials_survive_each_width() {
        for distinct in [2, 3, 9, 100, 300] {
            let mut b = with_distinct_materials(distinct);
            // Punch air into every third Voxel and check the rest are untouched.
            for i in (0..distinct).step_by(3) {
                b.set(local(i), None);
            }
            for i in 0..distinct {
                let want = (i % 3 != 0).then_some(Material(1000 + i as u16));
                assert_eq!(b.get(local(i)), want, "{distinct} Materials, voxel {i}");
            }
            assert_eq!(b.get(local(distinct)), None);
        }
    }

    #[test]
    fn content_hash_of_air_matches_an_independent_fnv1a() {
        // FNV-1a 64 over 512 Voxels of [0, 0, 0] (air), computed outside this crate.
        assert_eq!(Brick::AIR.content_hash(), 0x6ab0_5ef9_aa8b_9b25);
    }

    #[test]
    fn content_hash_ignores_how_the_content_was_built() {
        // A mixed brick that collapsed after using a second Material equals the plain flag.
        let mut built = Brick::solid(ROCK);
        built.set([0, 0, 0], Some(Material(1)));
        built.set([0, 0, 0], Some(ROCK));
        assert_eq!(built.content_hash(), Brick::solid(ROCK).content_hash());
        let mut dug = Brick::solid(ROCK);
        dug.set([0, 0, 0], None);
        assert_ne!(dug.content_hash(), Brick::solid(ROCK).content_hash());
    }

    #[test]
    fn the_view_exposes_the_packed_representation() {
        assert!(matches!(Brick::AIR.view(), BrickView::Air));
        assert!(matches!(Brick::solid(ROCK).view(), BrickView::Solid(m) if m == ROCK));
        let mut b = Brick::AIR;
        b.set([1, 0, 0], Some(ROCK));
        b.set([2, 0, 0], Some(Material(9)));
        let BrickView::Mixed {
            occupancy,
            palette,
            bits,
            indices,
        } = b.view()
        else {
            panic!("expected a mixed brick");
        };
        assert_eq!(occupancy[0], 0b110);
        assert_eq!(palette, [ROCK, Material(9)]);
        assert_eq!(bits, 1);
        // Voxel 1 uses palette index 0 and voxel 2 uses index 1: bit 2 of byte 0.
        assert_eq!(indices[0], 0b100);
    }

    #[test]
    fn storage_grows_with_index_width() {
        let small = with_distinct_materials(2).storage_bytes();
        let large = with_distinct_materials(17).storage_bytes();
        assert!(large > small);
    }
}
