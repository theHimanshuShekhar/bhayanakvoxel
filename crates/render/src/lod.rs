//! The LOD band rule: which level of the brick pyramid (ADR 0002) a brick is drawn at, from its
//! distance to the camera.
//!
//! A level `L` Voxel is `2^L` Voxels of level 0 across, and should cover at most one *output*
//! pixel. A level 0 Voxel covers one output pixel at `d0` from the camera, so level `L` is used
//! from `2^L · d0` on, until a Voxel of level `L + 1` covers one pixel, which is `2^(L+1) · d0`:
//!
//! - level 0 owns distances `[0, 2·d0)`;
//! - level `L ≥ 1` owns `[2^L·d0, 2^(L+1)·d0)`.
//!
//! `d0 = v0 · f_out / bias`, with `v0` the size of a Voxel, `f_out = output_height / (2·tan(vfov/2))`
//! the focal length in output pixels, and `bias` the output pixels one Voxel may cover (1.0 by
//! default; 2.0 is the loose rule that does a quarter of the work for up to one pixel more of
//! error). At 1440p and a 70° vertical field of view that is 102.8 m for 10 cm Voxels.
//!
//! The distance of a brick is that of the nearest point of its box, not of its centre (see
//! [`brick_distance`]). A brick is moved out to the coarser level at 1.05 times the boundary and
//! back in at 0.95 times it ([`HYSTERESIS`]), so a camera that hovers on a boundary does not flip
//! it every frame.
//!
//! All distances are in Voxels of level 0, the unit of the Planet frame, so `v0` is 1 and
//! `d0 = f_out / bias`.

use brick_store::BrickCoord;

/// Levels of the pyramid that are drawn: 0 to 12. Level 12's Voxels are 410 m across, which is
/// under a pixel at 800 km, the greatest distance of the Zoom range.
pub const LEVELS: usize = 13;

/// How far past a boundary a brick must go before it changes level, as a fraction of the
/// boundary's distance: out at `1 + HYSTERESIS` times it, back in at `1 - HYSTERESIS` times.
pub const HYSTERESIS: f64 = 0.05;

/// Height in pixels of the image the bands are for: 1440p, the output that the frame budgets are
/// for (the traced image is half of it, and TAAU upscales it).
pub const OUTPUT_HEIGHT: u32 = 1440;

/// The vertical field of view that the bands are for, in radians: the camera's, 70 degrees.
pub const VFOV: f64 = 70.0 * std::f64::consts::PI / 180.0;

/// The band rule for one view: where the boundaries between levels are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bands {
    d0: f64,
}

impl Bands {
    /// The rule for an image `output_height` pixels high with a vertical field of view of `vfov`
    /// radians, where a Voxel may cover `bias` output pixels (1.0 is the strict rule).
    pub fn new(output_height: u32, vfov: f64, bias: f64) -> Self {
        assert!(bias > 0.0 && bias.is_finite(), "bias is {bias}");
        let focal = f64::from(output_height) / (2.0 * (vfov / 2.0).tan());
        Self { d0: focal / bias }
    }

    /// The rule for 1440p output and the camera's field of view, with a Voxel allowed `bias`
    /// output pixels. `d0` is then 102.8 m for 10 cm Voxels at a bias of 1.
    pub fn standard(bias: f64) -> Self {
        Self::new(OUTPUT_HEIGHT, VFOV, bias)
    }

    /// The rule with `d0` given directly, in Voxels.
    pub fn with_d0(d0: f64) -> Self {
        assert!(d0 > 0.0 && d0.is_finite(), "d0 is {d0}");
        Self { d0 }
    }

    /// The distance at which a Voxel of level 0 covers one output pixel, in Voxels.
    pub fn d0(&self) -> f64 {
        self.d0
    }

    /// The distance at which `level` starts: 0 for level 0, else `2^level · d0`.
    pub fn floor(&self, level: usize) -> f64 {
        if level == 0 {
            0.0
        } else {
            self.d0 * (1u64 << level) as f64
        }
    }

    /// The level that owns distance `distance`, with no hysteresis, at most `top`.
    pub fn level_at(&self, distance: f64, top: usize) -> usize {
        (0..top)
            .rev()
            .find(|&level| distance >= self.floor(level + 1))
            .map_or(0, |level| level + 1)
    }

    /// Whether a brick of `level` at `distance` is to be split into the eight of the level
    /// below, because it is nearer than its level is used from. `was_split` is what the brick
    /// was last time, if it was: a brick stays as it was until it is [`HYSTERESIS`] past the
    /// boundary, which is where a camera hovering on one cannot flip it. A brick that is new
    /// is split exactly at the boundary. Level 0 is never split.
    pub fn split(&self, level: usize, distance: f64, was_split: Option<bool>) -> bool {
        if level == 0 {
            return false;
        }
        let floor = self.floor(level);
        match was_split {
            None => distance < floor,
            Some(false) => distance < floor * (1.0 - HYSTERESIS),
            Some(true) => distance < floor * (1.0 + HYSTERESIS),
        }
    }

    /// How far the camera can move before a brick of `level` at `distance`, which is `split`,
    /// could change that: the distance of a box is 1-Lipschitz in the camera's position, so it
    /// cannot change by more than the camera moves. Infinite for level 0.
    pub fn slack(&self, level: usize, distance: f64, split: bool) -> f64 {
        if level == 0 {
            return f64::INFINITY;
        }
        let floor = self.floor(level);
        let edge = if split {
            floor * (1.0 + HYSTERESIS)
        } else {
            floor * (1.0 - HYSTERESIS)
        };
        (distance - edge).abs()
    }

    /// The radius of the ball about the camera that no brick of `level` is in: all of them are
    /// at least this far. Zero for level 0.
    pub fn clear_radius(&self, level: usize) -> f64 {
        self.floor(level) * (1.0 - HYSTERESIS)
    }
}

/// The distance from `camera`, a position in Voxels, to the nearest point of the box of the
/// brick at `coord` of mip `level`: 0 if the camera is inside it.
pub fn brick_distance(camera: [f64; 3], level: usize, coord: BrickCoord) -> f64 {
    let size = (8u64 << level) as f64;
    let mut sum = 0.0;
    for (&at, &b) in camera.iter().zip(&coord.0) {
        let low = f64::from(b) * size;
        let outside = (low - at).max(at - (low + size)).max(0.0);
        sum += outside * outside;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const VFOV: f64 = 70.0 * std::f64::consts::PI / 180.0;

    #[test]
    fn d0_is_a_voxel_times_the_focal_length_over_the_bias() {
        // 1440p, 70 degrees: f is 1028.2 pixels, so a 10 cm Voxel is one pixel at 102.8 m.
        let bands = Bands::new(1440, VFOV, 1.0);
        assert!((bands.d0() - 1028.2).abs() < 0.1, "{}", bands.d0());
        // In metres: 0.1 m a Voxel.
        assert!((bands.d0() * 0.1 - 102.8).abs() < 0.05);
        // The bias scales every distance.
        assert!((Bands::new(1440, VFOV, 2.0).d0() * 2.0 - bands.d0()).abs() < 1e-9);
        // So does the image height, and a wider view shortens them.
        assert!((Bands::new(720, VFOV, 1.0).d0() * 2.0 - bands.d0()).abs() < 1e-9);
        assert!(Bands::new(1440, 2.0 * VFOV / 1.5, 1.0).d0() < bands.d0());
    }

    #[test]
    fn level_0_owns_two_d0_and_every_level_above_owns_a_doubling() {
        let bands = Bands::with_d0(100.0);
        assert_eq!(bands.floor(0), 0.0);
        assert_eq!(bands.floor(1), 200.0);
        assert_eq!(bands.floor(2), 400.0);
        assert_eq!(bands.floor(12), 100.0 * 4096.0);
        for (distance, level) in [
            (0.0, 0),
            (199.9, 0),
            (200.0, 1),
            (399.9, 1),
            (400.0, 2),
            (799.9, 2),
            (800.0, 3),
            (1e9, 12),
        ] {
            assert_eq!(bands.level_at(distance, 12), level, "{distance}");
        }
        // A pyramid with fewer levels stops at its top.
        assert_eq!(bands.level_at(1e9, 4), 4);
        assert_eq!(bands.level_at(1e9, 0), 0);
    }

    #[test]
    fn a_new_brick_splits_exactly_at_its_boundary_and_level_0_never_does() {
        let bands = Bands::with_d0(100.0);
        assert!(bands.split(1, 199.99, None));
        assert!(!bands.split(1, 200.0, None));
        assert!(bands.split(5, 0.0, None));
        assert!(!bands.split(0, 0.0, None));
        assert!(!bands.split(0, 0.0, Some(true)));
    }

    #[test]
    fn a_brick_changes_level_only_five_percent_past_the_boundary() {
        let bands = Bands::with_d0(100.0);
        // Level 1 starts at 200. Out of it (merged into the coarser brick) at 210, back at 190.
        assert!(bands.split(1, 209.9, Some(true)));
        assert!(!bands.split(1, 210.0, Some(true)));
        assert!(!bands.split(1, 190.0, Some(false)));
        assert!(bands.split(1, 189.9, Some(false)));
        // Between 190 and 210 a brick stays as it was.
        for distance in [190.0, 199.0, 200.0, 205.0, 209.9] {
            assert!(bands.split(1, distance, Some(true)), "{distance}");
        }
        for distance in [190.0, 195.0, 200.0, 209.9, 400.0] {
            assert!(!bands.split(1, distance, Some(false)), "{distance}");
        }
    }

    #[test]
    fn a_camera_hovering_on_a_boundary_never_flips_a_brick() {
        let bands = Bands::with_d0(100.0);
        for start in [true, false] {
            let mut state = start;
            let mut flips = 0;
            // Bobbing 4% either side of the boundary at 400 (level 2), 100 times.
            for i in 0..100 {
                let distance = 400.0 * if i % 2 == 0 { 0.96 } else { 1.04 };
                let next = bands.split(2, distance, Some(state));
                flips += usize::from(next != state);
                state = next;
            }
            assert_eq!(
                flips,
                0,
                "starting {}",
                if start { "split" } else { "whole" }
            );
        }
        // Past the 5%, it flips once each way.
        let mut state = false;
        let mut flips = 0;
        for distance in [420.0, 380.0, 360.0, 380.0, 410.0, 430.0] {
            let next = bands.split(2, distance, Some(state));
            flips += usize::from(next != state);
            state = next;
        }
        assert_eq!(flips, 2);
    }

    #[test]
    fn the_slack_is_how_far_the_camera_must_go_to_reach_the_edge_of_the_hysteresis() {
        let bands = Bands::with_d0(100.0);
        // Whole at 460 (level 2 starts at 400, and it is whole until 380 is crossed): 80 to go.
        assert!((bands.slack(2, 460.0, false) - 80.0).abs() < 1e-9);
        // Split, at 300: it merges at 420, 120 away.
        assert!((bands.slack(2, 300.0, true) - 120.0).abs() < 1e-9);
        assert!(bands.slack(0, 5.0, false).is_infinite());
    }

    #[test]
    fn a_ball_about_the_camera_has_no_brick_of_the_level_in_it() {
        let bands = Bands::with_d0(100.0);
        assert_eq!(bands.clear_radius(0), 0.0);
        assert!((bands.clear_radius(3) - 760.0).abs() < 1e-9);
    }

    #[test]
    fn a_brick_is_as_far_as_the_nearest_point_of_its_box() {
        // Level 1: 16 Voxels across. Brick (1, 0, 0) is x 16..32, y 0..16, z 0..16.
        let coord = BrickCoord([1, 0, 0]);
        assert_eq!(brick_distance([20.0, 5.0, 5.0], 1, coord), 0.0);
        assert_eq!(brick_distance([40.0, 5.0, 5.0], 1, coord), 8.0);
        assert_eq!(brick_distance([0.0, 5.0, 5.0], 1, coord), 16.0);
        // Off a corner, along the diagonal.
        let corner = brick_distance([35.0, 19.0, -4.0], 1, coord);
        assert!((corner - (3.0f64 * 3.0 + 3.0 * 3.0 + 4.0 * 4.0).sqrt()).abs() < 1e-12);
        // Negative bricks, and a level that is not 1.
        assert_eq!(
            brick_distance([0.0, 0.0, 0.0], 0, BrickCoord([-1, 0, 0])),
            0.0
        );
        assert_eq!(
            brick_distance([4.0, 0.0, 0.0], 2, BrickCoord([-1, 0, 0])),
            4.0
        );
    }

    #[test]
    fn the_distance_moves_no_faster_than_the_camera() {
        // What `slack` relies on.
        let coord = BrickCoord([3, -2, 5]);
        let mut state = 7u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 40) as f64 / (1u64 << 24) as f64 * 400.0 - 200.0
        };
        for level in 0..4 {
            for _ in 0..500 {
                let a = [next(), next(), next()];
                let b = [next(), next(), next()];
                let moved = (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt();
                let change =
                    (brick_distance(a, level, coord) - brick_distance(b, level, coord)).abs();
                assert!(change <= moved + 1e-9);
            }
        }
    }
}
