//! The test Planet in rings (`Scene::rings`), each brick at the level its distance from the camera
//! says (the LOD bands), at every distance from the surface to orbit: that the GPU and the CPU
//! walk them alike, that they show no cracks and match the approved golden images, that the
//! surface moves by under a pixel where the level changes, that moving the camera writes only
//! what changes, and that the GPU memory they take stays bounded. Needs a Vulkan device
//! (lavapipe in CI).
//!
//! The whole Planet takes about half a minute to generate, so the tests share one.

mod common;

use std::sync::OnceLock;

use brick_store::BrickStore;
use common::{Lcg, PRIMARY, assert_agree, check_golden, ray};
use render::lod::Bands;
use render::reference::{Hit, Ray, trace};
use render::scene::{Bound, Core};
use render::{Camera, Image, Offscreen, Scene, SceneUpdate};

const SEED: u64 = 0xB4A7_A9AC;
/// Levels of the Planet's pyramid: as many as the client draws (0 to 12).
const LEVELS: usize = render::lod::LEVELS;
const WIDTH: u32 = 256;
const HEIGHT: u32 = 144;

/// The Planet with the levels of its pyramid that the rings are made of.
fn planet() -> &'static BrickStore {
    static PLANET: OnceLock<BrickStore> = OnceLock::new();
    PLANET.get_or_init(|| {
        let mut store = planet_gen::generate_planet(SEED);
        store.build_lod(LEVELS - 1);
        store
    })
}

/// The Planet's bounding ball, as the client gives it to the scene.
fn ball() -> Bound {
    Bound {
        centre: [0; 3],
        radius: planet_gen::MAX_RADIUS,
    }
}

fn rings_with(camera: [f64; 3], bands: Bands) -> Scene {
    Scene::rings(planet(), camera, bands, ball()).unwrap()
}

/// The strict rule at 1440p output and a 70 degree field of view: a level 0 Voxel is a pixel
/// at 102.8 m.
fn rings_at(camera: [f64; 3]) -> Scene {
    rings_with(camera, Bands::standard(1.0))
}

/// `scene` without its bounding ball, so that rays are not clipped to it.
fn unclipped(mut scene: Scene) -> Scene {
    scene.bound = None;
    scene
}

/// Where the camera is and what it looks at, for the views that are at the surface, at the
/// horizon of a hill (where the ground crosses from one ring to the next), and in orbit.
const SURFACE: ([f64; 3], [f64; 3]) = ([0.0, 1548.0, 250.0], [0.0, 1535.0, -300.0]);
const HORIZON: ([f64; 3], [f64; 3]) = ([0.0, 1700.0, 600.0], [0.0, 1450.0, -400.0]);
const ORBIT: ([f64; 3], [f64; 3]) = ([0.0, 400.0, 3400.0], [0.0; 3]);

fn render_view((from, to): ([f64; 3], [f64; 3])) -> Image {
    Offscreen::new(WIDTH, HEIGHT, &rings_at(from))
        .unwrap()
        .render(&Camera::looking_at(from, to))
}

#[test]
fn the_test_planet_from_the_surface_matches_the_approved_golden_image() {
    check_golden(
        "planet_rings_surface_256x144.png",
        &render_view(SURFACE),
        &PRIMARY,
    );
}

/// The ground runs on from the near ring to the next one at the horizon: the place where a
/// crack between rings would show as sky.
#[test]
fn the_test_planet_across_two_rings_matches_the_approved_golden_image() {
    check_golden(
        "planet_rings_horizon_256x144.png",
        &render_view(HORIZON),
        &PRIMARY,
    );
}

#[test]
fn the_test_planet_from_orbit_matches_the_approved_golden_image() {
    check_golden(
        "planet_rings_orbit_256x144.png",
        &render_view(ORBIT),
        &PRIMARY,
    );
}

/// The view from the surface is all in the first ring, which holds the same Voxels as the
/// single-level scene of a patch of the Planet that the older golden image is of: the two must
/// agree.
#[test]
fn the_rings_draw_the_surface_as_the_patch_of_the_planet_does() {
    check_golden("planet_256x144.png", &render_view(SURFACE), &PRIMARY);
}

/// Rays from `camera` at random points of the Planet, a little above or below its surface.
fn rays_at_the_planet(camera: [f64; 3], count: usize, rng: &mut Lcg) -> Vec<Ray> {
    (0..count)
        .map(|_| {
            let on = [rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5];
            let len = on.iter().map(|c| c * c).sum::<f32>().sqrt();
            let radius = f64::from(1450.0 + 150.0 * rng.next());
            ray(camera, on.map(|c| f64::from(c / len) * radius))
        })
        .collect()
}

/// Cameras at a hand's breadth from the ground, over a hill, in the sky and in orbit.
const CAMERAS: [[f64; 3]; 5] = [
    [0.0, 1548.0, 250.0],
    [0.0, 1700.0, 600.0],
    [800.0, 2600.0, 1500.0],
    [0.0, 400.0, 3400.0],
    [-9000.0, 700.0, 6000.0],
];

#[test]
fn the_gpu_and_the_cpu_agree_on_rings_of_the_test_planet_from_the_surface_to_orbit() {
    let mut rng = Lcg(2025);
    for camera in CAMERAS {
        let scene = rings_at(camera);
        let rays = rays_at_the_planet(camera, 300, &mut rng);
        let hits = assert_agree(&format!("camera {camera:?}"), &rays, &scene);
        assert!(hits > 100, "camera {camera:?}: {hits} of 300 rays hit");
    }
}

/// The Planet at every Voxel of level 0, which is what the rings are of: the fine truth that no
/// ray may get through to the sky past.
fn whole_planet() -> Scene {
    let core = Core {
        radius: planet_gen::CORE_RADIUS,
        material: planet_gen::ROCK,
    };
    Scene::new(planet(), Some(core))
}

#[test]
fn no_ray_gets_through_to_the_sky_where_the_whole_planet_has_ground() {
    let truth = whole_planet();
    let mut rng = Lcg(77);
    for camera in CAMERAS {
        let scene = rings_at(camera);
        let (mut through, mut hits) = (0, 0);
        for ray in rays_at_the_planet(camera, 1500, &mut rng) {
            let (want, got): (Option<Hit>, Option<Hit>) =
                (trace(&truth, &ray), trace(&scene, &ray));
            hits += usize::from(want.is_some());
            through += usize::from(want.is_some() && got.is_none());
        }
        assert!(hits > 300, "camera {camera:?}: {hits} rays hit the Planet");
        assert_eq!(
            through, 0,
            "camera {camera:?}: {through} of {hits} rays get through the rings to the sky"
        );
    }
}

#[test]
fn bricks_are_selected_by_level_where_the_bands_say_and_the_bias_changes_them() {
    // From the surface most of the test Planet is within 2 * d0 of the camera: level 0. From
    // orbit it is far, and the levels in use are coarse.
    let near = rings_at(SURFACE.0);
    assert!(near.level_bricks(0) > 10_000);
    // The far side is over 2 * d0 (2056 Voxels) away, and under 4 * d0.
    let counts: Vec<usize> = (0..LEVELS).map(|l| near.level_bricks(l)).collect();
    eprintln!("surface bricks per level {counts:?}");
    assert!(
        near.level_bricks(1) > 0 && near.level_bricks(2) == 0,
        "{counts:?}"
    );
    let far = rings_at(ORBIT.0);
    let used: Vec<_> = (0..LEVELS).filter(|&l| far.level_bricks(l) > 0).collect();
    eprintln!("orbit uses levels {used:?}");
    assert!(!used.is_empty());
    // A bias of 2 halves every switch distance: the same view uses coarser levels, and fewer
    // bricks.
    let loose = rings_with(ORBIT.0, Bands::standard(2.0));
    let bricks = |s: &Scene| (0..LEVELS).map(|l| s.level_bricks(l)).sum::<usize>();
    assert!(
        bricks(&loose) < bricks(&far),
        "{} {}",
        bricks(&loose),
        bricks(&far)
    );
    // 2200 Voxels from the surface level 1 has begun with a bias of 1 (it starts at 2056), but
    // not with a bias of 2, where level 0 has ended at 1028.
    let view = [0.0, f64::from(planet_gen::MAX_RADIUS) + 2200.0, 250.0];
    assert!(rings_at(view).level_bricks(1) > 0);
    assert_eq!(rings_with(view, Bands::standard(2.0)).level_bricks(0), 0);
    assert!(rings_with(view, Bands::standard(2.0)).level_bricks(2) > 0);
}

/// Camera positions straight above the Planet's +y pole, at the switch distances of levels 1
/// to 6 (from the surface's highest point), and just either side of them.
fn switch_cameras(bias: f64) -> Vec<(usize, [f64; 3])> {
    let d0 = Bands::standard(bias).d0();
    let top = f64::from(planet_gen::MAX_RADIUS);
    (1..=6)
        .flat_map(|level| {
            let at = d0 * (1u64 << level) as f64;
            [0.97, 1.0, 1.03]
                .into_iter()
                .map(move |k| (level, [0.0, top + at * k, 0.0]))
        })
        .collect()
}

/// Rays from `camera` at the cap of the Planet that faces it.
fn rays_at_the_disc(camera: [f64; 3], count: usize, rng: &mut Lcg) -> Vec<Ray> {
    let distance = camera.iter().map(|c| c * c).sum::<f64>().sqrt();
    let to_camera = camera.map(|c| c / distance);
    // Two unit vectors across the direction to the camera.
    let across = if to_camera[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let dot: f64 = (0..3).map(|a| across[a] * to_camera[a]).sum();
    let u: [f64; 3] = std::array::from_fn(|a| across[a] - to_camera[a] * dot);
    let len = u.iter().map(|c| c * c).sum::<f64>().sqrt();
    let u = u.map(|c| c / len);
    let w = [
        to_camera[1] * u[2] - to_camera[2] * u[1],
        to_camera[2] * u[0] - to_camera[0] * u[2],
        to_camera[0] * u[1] - to_camera[1] * u[0],
    ];
    (0..count)
        .map(|_| {
            // A point of the cap that faces the camera, within 40 degrees of its pole.
            let (r, v) = (f64::from(rng.next()), f64::from(rng.next()));
            let angle = 0.7 * r.sqrt();
            let turn = v * std::f64::consts::TAU;
            let on: [f64; 3] = std::array::from_fn(|a| {
                to_camera[a] * angle.cos() + (u[a] * turn.cos() + w[a] * turn.sin()) * angle.sin()
            });
            let radius = 1400.0 + 160.0 * f64::from(rng.next());
            ray(camera, on.map(|c| c * radius))
        })
        .collect()
}

/// No ray that hits the Planet at level 0 misses it after the bands have chosen the levels, at
/// every switch distance and a hair either side of it, nor does it with a bias of 2: the bands
/// partition space, so a level boundary can open no gap.
#[test]
fn rays_that_hit_at_level_0_still_hit_after_selection_at_every_switch_distance() {
    let truth = whole_planet();
    let mut rng = Lcg(5150);
    for bias in [1.0, 2.0] {
        for (level, camera) in switch_cameras(bias) {
            let scene = rings_with(camera, Bands::standard(bias));
            let (mut hits, mut through) = (0, 0);
            for ray in rays_at_the_disc(camera, 400, &mut rng) {
                let want = trace(&truth, &ray);
                hits += usize::from(want.is_some());
                through += usize::from(want.is_some() && trace(&scene, &ray).is_none());
            }
            assert!(hits > 100, "level {level}: {hits} hits");
            assert_eq!(through, 0, "bias {bias}, level {level}, camera {camera:?}");
        }
    }
}

/// How far the surface moves where a level takes over: for rays that hit both the whole Planet
/// at level 0 and the bands' selection, the difference of the distances along the ray, in
/// output pixels (a Voxel of size 1 covers one pixel at `t` of `f_out`, so a distance of `d`
/// Voxels covers `d * f_out / t` of them).
fn displacements(bias: f64, held: bool, rng: &mut Lcg) -> Vec<(usize, Vec<f64>)> {
    let truth = whole_planet();
    let bands = Bands::standard(bias);
    let focal = bands.d0() * bias;
    let top = f64::from(planet_gen::MAX_RADIUS);
    (1..=6)
        .map(|level| {
            // The nearest surface is at the inner edge of the level's band: as near as a fresh
            // selection goes. With `held`, the camera came in from further out, and the bricks
            // stay at their level down to 0.95 of the distance (the hysteresis): as near as any
            // brick goes.
            let (from, to) = if held {
                (bands.floor(level) * 1.06, bands.floor(level) * 0.951)
            } else {
                (bands.floor(level) * 1.001, bands.floor(level) * 1.001)
            };
            let mut scene = rings_with([0.0, top + from, 0.0], bands);
            let camera = [0.0, top + to, 0.0];
            if held {
                // In steps that are patches, not jumps that make the scene again (see `JUMP`),
                // so that the bricks are the ones that stayed at their level on the way in.
                let step = bands.d0();
                let distance = from - to;
                let steps = (distance / step).ceil() as u32;
                for i in 1..=steps {
                    let at = from - distance * f64::from(i) / f64::from(steps);
                    let update = scene.recentre(planet(), [0.0, top + at, 0.0]).unwrap();
                    assert_ne!(update, Some(SceneUpdate::Rebuilt), "level {level}");
                }
            }
            let mut errors = Vec::new();
            for ray in rays_at_the_disc(camera, 1500, rng) {
                let (Some(want), Some(got)) = (trace(&truth, &ray), trace(&scene, &ray)) else {
                    continue;
                };
                errors.push(f64::from((want.t - got.t).abs()) * focal / f64::from(want.t));
            }
            (level, errors)
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() as f64 * p).ceil() as usize).clamp(1, sorted.len()) - 1]
}

/// The surface at the switch distance is within a pixel of the Voxels of level 0 for 95% of rays
/// (the catalogue's gate, #81): measured, and printed for the report, with the worst ray. A fresh
/// selection switches at the boundary itself; the hysteresis lets a brick stay at its level down
/// to 0.95 of it, where its Voxels are a twentieth of a pixel bigger, so that is measured too: the
/// camera comes in from 1.06 times the boundary in steps that patch the scene (it would make it
/// again for a jump), and p95 is under a pixel at every level either way. The worst ray is not:
/// 1.7 px fresh and 4.3 px held, at level 1.
#[test]
fn the_surface_moves_by_under_a_pixel_at_the_switch_distances() {
    let mut rng = Lcg(81);
    for held in [false, true] {
        for (level, mut errors) in displacements(1.0, held, &mut rng) {
            assert!(errors.len() > 400, "level {level}: {} rays", errors.len());
            errors.sort_by(f64::total_cmp);
            let mean = errors.iter().sum::<f64>() / errors.len() as f64;
            let (p95, max) = (percentile(&errors, 0.95), errors[errors.len() - 1]);
            eprintln!(
                "{} level {level}: displacement {mean:.2} px mean, {p95:.2} p95, {max:.2} max ({} rays)",
                if held { "at 0.95" } else { "at 1.00" },
                errors.len()
            );
            let limit = 1.0;
            assert!(p95 <= limit, "level {level}: p95 is {p95:.2} px");
        }
    }
}

#[test]
fn moving_the_camera_writes_only_what_changes() {
    // Two thousand Voxels above the pole, where the band of level 1 starts: a short move brings
    // bricks into it and out of it, and leaves the rest alone.
    let start = [0.0, 1548.0 + 2000.0, 200.0];
    let end = [0.0, 1548.0 + 2000.0, 330.0];
    let mut scene = rings_at(start);
    let mut gpu = Offscreen::new(WIDTH, HEIGHT, &scene).unwrap();
    let whole = gpu.uploaded_bytes();
    assert_eq!(whole, gpu.scene_bytes());

    // Not moved: nothing is written.
    assert!(scene.recentre(planet(), start).unwrap().is_none());
    assert_eq!(gpu.uploaded_bytes(), whole);

    let Some(update) = scene.recentre(planet(), end).unwrap() else {
        panic!("the bands move");
    };
    let SceneUpdate::Rings(patch) = &update else {
        panic!("the rings are patched: {update:?}");
    };
    gpu.update_scene(&scene, &update).unwrap();
    let written = gpu.uploaded_bytes() - whole;
    let words = patch.words() as u64 * 4;
    assert!(
        (words..words + 4096).contains(&written),
        "{written} bytes written for a patch of {words}"
    );
    assert!(
        written * 4 < gpu.scene_bytes(),
        "{written} bytes written of {}",
        gpu.scene_bytes()
    );

    // And what is drawn is what the same selection packed from scratch draws.
    let camera = Camera::looking_at(end, [0.0; 3]);
    let fresh = Offscreen::new(WIDTH, HEIGHT, &scene.rebuilt_rings(planet()).unwrap())
        .unwrap()
        .render(&camera);
    assert_eq!(gpu.render(&camera), fresh);
}

#[test]
fn a_flight_from_the_surface_to_orbit_and_back_keeps_the_gpu_memory_bounded() {
    let (low, high) = ([0.0, 1548.0, 250.0], [0.0, 3600.0, 3000.0]);
    let mut scene = rings_at(low);
    let mut gpu = Offscreen::new(64, 36, &scene).unwrap();
    let (mut written, mut moves, mut rebuilt, mut peak) = (0, 0, 0, gpu.scene_bytes());
    let steps = 60;
    for step in 0..=2 * steps {
        // Up in a straight line, and down again.
        let t = f64::from(step.min(2 * steps - step)) / f64::from(steps);
        let at: [f64; 3] = std::array::from_fn(|a| low[a] + (high[a] - low[a]) * t);
        let before = gpu.uploaded_bytes();
        if let Some(update) = scene.recentre(planet(), at).unwrap() {
            rebuilt += usize::from(update == SceneUpdate::Rebuilt);
            gpu.update_scene(&scene, &update).unwrap();
            moves += 1;
        }
        written += gpu.uploaded_bytes() - before;
        peak = peak.max(gpu.scene_bytes());
        // What the camera sees from here is what the same selection packed from scratch shows.
        if step % 30 == 0 {
            let camera = Camera::looking_at(at, [0.0; 3]);
            let fresh = Offscreen::new(64, 36, &scene.rebuilt_rings(planet()).unwrap())
                .unwrap()
                .render(&camera);
            assert_eq!(gpu.render(&camera), fresh, "step {step} at {at:?}");
        }
    }
    eprintln!("{moves} moves, {rebuilt} rebuilds, {written} bytes written, peak {peak}");
    assert!(moves > 20, "the rings moved {moves} times");
    assert!(rebuilt <= 4, "packed again {rebuilt} times");
    // Under the budget (~2 GB), and by far.
    assert!(peak < 512 << 20, "{peak} bytes of GPU memory");
}

/// What the walk costs, in steps (see `reference::Steps`), averaged over a fan of primary rays
/// from cameras along the bench's flight over the Planet (`planet-flyover-v1`), by what the rays
/// do: miss, hit the ground face on, or graze it.
#[derive(Clone, Copy, Default)]
struct Cost {
    rays: u32,
    steps: render::reference::Steps,
}

impl Cost {
    fn add(&mut self, steps: render::reference::Steps) {
        let s = &mut self.steps;
        self.rays += 1;
        s.examined += steps.examined;
        s.levels += steps.levels;
        s.regions += steps.regions;
        s.probes += steps.probes;
        s.bricks += steps.bricks;
        s.voxels += steps.voxels;
    }

    /// Mean steps of each kind per ray: levels looked at, levels walked, regions, probes,
    /// bricks, Voxels.
    fn mean(&self) -> [f64; 6] {
        let s = &self.steps;
        [
            s.examined, s.levels, s.regions, s.probes, s.bricks, s.voxels,
        ]
        .map(|n| f64::from(n) / f64::from(self.rays.max(1)))
    }

    /// Region, brick and Voxel steps per ray.
    fn total(&self) -> f64 {
        let m = self.mean();
        m[2] + m[4] + m[5]
    }
}

/// The keyframes of the bench's `planet-flyover-v1`: where the camera is, and what it looks at.
const FLYOVER: [([f64; 3], [f64; 3]); 5] = [
    ([0.0, 400.0, 3400.0], [0.0; 3]),
    ([0.0, 1900.0, 1500.0], [0.0, 1500.0, 0.0]),
    ([0.0, 1548.0, 250.0], [0.0, 1535.0, -300.0]),
    ([0.0, 1552.0, 0.0], [200.0, 1530.0, -300.0]),
    ([-150.0, 1556.0, -200.0], [-300.0, 1530.0, -360.0]),
];

/// The camera `frame` of `frames` along the flyover, as the bench's `CameraPath` has it.
fn flyover_camera(frame: u32, frames: u32) -> Camera {
    let segments = FLYOVER.len() - 1;
    let along = f64::from(frame) * segments as f64 / f64::from(frames - 1);
    let segment = (along.floor() as usize).min(segments - 1);
    let t = along - segment as f64;
    let lerp = |a: [f64; 3], b: [f64; 3]| [0, 1, 2].map(|i| a[i] * (1.0 - t) + b[i] * t);
    let (from, to) = (FLYOVER[segment], FLYOVER[segment + 1]);
    Camera::looking_at(lerp(from.0, to.0), lerp(from.1, to.1))
}

/// A fan of `width` by `width * 9 / 16` primary rays from `camera`.
fn primary_rays(camera: &Camera, width: u32) -> Vec<Ray> {
    let height = width * 9 / 16;
    let tan_half = (camera.fov_y / 2.0).tan();
    let (right, up, forward) = (camera.right(), camera.up(), camera.forward());
    let mut rays = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let ndc = [
                (f64::from(x) + 0.5) / f64::from(width) * 2.0 - 1.0,
                (f64::from(y) + 0.5) / f64::from(height) * 2.0 - 1.0,
            ];
            let aspect = f64::from(width) / f64::from(height);
            let d: [f64; 3] = std::array::from_fn(|a| {
                forward[a] + right[a] * ndc[0] * aspect * tan_half - up[a] * ndc[1] * tan_half
            });
            let len = d.iter().map(|c| c * c).sum::<f64>().sqrt();
            let (cell, origin) = render::camera::split(camera.position);
            rays.push(Ray {
                cell,
                origin,
                dir: d.map(|c| (c / len) as f32),
            });
        }
    }
    rays
}

/// The ray that leaves the face `hit` found on `ray`, along `dir`, which the path tracer sends
/// for the sun's light (`dir` is the sun) or for the light of the next bounce: it starts in the
/// empty Voxel in front of the face.
fn leave(hit: &Hit, dir: [f32; 3]) -> Ray {
    Ray {
        cell: std::array::from_fn(|a| hit.voxel[a] + hit.normal[a]),
        origin: [0.5; 3],
        dir,
    }
}

/// Costs of `scene` along the flyover for the three kinds of ray that the renderer casts: those
/// from the camera (primary), those from where they hit towards the sun (shadow), and those
/// bounced off it (bounce). Each is a [`Cost`] of sky, ground and grazing rays and of all.
fn flyover_cost(mut scene: Scene, rings: bool) -> [[Cost; 4]; 3] {
    let mut costs = [[Cost::default(); 4]; 3];
    let mut rng = Lcg(9);
    let sun = {
        let d: [f32; 3] = [0.4, 0.7, 0.59];
        let len = d.iter().map(|c| c * c).sum::<f32>().sqrt();
        d.map(|c| c / len)
    };
    for frame in 0..24 {
        let camera = flyover_camera(frame * 10, 240);
        if rings {
            scene.recentre(planet(), camera.position).unwrap();
        }
        for primary in primary_rays(&camera, 48) {
            let (hit, steps) = render::reference::trace_counting(&scene, &primary);
            let mut classify = |kind: usize, ray: &Ray, hit: Option<Hit>, steps| {
                let class = match hit {
                    None => 0,
                    Some(hit) => {
                        let facing: f32 = (0..3).map(|a| ray.dir[a] * hit.normal[a] as f32).sum();
                        if facing.abs() > 0.3 { 1 } else { 2 }
                    }
                };
                costs[kind][class].add(steps);
                costs[kind][3].add(steps);
            };
            classify(0, &primary, hit, steps);
            let Some(hit) = hit else { continue };
            let normal = hit.normal.map(|n| n as f32);
            // Towards the sun, if the face is lit; and bounced.
            if (0..3).map(|a| normal[a] * sun[a]).sum::<f32>() > 0.0 {
                let ray = leave(&hit, sun);
                let (found, steps) = render::reference::trace_counting(&scene, &ray);
                classify(1, &ray, found, steps);
            }
            let bounce = loop {
                let d: [f32; 3] = std::array::from_fn(|a| normal[a] + 2.0 * rng.next() - 1.0);
                let len = d.iter().map(|c| c * c).sum::<f32>().sqrt();
                if len > 0.2 {
                    break d.map(|c| c / len);
                }
            };
            if (0..3).map(|a| normal[a] * bounce[a]).sum::<f32>() > 0.0 {
                let ray = leave(&hit, bounce);
                let (found, steps) = render::reference::trace_counting(&scene, &ray);
                classify(2, &ray, found, steps);
            }
        }
    }
    costs
}

#[test]
fn the_bands_walk_about_as_many_steps_as_a_single_level_scene_along_the_bench_flight() {
    let single = flyover_cost(whole_planet(), false);
    let start = flyover_camera(0, 240).position;
    let unclipped = flyover_cost(unclipped(rings_at(start)), true);
    let rings = flyover_cost(rings_at(start), true);
    for (kind, name) in ["primary", "shadow", "bounce"].iter().enumerate() {
        for (class, what) in ["sky", "ground", "grazing", "all"].iter().enumerate() {
            let (a, u, b) = (
                single[kind][class],
                unclipped[kind][class],
                rings[kind][class],
            );
            let show = |c: &Cost| {
                let m = c.mean();
                format!(
                    "looked {:.1} walked {:.1} steps {:.1} (regions {:.1} probes {:.1} bricks {:.1} voxels {:.1})",
                    m[0],
                    m[1],
                    c.total(),
                    m[2],
                    m[3],
                    m[4],
                    m[5]
                )
            };
            eprintln!(
                "{name:8} {what:8} rays {:5}\n   single  {}\n   rings   {}\n   +clip   {}",
                b.rays,
                show(&a),
                show(&u),
                show(&b)
            );
        }
        // The rings cost what the one level does, whatever the ray: those that miss cross every
        // ring, and those from a hit cast at every level from the one they start in.
        let (a, b) = (single[kind][3], rings[kind][3]);
        assert!(
            b.total() <= 1.1 * a.total(),
            "{name}: {} steps a ray, against {}",
            b.total(),
            a.total()
        );
        // And they look at few levels: only those that list anything are walked at all.
        assert!(
            b.mean()[0] < 4.0,
            "{name}: looks at {} levels a ray",
            b.mean()[0]
        );
    }
}

/// The ball takes nothing that is in it away: what is drawn is the same with it and without it.
#[test]
fn clipping_to_the_bounding_ball_changes_no_image() {
    for view in [SURFACE, HORIZON, ORBIT, ([0.0, 9000.0, 4000.0], [0.0; 3])] {
        let camera = Camera::looking_at(view.0, view.1);
        let draw = |scene: &Scene| {
            Offscreen::new(WIDTH, HEIGHT, scene)
                .unwrap()
                .render(&camera)
        };
        let plain = unclipped(rings_at(view.0));
        let clipped = rings_at(view.0);
        assert_eq!(draw(&plain), draw(&clipped), "{view:?}");
        // The CPU walk agrees with itself too: the same Voxel, face and Material, and the
        // distance to a rounding (a region is entered where the clip began the walk, or by
        // crossing from the one before, which rounds differently).
        let mut rng = Lcg(3);
        for ray in rays_at_the_planet(view.0, 400, &mut rng) {
            let (a, b) = (trace(&plain, &ray), trace(&clipped, &ray));
            assert_eq!(
                a.map(|h| (h.voxel, h.normal, h.material)),
                b.map(|h| (h.voxel, h.normal, h.material)),
                "{ray:?}"
            );
            if let (Some(a), Some(b)) = (a, b) {
                assert!((a.t - b.t).abs() <= 1e-5 * (1.0 + a.t), "{ray:?}");
            }
        }
    }
}

/// Rays from `camera` at the limb of the Planet and a little beyond it, where a coarse Voxel
/// that sticks out of the ball of the finest Voxels is what a ray meets.
fn limb_rays(camera: [f64; 3], count: usize, rng: &mut Lcg) -> Vec<Ray> {
    let len = camera.iter().map(|c| c * c).sum::<f64>().sqrt();
    let to_camera = camera.map(|c| c / len);
    let mut rays = Vec::new();
    while rays.len() < count {
        // A direction at right angles to the one the camera is in, give or take a few degrees.
        let r = [rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5].map(f64::from);
        let along: f64 = (0..3).map(|a| r[a] * to_camera[a]).sum();
        let side: [f64; 3] = std::array::from_fn(|a| r[a] - to_camera[a] * along);
        let side_len = side.iter().map(|c| c * c).sum::<f64>().sqrt();
        if side_len < 1e-3 {
            continue;
        }
        let tilt = f64::from(rng.next() - 0.5) * 0.12;
        let radius = f64::from(1300.0 + 500.0 * rng.next());
        let at: [f64; 3] =
            std::array::from_fn(|a| (side[a] / side_len + to_camera[a] * tilt) * radius);
        rays.push(ray(camera, at));
    }
    rays
}

/// Cameras far out and near, so that the limb is met at every distance.
const FAR_CAMERAS: [[f64; 3]; 6] = [
    [0.0, 9000.0, 4000.0],
    [20000.0, 5000.0, -3000.0],
    [-30000.0, 2000.0, 12000.0],
    [3000.0, 1700.0, 2500.0],
    [0.0, 3600.0, 3000.0],
    [1000.0, -9000.0, 500.0],
];

/// The ball does not cut into Voxels of the coarse levels, which are drawn whole and stick out of
/// it by their diagonal, not even at the limb. The clip allows for the coarsest level drawn.
#[test]
fn clipping_to_the_bounding_ball_does_not_cut_into_coarse_voxels_at_the_limb() {
    let outermost = f64::from(planet_gen::MAX_RADIUS);
    let mut rng = Lcg(808);
    let (mut rays_in_all, mut sticking_out, mut off_by_an_edge) = (0, 0, 0);
    for camera in FAR_CAMERAS {
        let plain = unclipped(rings_at(camera));
        let clipped = rings_at(camera);
        let (cell, origin) = render::camera::split(camera);
        let mut rays = limb_rays(camera, 4000, &mut rng);
        rays.push(Ray {
            cell,
            origin,
            dir: [0.053_175_47, -0.962_329_6, -0.266_634_64],
        });
        for ray in rays {
            rays_in_all += 1;
            let (a, b) = (trace(&plain, &ray), trace(&clipped, &ray));
            let (Some(a), Some(b)) = (a, b) else {
                assert!(
                    a.is_none() && b.is_none(),
                    "a hit and a miss: {a:?} {b:?} {ray:?}"
                );
                continue;
            };
            let at = a.voxel.map(f64::from);
            sticking_out += usize::from(at.iter().map(|c| c * c).sum::<f64>().sqrt() > outermost);
            let key = |h: &Hit| (h.voxel, h.normal, h.material);
            if key(&a) != key(&b) {
                // A ray along the very edge of a Voxel is in one or the next, by a last bit of
                // `t`: the same distance, a neighbouring Voxel.
                let near = (0..3).all(|c| (a.voxel[c] - b.voxel[c]).abs() <= 64);
                assert!(
                    near && (a.t - b.t).abs() <= 1e-5 * (1.0 + a.t),
                    "{a:?} against {b:?}, {ray:?}"
                );
                off_by_an_edge += 1;
            }
        }
    }
    eprintln!(
        "{rays_in_all} rays, {sticking_out} hits beyond the ball, {off_by_an_edge} on an edge"
    );
    // The bands keep a Voxel under a pixel across, so only the coarse levels far out stick out
    // of the ball by much; there must be some, or the test proves nothing.
    assert!(sticking_out > 0, "no hits beyond the ball");
    assert!(
        off_by_an_edge * 500 < rays_in_all,
        "{off_by_an_edge} of {rays_in_all}"
    );
}

#[test]
fn the_gpu_and_the_cpu_agree_on_rays_that_are_clipped_to_the_bounding_ball() {
    let mut rng = Lcg(404);
    for camera in CAMERAS {
        let scene = rings_at(camera);
        let mut rays = rays_at_the_planet(camera, 300, &mut rng);
        // Rays that pass the ball and rays that start in it, pointing every way.
        for _ in 0..200 {
            let d = [rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5];
            let len = d.iter().map(|c| c * c).sum::<f32>().sqrt();
            let (cell, origin) = render::camera::split(camera);
            rays.push(Ray {
                cell,
                origin,
                dir: d.map(|c| c / len),
            });
        }
        assert_agree(&format!("ball, camera {camera:?}"), &rays, &scene);
    }
    // And at the limb from far out, where coarse Voxels are what is hit.
    for camera in FAR_CAMERAS {
        let scene = rings_at(camera);
        let rays = limb_rays(camera, 400, &mut rng);
        assert_agree(&format!("ball limb, camera {camera:?}"), &rays, &scene);
    }
}

/// What following the bench's flight costs the CPU, frame by frame: `Scene::recentre` along
/// `planet-flyover-v1` (240 frames), after the first frame's scene was made, and then the jump from
/// the end of the path back to its start, which is what the bench's warm-up does. Run in release:
/// `cargo t --release -p render --test rings -- --ignored --nocapture flight_cpu_cost`.
#[test]
#[ignore = "a measurement, for a release build"]
fn flight_cpu_cost() {
    use std::time::Instant;
    let first = flyover_camera(0, 240).position;
    let built = Instant::now();
    let mut scene = rings_at(first);
    eprintln!("first scene: {:.1?}", built.elapsed());
    let (mut times, mut words, mut rebuilt) = (Vec::new(), 0usize, 0);
    for frame in 1..240 {
        let at = flyover_camera(frame, 240).position;
        let started = Instant::now();
        let update = scene.recentre(planet(), at).unwrap();
        times.push(started.elapsed().as_secs_f64() * 1e3);
        match update {
            Some(SceneUpdate::Rings(patch)) => words += patch.words(),
            Some(SceneUpdate::Rebuilt) => rebuilt += 1,
            _ => {}
        }
    }
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    eprintln!(
        "flight: mean {:.2} ms, median {:.2}, p95 {:.2}, max {:.2}; {} KiB patched in all, {rebuilt} rebuilds",
        times.iter().sum::<f64>() / times.len() as f64,
        sorted[sorted.len() / 2],
        sorted[sorted.len() * 95 / 100],
        sorted[sorted.len() - 1],
        words * 4 / 1024
    );
    // Jumps of several sizes from the first camera: patched where the camera goes, and a new
    // scene made there.
    let start = flyover_camera(0, 240).position;
    for jump in [100.0, 300.0, 1000.0, 2000.0, 4000.0, 8000.0] {
        let mut scene = rings_at(start);
        let to = [start[0] + jump, start[1], start[2]];
        let started = Instant::now();
        let update = scene.recentre(planet(), to).unwrap();
        let moved = started.elapsed();
        let started = Instant::now();
        let _ = rings_at(to);
        eprintln!(
            "jump of {jump:6}: {moved:8.1?} ({}), a new scene there {:.1?}",
            match update {
                Some(SceneUpdate::Rebuilt) => "rebuilt",
                Some(_) => "patched",
                None => "nothing",
            },
            started.elapsed()
        );
    }
}
