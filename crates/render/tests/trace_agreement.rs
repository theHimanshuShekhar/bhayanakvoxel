//! The GPU `trace()` and the CPU reference must agree on hit, position, normal and Material.
//! Needs a Vulkan device: CI and local runs use lavapipe (see `.github/workflows/ci.yml`).

mod common;

use brick_store::{Brick, BrickCoord, BrickStore, Interior, Material, VoxelCoord};
use common::{Lcg, assert_agree, ray};
use planet_gen::generate_region;
use render::Offscreen;
use render::lod::Bands;
use render::reference::{Ray, trace};
use render::scene::{Bound, Core, Scene};

const STONE: Material = Material(7);

fn store_with(voxels: &[[i32; 3]]) -> BrickStore {
    let mut store = BrickStore::new();
    for &v in voxels {
        store.set(VoxelCoord(v), Some(STONE));
    }
    store
}

#[test]
fn a_hand_made_scene_agrees_on_edge_cases() {
    let mut store = store_with(&[
        [10, 0, 0],
        [-900, 35, 70],
        [1000, 3, -500],
        [63, 5, 0],
        [64, 0, 0],
        [3, 4, 5],
    ]);
    store.insert(BrickCoord([5, 8, 0]), Brick::solid(Material(300)));
    for (x, m) in [(1, 1), (2, 2), (3, 300), (4, 9)] {
        store.set(VoxelCoord([x, -20, 0]), Some(Material(m)));
    }
    let scene = Scene::new(&store, None);
    let at = |cell: [i32; 3], dir: [f32; 3]| {
        let len = dir.iter().map(|d| d * d).sum::<f32>().sqrt();
        Ray {
            cell,
            origin: [0.5; 3],
            dir: dir.map(|d| d / len),
        }
    };
    let rays = [
        // Axis-aligned, from all six sides.
        at([20, 0, 0], [-1.0, 0.0, 0.0]),
        at([0, 0, 0], [1.0, 0.0, 0.0]),
        at([10, 9, 0], [0.0, -1.0, 0.0]),
        at([10, -9, 0], [0.0, 1.0, 0.0]),
        at([10, 0, 9], [0.0, 0.0, -1.0]),
        at([10, 0, -9], [0.0, 0.0, 1.0]),
        // Starting inside a solid Voxel and inside a solid brick.
        at([3, 4, 5], [0.2, -1.0, 0.1]),
        at([42, 67, 4], [1.0, 0.0, 0.0]),
        // Misses: away, past, empty scene directions.
        at([20, 0, 0], [1.0, 0.0, 0.0]),
        at([0, 0, 0], [1.0, 0.1, 0.0]),
        at([0, 100, 0], [0.0, 1.0, 0.0]),
        // Across empty regions to far Voxels, either way.
        at([-1000, 0, 0], [100.0, 35.0, 70.0]),
        at([0, 0, 0], [1000.0, 3.0, -500.0]),
        at([1500, 3, -500], [-1.0, 0.0, 0.0]),
        // Region and brick boundaries.
        at([40, 0, 0], [1.0, 0.0, 0.0]),
        at([40, 5, 0], [1.0, 0.0, 0.0]),
        // Every palette entry, and one on a plane exactly.
        at([1, 3, 0], [0.0, -1.0, 0.0]),
        at([2, -10, 0], [0.0, -1.0, 0.0]),
        at([3, -10, 0], [0.0, -1.0, 0.0]),
        at([4, -10, 0], [0.0, -1.0, 0.0]),
        Ray {
            cell: [0, 1, 0],
            origin: [0.0, 0.0, 0.5],
            dir: [1.0, 0.0, 0.0],
        },
    ];
    let hits = assert_agree("hand-made", &rays, &scene);
    assert!(hits >= 12, "{hits}");
}

/// A brick whose first `n` Voxels (x fastest) are `n` distinct Materials.
fn brick_with_materials(n: usize, first: u16) -> Brick {
    let mut brick = Brick::AIR;
    for i in 0..n {
        let local = [(i % 8) as u8, (i / 8 % 8) as u8, (i / 64) as u8];
        brick.set(local, Some(Material(first + i as u16)));
    }
    brick
}

#[test]
fn every_palette_width_decodes_the_same_on_the_gpu_and_the_cpu() {
    // Palettes of 2, 3, 5, 17 and 300 Materials use 1, 2, 4, 8 and 16 bits per index.
    let sizes = [2, 3, 5, 17, 300];
    let mut store = BrickStore::new();
    for (i, n) in sizes.into_iter().enumerate() {
        let first = 1000 * (i as u16 + 1);
        store.insert(
            BrickCoord([i as i32 * 3, 0, 0]),
            brick_with_materials(n, first),
        );
    }
    let scene = Scene::new(&store, None);
    // One ray per solid Voxel, starting inside it, so that each Voxel's own Material is read.
    let mut rays = Vec::new();
    let mut expected = Vec::new();
    for (i, n) in sizes.into_iter().enumerate() {
        for v in 0..n {
            let voxel = [
                i as i32 * 24 + (v % 8) as i32,
                (v / 8 % 8) as i32,
                (v / 64) as i32,
            ];
            rays.push(Ray {
                cell: voxel,
                origin: [0.5; 3],
                dir: [0.0, 0.0, 1.0],
            });
            expected.push((voxel, 1000 * (i as u16 + 1) + v as u16));
        }
    }
    assert_eq!(rays.len(), 327);
    let gpu = Offscreen::new(8, 8, &scene).unwrap().trace_rays(&rays);
    for ((ray, want), got) in rays.iter().zip(&expected).zip(&gpu) {
        let cpu = trace(&scene, ray).expect("starts inside a solid Voxel");
        let got = got.expect("starts inside a solid Voxel");
        assert_eq!((cpu.voxel, cpu.material), *want, "cpu {ray:?}");
        assert_eq!((got.voxel, got.material), *want, "gpu {ray:?}");
    }
}

#[test]
fn a_scene_with_a_core_agrees_inside_on_the_surface_and_outside() {
    let core = Core {
        radius: 150,
        material: STONE,
    };
    let scene = Scene::new(&BrickStore::new(), Some(core));
    let mut rng = Lcg(11);
    let mut rays = Vec::new();
    for _ in 0..200 {
        let from = [
            (rng.next() * 500.0 - 250.0) as f64,
            (rng.next() * 500.0 - 250.0) as f64,
            (rng.next() * 500.0 - 250.0) as f64,
        ];
        let to = [
            (rng.next() * 200.0 - 100.0) as f64,
            (rng.next() * 200.0 - 100.0) as f64,
            (rng.next() * 200.0 - 100.0) as f64,
        ];
        rays.push(ray(from, to));
    }
    let hits = assert_agree("core", &rays, &scene);
    assert!(hits > 100, "{hits}");
}

#[test]
fn a_patch_of_the_test_planet_agrees_for_a_fixed_set_of_rays() {
    // The Planet's surface near +y, from inside the rock to above the highest peak.
    let store = generate_region(7, BrickCoord([-12, 178, -12]), BrickCoord([12, 193, 12]));
    let scene = Scene::new(&store, None);
    let mut rng = Lcg(2024);
    let mut rays = Vec::new();
    for _ in 0..1024 {
        let from = [
            f64::from(rng.next() * 160.0 - 80.0),
            1560.0 + f64::from(rng.next() * 100.0),
            f64::from(rng.next() * 160.0 - 80.0),
        ];
        let to = [
            f64::from(rng.next() * 160.0 - 80.0),
            1400.0 + f64::from(rng.next() * 120.0),
            f64::from(rng.next() * 160.0 - 80.0),
        ];
        rays.push(ray(from, to));
    }
    let hits = assert_agree("patch", &rays, &scene);
    assert!(hits > 800, "only {hits} of 1024 rays hit");
}

#[test]
fn rays_far_from_the_origin_agree_too() {
    let far = 1_900_000;
    let store = store_with(&[[far + 10, 7, -3], [far + 200, 7, -3], [far + 13, 8, -3]]);
    let scene = Scene::new(&store, None);
    let rays = [
        ray(
            [f64::from(far) + 0.25, 7.5, -2.5],
            [f64::from(far) + 50.0, 7.5, -2.5],
        ),
        ray(
            [f64::from(far) + 100.5, 7.5, -2.5],
            [f64::from(far) + 20.0, 7.5, -2.5],
        ),
        ray(
            [f64::from(far) + 12.5, 20.5, -2.5],
            [f64::from(far) + 13.5, 0.0, -2.5],
        ),
    ];
    let hits = assert_agree("far", &rays, &scene);
    assert_eq!(hits, 3);
}

/// A ball that has the Planet in it, centred on the origin.
fn bound(radius: i32) -> Bound {
    Bound {
        centre: [0; 3],
        radius,
    }
}

/// A rock ball of 600 Voxels' radius whose inside is read from the ball itself (see `Interior`),
/// and whose crust, the bricks over its edge, is held, as the Planet's is: so that every level of
/// its pyramid is exact, and the surface is at its coarsest at the far levels.
fn ball() -> BrickStore {
    let radius = 600;
    let mut store = BrickStore::new();
    store.set_source(Interior {
        radius,
        material: STONE,
    });
    let reach = radius / 8 + 1;
    for x in -reach..=reach {
        for y in -reach..=reach {
            for z in -reach..=reach {
                let coord = BrickCoord([x, y, z]);
                // The bricks the ball touches that it does not wholly contain.
                let near = render::lod::brick_distance([0.0; 3], 0, coord);
                if coord.within_ball(radius) || near > f64::from(radius) {
                    continue;
                }
                let mut brick = Brick::AIR;
                for l in 0..512usize {
                    let local = [(l & 7) as u8, (l >> 3 & 7) as u8, (l >> 6) as u8];
                    let at = coord.origin().0;
                    let d: i64 = (0..3)
                        .map(|a| i64::from(at[a] + i32::from(local[a])).pow(2))
                        .sum();
                    if d <= i64::from(radius).pow(2) {
                        brick.set(local, Some(STONE));
                    }
                }
                if !brick.is_air() {
                    store.insert(coord, brick);
                }
            }
        }
    }
    store.build_lod(6);
    store
}

#[test]
fn rings_agree_for_rays_that_end_at_every_level() {
    let store = ball();
    let mut rng = Lcg(77);
    // Distances along the rays that hit, by the nearest level: what the ring that holds the
    // Voxel it hits is, as far as a ring can be told from how far away it is.
    let mut by_distance = [0usize; 3];
    for (n, camera) in [
        [0.0, 610.0, 0.0],
        [0.0, 700.0, 200.0],
        [1040.0, 360.0, -180.0],
        [-2800.0, 0.0, 60.0],
    ]
    .into_iter()
    .enumerate()
    {
        let scene = Scene::rings(&store, camera, Bands::with_d0(300.0), bound(700)).unwrap();
        let rays: Vec<Ray> = (0..300)
            .map(|_| {
                // A point on the ball, or a little above or below it.
                let on = [rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5];
                let len = on.iter().map(|c| c * c).sum::<f32>().sqrt();
                let at = on.map(|c| f64::from(c / len) * f64::from(560.0 + 80.0 * rng.next()));
                ray(camera, at)
            })
            .collect();
        let hits = assert_agree(&format!("ball camera {n}"), &rays, &scene);
        assert!(hits > 20, "camera {n}: {hits} hits");
        for r in &rays {
            if let Some(hit) = trace(&scene, r) {
                by_distance[usize::from(hit.t > 300.0) + usize::from(hit.t > 1500.0)] += 1;
            }
        }
    }
    assert!(
        by_distance.iter().all(|&n| n > 10),
        "hits near, in the middle and far: {by_distance:?}"
    );
}

#[test]
fn rings_of_a_patch_of_the_test_planet_agree_with_mixed_bricks_at_every_level() {
    let mut store = generate_region(7, BrickCoord([-12, 178, -12]), BrickCoord([12, 193, 12]));
    store.build_lod(5);
    let mut rng = Lcg(31);
    let mut rays = Vec::new();
    let mut scenes = 0;
    for camera in [
        [0.0, 1565.0, 0.0],
        [90.0, 1900.0, -50.0],
        [0.0, 3000.0, 400.0],
    ] {
        let scene = Scene::rings(&store, camera, Bands::with_d0(64.0), bound(1600)).unwrap();
        rays.clear();
        for _ in 0..400 {
            let at = [
                f64::from(rng.next() * 160.0 - 80.0),
                1380.0 + f64::from(rng.next() * 140.0),
                f64::from(rng.next() * 160.0 - 80.0),
            ];
            rays.push(ray(camera, at));
        }
        // And rays that begin in a coarse ring and look back at where the camera is, as the
        // path tracer's bounces do.
        for _ in 0..100 {
            let from = [
                f64::from(rng.next() * 2000.0 - 1000.0),
                1500.0 + f64::from(rng.next() * 100.0),
                f64::from(rng.next() * 2000.0 - 1000.0),
            ];
            rays.push(ray(from, camera));
        }
        let hits = assert_agree("patch rings", &rays, &scene);
        assert!(hits > 200, "{hits} of {} rays hit", rays.len());
        scenes += 1;
    }
    assert_eq!(scenes, 3);
}

/// A floor under the camera and a short line of Voxels a little way off, on the edge of what
/// the coarser levels list.
fn floor_and_a_line() -> BrickStore {
    let mut store = BrickStore::new();
    for x in -64..64 {
        for z in -8..8 {
            store.insert(BrickCoord([x, -1, z]), Brick::solid(STONE));
        }
    }
    for (x, y) in [(512, 126), (513, 126), (512, 127), (513, 127)] {
        store.set(VoxelCoord([x, y, 0]), Some(STONE));
    }
    store.build_lod(2);
    store
}

#[test]
fn rays_that_start_in_the_hole_of_a_level_and_leave_it_agree_with_the_reference() {
    let store = floor_and_a_line();
    let camera = [0.5, 50.5, 0.5];
    // Level 1 starts 64 Voxels from the camera and level 2 128: the camera is in the holes of both.
    let scene = Scene::rings(&store, camera, Bands::with_d0(32.0), bound(2000)).unwrap();
    let rays: Vec<Ray> = (0..80)
        .map(|i| {
            let slope = -0.5 + 0.02 * f64::from(i);
            ray(camera, [camera[0] + 1.0, camera[1] + slope, camera[2]])
        })
        .collect();
    let hits = assert_agree("holes", &rays, &scene);
    assert!(hits > 5, "{hits}");
}
