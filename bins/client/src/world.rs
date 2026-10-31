//! The client's view of the Planet: the bricks it predicts on, the packed copy that gets drawn,
//! and the loopback to the server. All Edits come through here, and reach the Voxels only via
//! [`sim::Predictor`], which calls [`sim::apply`].

use brick_store::VoxelCoord;
use net::{ClientEnd, Disconnected, EditOp, Leash, MoveInput, TryRecvError};
use render::reference::{Hit, Ray, trace};
use render::{Camera, RingsFull, Scene, SceneUpdate, camera};
use sim::Predictor;

/// Radius of the sphere a click digs: 1 m at 10 cm Voxels.
const DIG_RADIUS: u16 = 10;

/// Why the Planet cannot be kept going.
#[derive(Debug, PartialEq, Eq)]
pub enum Stopped {
    /// The server is gone.
    Disconnected,
    /// The Planet no longer fits the rings' buffers; the scene is not to be used again.
    RingsFull,
}

impl std::fmt::Display for Stopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => write!(f, "the server stopped"),
            Self::RingsFull => write!(f, "{RingsFull}"),
        }
    }
}

impl std::error::Error for Stopped {}

impl From<Disconnected> for Stopped {
    fn from(_: Disconnected) -> Self {
        Self::Disconnected
    }
}

impl From<RingsFull> for Stopped {
    fn from(_: RingsFull) -> Self {
        Self::RingsFull
    }
}

pub struct World {
    scene: Scene,
    predictor: Predictor,
    link: ClientEnd,
}

/// The first solid Voxel along the camera's view through the middle of the screen.
fn centre_hit(scene: &Scene, camera: &Camera) -> Option<Hit> {
    let (cell, origin) = camera::split(camera.position);
    trace(
        scene,
        &Ray {
            cell,
            origin,
            dir: camera.forward().map(|c| c as f32),
        },
    )
}

impl World {
    /// `scene` must be the packing of the predictor's bricks.
    pub fn new(scene: Scene, predictor: Predictor, link: ClientEnd) -> Self {
        Self {
            scene,
            predictor,
            link,
        }
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Puts the character on the Planet at `position` (Voxels), and tells the server. It falls
    /// from there, and cannot leave `leash`.
    pub fn spawn(&mut self, position: [f64; 3], leash: Leash) -> Result<(), Stopped> {
        self.link
            .send(self.predictor.spawn(position, Some(leash)))?;
        Ok(())
    }

    /// Runs one tick of the character on `input` here and on the server. Does nothing before
    /// the character has spawned.
    pub fn walk(&mut self, input: MoveInput) -> Result<(), Stopped> {
        if let Some(request) = self.predictor.walk(input) {
            self.link.send(request)?;
        }
        Ok(())
    }

    /// Where the character's eyes are, if it has spawned.
    pub fn eye(&self) -> Option<[f64; 3]> {
        self.predictor.character().map(sim::Character::eye)
    }

    /// Moves the rings of the Planet to centre on `camera`, a position in Voxels. Returns how
    /// the scene changed, if it did.
    pub fn recentre(&mut self, camera: [f64; 3]) -> Result<Option<SceneUpdate>, RingsFull> {
        self.scene.recentre(self.predictor.store(), camera)
    }

    /// Selects the bricks again for other LOD `bands` (a new bias). Returns how the scene
    /// changed, if it did.
    pub fn set_bands(
        &mut self,
        bands: render::lod::Bands,
    ) -> Result<Option<SceneUpdate>, RingsFull> {
        self.scene.set_bands(self.predictor.store(), bands)
    }

    /// Digs a sphere where the camera looks: shows it at once, and asks the server for it.
    /// Returns how the scene changed, if there was anything to dig.
    pub fn dig(&mut self, camera: &Camera) -> Result<Option<SceneUpdate>, Stopped> {
        let Some(hit) = centre_hit(&self.scene, camera) else {
            return Ok(None);
        };
        let op = EditOp::dig_sphere(VoxelCoord(hit.voxel), DIG_RADIUS);
        let (request, changed) = self.predictor.edit(op);
        self.link.send(request)?;
        Ok(self.refresh(&changed)?)
    }

    /// Takes in what the server has sent. Returns how the scene changed, if it did.
    pub fn poll(&mut self) -> Result<Option<SceneUpdate>, Stopped> {
        let mut changed = Vec::new();
        loop {
            match self.link.try_recv() {
                Ok(message) => changed.extend(self.predictor.receive(message)),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Err(Disconnected.into()),
            }
        }
        Ok(self.refresh(&changed)?)
    }

    /// Packs the bricks that changed.
    fn refresh(
        &mut self,
        changed: &[brick_store::BrickCoord],
    ) -> Result<Option<SceneUpdate>, RingsFull> {
        if changed.is_empty() {
            return Ok(None);
        }
        self.scene.update(self.predictor.store(), changed).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brick_store::{Brick, BrickCoord, BrickStore, Material};
    use net::{ClientMessage, RejectReason, ServerEnd, ServerMessage, loopback};

    const STONE: Material = Material(7);

    /// A wall of stone, well outside the Planet: Voxels x 4000..4032, y 0..32, z -32..0, in bricks of their own.
    fn wall() -> BrickStore {
        let mut store = BrickStore::new();
        for x in 500..504 {
            for y in 0..4 {
                for z in -4..0 {
                    store.insert(BrickCoord([x, y, z]), Brick::solid(STONE));
                }
            }
        }
        store
    }

    /// Looking at the wall's middle from in front of it.
    fn camera() -> Camera {
        Camera::looking_at([4016.5, 16.5, 20.5], [4016.5, 16.5, 0.0])
    }

    fn world() -> (World, ServerEnd) {
        let store = wall();
        let scene = Scene::new(&store, None);
        let (client, server) = loopback();
        (World::new(scene, Predictor::new(store), client), server)
    }

    #[test]
    fn a_click_digs_at_the_voxel_in_the_middle_of_the_view_and_asks_the_server() {
        let (mut world, server) = world();
        let update = world.dig(&camera()).unwrap();
        assert!(
            matches!(update, Some(SceneUpdate::Patched { .. })),
            "{update:?}"
        );
        // The wall's front face is z = -1; the view's middle is Voxel (4016, 16, -1).
        assert_eq!(
            server.try_recv(),
            Ok(ClientMessage::Edit {
                client_seq: 0,
                op: EditOp::dig_sphere(VoxelCoord([4016, 16, -1]), DIG_RADIUS)
            })
        );
        // The hole is drawn at once: the same ray now reaches deeper than the front face.
        let deeper = centre_hit(world.scene(), &camera()).unwrap();
        assert!(deeper.voxel[2] < -1, "{deeper:?}");
    }

    #[test]
    fn a_dig_in_a_scene_of_rings_is_drawn_at_once_and_the_rings_follow_the_camera() {
        let mut store = wall();
        store.build_lod(4);
        let bound = render::scene::Bound {
            centre: [0; 3],
            radius: 4200,
        };
        let scene = Scene::rings(
            &store,
            camera().position,
            render::lod::Bands::with_d0(64.0),
            bound,
        )
        .unwrap();
        let (client, _server) = loopback();
        let mut world = World::new(scene, Predictor::new(store), client);

        let update = world.dig(&camera()).unwrap();
        assert!(matches!(update, Some(SceneUpdate::Rings(_))), "{update:?}");
        let deeper = centre_hit(world.scene(), &camera()).unwrap();
        assert!(deeper.voxel[2] < -1, "{deeper:?}");

        // Where the camera is, the bands stay; a little way off, the bricks that change level are
        // patched; and a jump of more than a band's width makes the scene again.
        assert_eq!(world.recentre(camera().position), Ok(None));
        let [x, y, z] = camera().position;
        assert!(matches!(
            world.recentre([x + 60.0, y, z]),
            Ok(None | Some(SceneUpdate::Rings(_)))
        ));
        assert!(matches!(
            world.recentre([x + 900.0, y, z]),
            Ok(Some(SceneUpdate::Rebuilt))
        ));
    }

    #[test]
    fn the_character_lands_on_the_wall_and_the_server_is_sent_its_spawn_and_every_tick() {
        let (mut world, server) = world();
        assert_eq!(world.eye(), None);
        world.walk(MoveInput::default()).unwrap();
        assert_eq!(
            server.try_recv(),
            Err(TryRecvError::Empty),
            "not spawned yet"
        );

        let leash = Leash {
            axis: [1.0, 0.0, 0.0],
            // Wide enough for the whole wall: it is 16 Voxels off the axis.
            radius: 40.0,
        };
        // Above the middle of the wall's +x face (x = 4032).
        world.spawn([4036.0, 16.0, -16.0], leash).unwrap();
        for _ in 0..120 {
            world.walk(MoveInput::default()).unwrap();
        }
        let eye = world.eye().unwrap();
        assert!((4032.0 + 15.0..4032.0 + 17.0).contains(&eye[0]), "{eye:?}");

        assert_eq!(
            server.try_recv(),
            Ok(ClientMessage::Spawn {
                position: [4036.0, 16.0, -16.0],
                leash: Some(leash)
            })
        );
        for seq in 0..120 {
            let Ok(ClientMessage::Input { seq: got, .. }) = server.try_recv() else {
                panic!("input {seq} missing");
            };
            assert_eq!(got, seq);
        }
    }

    #[test]
    fn a_click_on_the_sky_does_nothing() {
        let (mut world, server) = world();
        let up = Camera::looking_at([4016.5, 16.5, 20.5], [4016.5, 100.0, 20.5]);
        assert!(world.dig(&up).unwrap().is_none());
        assert_eq!(server.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn an_ack_keeps_the_hole_and_a_reject_fills_it_again() {
        let (mut world, server) = world();
        let hit = |world: &World| centre_hit(world.scene(), &camera()).unwrap().voxel;
        let whole = hit(&world);
        world.dig(&camera()).unwrap();
        let first = hit(&world);
        world.dig(&camera()).unwrap();
        let second = hit(&world);
        assert!(whole != first && first != second);
        let Ok(ClientMessage::Edit { op, .. }) = server.try_recv() else {
            panic!("no Edit sent");
        };

        server
            .send(ServerMessage::Applied {
                global_seq: 1,
                op,
                ack: Some(0),
            })
            .unwrap();
        assert!(world.poll().unwrap().is_none(), "the prediction was right");
        assert_eq!(hit(&world), second);

        // The second dig was predicted on top of the first; rejecting it leaves the first.
        server
            .send(ServerMessage::Rejected {
                client_seq: 1,
                reason: RejectReason::BrushTooLarge,
            })
            .unwrap();
        assert!(world.poll().unwrap().is_some());
        assert_eq!(hit(&world), first);
    }

    #[test]
    fn a_server_that_is_gone_is_an_error() {
        let (mut world, server) = world();
        drop(server);
        assert_eq!(world.poll().err(), Some(Stopped::Disconnected));
        assert_eq!(world.dig(&camera()).err(), Some(Stopped::Disconnected));
    }
}
