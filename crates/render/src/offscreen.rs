//! Rendering without a window, into a texture that is read back: for tests and tools.

use crate::camera::Camera;
use crate::debug_view::DebugView;
use crate::profiler::Profiler;
use crate::reference::{Hit, Ray};
use crate::scene::{Scene, SceneUpdate};
use crate::sun::Sun;
use crate::timing::Timings;
use crate::tracer::{Plane, Target, Tracer};
use crate::upscale::Resolution;
use crate::{Result, create_instance, finish_and_read, request_gpu, staging_buffer};

/// Copies a `width` x `height` texture of `bytes` per texel to the CPU after the work in
/// `encoder`, tightly packed, rows from the top.
fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    (width, height): (u32, u32),
    bytes: u32,
) -> Vec<u8> {
    // Rows in a texture-to-buffer copy start at multiples of 256 bytes.
    let row = (width * bytes).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let size = u64::from(row) * u64::from(height);
    let staging = staging_buffer(device, size);
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let padded = finish_and_read(device, queue, encoder, &staging, size);
    let tight = (width * bytes) as usize;
    if tight == row as usize {
        return padded;
    }
    let mut rows = Vec::with_capacity(tight * height as usize);
    for r in padded.chunks(row as usize) {
        rows.extend_from_slice(&r[..tight]);
    }
    rows
}

/// A finished frame: tightly packed RGBA8, rows from the top.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// What primary visibility wrote beside the image, at the traced resolution, rows from the top.
#[derive(Clone, Debug, PartialEq)]
pub struct GBuffer {
    pub width: u32,
    pub height: u32,
    /// Distance along the ray to the first hit, in Voxels; negative where it hit nothing.
    pub depth: Vec<f32>,
    /// The first hit's face normal, a unit vector along an axis; zero where nothing was hit.
    pub normal: Vec<[f32; 3]>,
    /// The first hit's surface colour before it is lit, as RGBA8 like an [`Image`]; black where
    /// nothing was hit.
    pub albedo: Vec<u8>,
}

/// Which GPU and driver a device is on, for tools that report where they ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuInfo {
    pub name: String,
    /// `Discrete`, `Integrated`, `Virtual`, `Cpu` (software, such as lavapipe) or `Other`.
    pub device_type: String,
    pub driver: String,
    pub driver_info: String,
    pub backend: String,
}

/// The renderer with an image of its own instead of a window.
pub struct Offscreen {
    info: GpuInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    tracer: Tracer,
    profiler: Profiler,
    target: Target,
    sun: Sun,
    /// Whether `update_scene` sends its writes at once; see `set_flush_scene_updates`.
    flush_scene_updates: bool,
}

impl Offscreen {
    /// Traces and presents at the same size. Needs a Vulkan device; CI and local tests use
    /// lavapipe.
    pub fn new(width: u32, height: u32, scene: &Scene) -> Result<Self> {
        Self::with_resolution(Resolution::native(width, height), scene)
    }

    /// Traces at `resolution.internal` and presents at `resolution.output`.
    pub fn with_resolution(resolution: Resolution, scene: &Scene) -> Result<Self> {
        Self::create(resolution, scene, true)
    }

    /// Like [`Offscreen::new`], on a device without timestamp queries, as on a GPU that lacks
    /// them: passes are not timed. For testing that path wherever the driver has them.
    pub fn without_timers(width: u32, height: u32, scene: &Scene) -> Result<Self> {
        Self::create(Resolution::native(width, height), scene, false)
    }

    fn create(resolution: Resolution, scene: &Scene, timers: bool) -> Result<Self> {
        let (adapter, device, queue) = request_gpu(&create_instance(), None, timers)?;
        let info = adapter.get_info();
        let info = GpuInfo {
            name: info.name,
            device_type: format!("{:?}", info.device_type),
            driver: info.driver,
            driver_info: info.driver_info,
            backend: format!("{:?}", info.backend),
        };
        let tracer = Tracer::new(&device, scene)?;
        let profiler = Profiler::new(&device, &queue);
        let target = Target::new(&device, resolution);
        Ok(Self {
            info,
            device,
            queue,
            tracer,
            profiler,
            target,
            sun: Sun::default(),
            flush_scene_updates: true,
        })
    }

    /// Brings the GPU copy of the Planet up to date with `scene`, after `update` (the result of
    /// [`Scene::update`]) changed it. This starts the reference image over.
    pub fn update_scene(&mut self, scene: &Scene, update: &SceneUpdate) -> Result<()> {
        self.tracer
            .update_scene(&self.device, &self.queue, scene, update)?;
        // The writes are queued until the next submit: send them now, so that their cost is the
        // update's and not that of the frame drawn next.
        if self.flush_scene_updates {
            self.queue.submit(std::iter::empty());
        }
        Ok(())
    }

    /// Whether [`Offscreen::update_scene`] sends the writes to the GPU at once (the default), or
    /// leaves them to go with the next frame's commands. For finding out what the first costs.
    pub fn set_flush_scene_updates(&mut self, flush: bool) {
        self.flush_scene_updates = flush;
    }

    /// Changes the sun that both ways of rendering light the Planet with.
    pub fn set_sun(&mut self, sun: Sun) {
        self.sun = sun;
    }

    /// Chooses what is drawn: the lit image, or one of the debug views.
    pub fn set_debug_view(&mut self, view: DebugView) {
        self.tracer.set_debug_view(view);
    }

    /// The GPU time of each pass, as of the last frame rendered. Unlike in a window, a frame's
    /// times are in by the time it has been read back. Without timestamp queries there are
    /// none (see [`Offscreen::timers_supported`]).
    pub fn timings(&self) -> &Timings {
        self.profiler.timings()
    }

    /// Bytes of GPU memory that the Planet's bricks take.
    pub fn scene_bytes(&self) -> u64 {
        self.tracer.scene_bytes()
    }

    /// Bytes of the Planet written to the GPU so far, at the start and by each update.
    pub fn uploaded_bytes(&self) -> u64 {
        self.tracer.uploaded_bytes()
    }

    /// The GPU that this renders on.
    pub fn gpu_info(&self) -> &GpuInfo {
        &self.info
    }

    /// Whether the device can time passes.
    pub fn timers_supported(&self) -> bool {
        self.profiler.supported()
    }

    /// Keeps every pass time from now on, for a report over many frames.
    pub fn set_keep_history(&mut self, keep: bool) {
        self.profiler.set_keep_history(keep);
    }

    /// Renders primary visibility from `camera` at the traced resolution and reads it back: one
    /// ray through the centre of each pixel, with nothing carried over from earlier frames and no
    /// upscaling. This is also what [`Offscreen::gbuffer`] then holds.
    pub fn render(&mut self, camera: &Camera) -> Image {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.profiler.begin_frame();
        self.tracer.encode_primary(
            &self.device,
            &self.queue,
            &mut self.profiler,
            &mut encoder,
            camera,
            &self.sun,
            &self.target,
        );
        let size = self.target.resolution().internal;
        let texture = self.target.lit().texture.clone();
        self.read_image(encoder, &texture, size)
    }

    /// Renders one frame of what the window shows in primary mode, and reads back the image at
    /// the output resolution: primary visibility traced at the internal resolution, through a
    /// different point of each pixel every frame, upscaled together with the frames before. Call
    /// it every frame with the camera of that frame: a still camera converges. A new size or a
    /// debug view that is not the lit image or the history length starts the sequence over.
    pub fn render_upscaled(&mut self, camera: &Camera) -> Image {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.profiler.begin_frame();
        self.tracer.encode_upscaled(
            &self.device,
            &self.queue,
            &mut self.profiler,
            &mut encoder,
            camera,
            &self.sun,
            &self.target,
        );
        let size = self.target.resolution().output;
        let texture = self.target.image().texture.clone();
        self.read_image(encoder, &texture, size)
    }

    /// Like [`Offscreen::render_upscaled`], without reading the image back: the frame is drawn and
    /// its pass times are in when this returns. For a run of frames that is timed and whose
    /// pictures are not looked at (the bench), where reading back every output image (15 MB at
    /// 1440p) would be most of the CPU time of a frame and leave the GPU idle between frames.
    pub fn draw_upscaled(&mut self, camera: &Camera) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.profiler.begin_frame();
        self.tracer.encode_upscaled(
            &self.device,
            &self.queue,
            &mut self.profiler,
            &mut encoder,
            camera,
            &self.sun,
            &self.target,
        );
        self.profiler.resolve(&mut encoder);
        // The uniforms of the next frame are written over these.
        self.queue.submit([encoder.finish()]);
        self.profiler.collect(&self.device, true);
        // Without timers `collect` does not wait, and the frame is not over until it has.
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed");
    }

    /// The G-buffer of the last frame of primary visibility ([`Offscreen::render`] or
    /// [`Offscreen::render_upscaled`]).
    pub fn gbuffer(&mut self) -> GBuffer {
        let (width, height) = self.target.resolution().internal;
        let read = |plane: &Plane, bytes: u32| {
            let encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            read_texture(
                &self.device,
                &self.queue,
                encoder,
                &plane.texture,
                (width, height),
                bytes,
            )
        };
        let words = read(self.target.gbuffer(), 8);
        let mut gbuffer = GBuffer {
            width,
            height,
            depth: Vec::new(),
            normal: Vec::new(),
            albedo: Vec::new(),
        };
        for texel in words.as_chunks::<8>().0 {
            let [d0, d1, d2, d3, r, g, b, face] = *texel;
            gbuffer.depth.push(f32::from_le_bytes([d0, d1, d2, d3]));
            gbuffer.albedo.extend_from_slice(&[r, g, b, 255]);
            gbuffer.normal.push(match face {
                0 => [0.0; 3],
                code => {
                    let mut n = [0.0; 3];
                    n[usize::from(code - 1) / 2] = if (code - 1) % 2 == 1 { 1.0 } else { -1.0 };
                    n
                }
            });
        }
        gbuffer
    }

    /// Adds `samples` samples of the reference path tracer to the image of `camera` and reads
    /// back the mean so far. The image starts over if `camera` or the sun is not the one of the
    /// previous call. With zero samples it just reads the image again.
    pub fn render_reference(&mut self, camera: &Camera, samples: u32) -> Image {
        for _ in 0..samples {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            self.profiler.begin_frame();
            self.tracer.encode_reference(
                &self.device,
                &self.queue,
                &mut self.profiler,
                &mut encoder,
                camera,
                &self.sun,
                &self.target,
            );
            self.profiler.resolve(&mut encoder);
            // The uniforms of the next sample are written over these.
            self.queue.submit([encoder.finish()]);
            self.profiler.collect(&self.device, true);
        }
        let size = self.target.resolution().internal;
        let texture = self.target.lit().texture.clone();
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.read_image(encoder, &texture, size)
    }

    /// Reads back `texture`, an image `(width, height)` big, after the work recorded in `encoder`,
    /// which ends the frame the profiler is timing (if any).
    fn read_image(
        &mut self,
        mut encoder: wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        (width, height): (u32, u32),
    ) -> Image {
        self.profiler.resolve(&mut encoder);
        let rgba = read_texture(
            &self.device,
            &self.queue,
            encoder,
            texture,
            (width, height),
            4,
        );
        self.profiler.collect(&self.device, true);
        Image {
            width,
            height,
            rgba,
        }
    }

    /// Casts `rays` through the GPU `trace()` and returns the hits, in order.
    pub fn trace_rays(&self, rays: &[Ray]) -> Vec<Option<Hit>> {
        self.tracer.trace_rays(&self.device, &self.queue, rays)
    }
}
