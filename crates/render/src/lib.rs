//! wgpu rendering on the Vulkan backend: compute-shader DDA through the Planet's bricks.
//!
//! `trace(ray) -> Hit` in `shaders/trace.wgsl` is the single entry point the renderer calls to
//! find what a ray hits. [`reference`] is its CPU twin, used by tests. A window ([`Renderer`])
//! and an offscreen target ([`Offscreen`]) share one [`tracer::Tracer`], so tests render with the
//! code that the window runs.
//!
//! Two ways of drawing share that `trace()`: primary visibility, and the reference path tracer
//! (`shaders/pathtrace.wgsl`), which accumulates samples while nothing changes and is the ground
//! truth for later lighting techniques. [`accumulation`] decides when it starts over.
//!
//! Primary visibility traces at an internal resolution, half of the output's per axis by default,
//! and writes a G-buffer beside the image; [`upscale`] builds the presented image from it and
//! from the frames before (TAAU). The reference path tracer needs no upscaling.
//!
//! Every pass is timed on the GPU where the device allows ([`Profiler`], [`timing`]), and
//! [`overlay`] draws those times, with the render settings, as an egui panel. [`DebugView`]s show
//! one input of the shading at a time.

pub mod accumulation;
pub mod camera;
mod debug_view;
mod gpu_materials;
pub mod lod;
mod offscreen;
pub mod overlay;
mod profiler;
pub mod reference;
#[cfg(test)]
mod sampling_tests;
pub mod scene;
mod shaders;
pub mod sun;
pub mod timing;
pub mod tracer;
pub mod upscale;
mod window;

use std::error::Error;

pub use camera::Camera;
pub use debug_view::DebugView;
pub use offscreen::{GpuInfo, Image, Offscreen};
pub use profiler::Profiler;
pub use scene::{RingsFull, Room, Scene, SceneUpdate};
pub use sun::Sun;
pub use timing::Timings;
pub use upscale::Resolution;
pub use window::{Mode, Renderer};

pub(crate) type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Vulkan is the only supported backend (ADR 0001).
fn create_instance() -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::VULKAN;
    wgpu::Instance::new(desc)
}

/// Requests a device on a Vulkan adapter, presentable to `surface` if one is given. With
/// `timers`, it has timestamp queries if the adapter does.
fn request_gpu(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
    timers: bool,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: surface,
        ..Default::default()
    }))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("bhayanak device"),
        // Pass timers, where the adapter has them; without them the timers read "n/a".
        required_features: if timers {
            adapter.features() & profiler::FEATURES
        } else {
            wgpu::Features::empty()
        },
        // The Planet's bricks are one big storage buffer, well over the portable 128 MiB limit.
        required_limits: adapter.limits(),
        ..Default::default()
    }))?;
    Ok((adapter, device, queue))
}

/// Submits `encoder` and returns the first `size` bytes of `staging`, a `MAP_READ` buffer that
/// the encoder copies into.
fn finish_and_read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    staging: &wgpu::Buffer,
    size: u64,
) -> Vec<u8> {
    queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    staging
        .slice(..size)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("receiver outlives the map");
        });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
    receiver
        .recv()
        .expect("map callback ran")
        .expect("staging buffer maps");
    let bytes = staging
        .slice(..size)
        .get_mapped_range()
        .expect("mapped range")
        .to_vec();
    staging.unmap();
    bytes
}

fn staging_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Copies `size` bytes of `buffer` (which needs `COPY_SRC`) to the CPU.
fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Vec<u8> {
    let staging = staging_buffer(device, size);
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    finish_and_read(device, queue, encoder, &staging, size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs a Vulkan driver; CI uses lavapipe (mesa-vulkan-drivers).
    #[test]
    fn vulkan_device_can_be_created_without_a_window() {
        let (adapter, _device, _queue) = request_gpu(&create_instance(), None, true).unwrap();
        assert_eq!(adapter.get_info().backend, wgpu::Backend::Vulkan);
    }
}
