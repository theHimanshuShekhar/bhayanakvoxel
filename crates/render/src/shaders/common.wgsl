// What the primary and reference kernels share: their bindings, what Materials look like, the
// sky, and how a linear colour is written to the image. Used after trace.wgsl.

struct Camera {
    // The Voxel the camera is in, then its position inside that Voxel. The CPU splits the f64
    // position, so these stay small and exact wherever the camera is.
    cell: vec4<i32>,
    origin: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    // Tangent of half the vertical field of view, then three unused.
    params: vec4<f32>,
}

struct Frame {
    // The unit vector towards the sun, then the cosine of the sun's angular radius.
    sun: vec4<f32>,
    // The index of this sample among those accumulated (0 starts a new image), the seed of the
    // random numbers, the debug view (0 is the lit image), then one unused.
    params: vec4<u32>,
    // Where in its pixel primary visibility traces its ray, (0, 0) to (1, 1), then two unused.
    jitter: vec4<f32>,
}

@group(1) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(2) var<uniform> frame: Frame;

// The Material table (materials.toml), one record per id. Four vec4s each; keep this in step
// with gpu_materials.rs, which documents them:
//   0: base colour, roughness
//   1: noise scale (Voxels), hue jitter (radians), value jitter, metalness
//   2: emitted light (colour times intensity)
//   3: transparency, IOR, 1 if the table defines the id, unused
@group(1) @binding(4) var<storage, read> materials: array<vec4<f32>>;

const RECORD_VEC4S = 4u;
const UNKNOWN = vec3<f32>(1.0, 0.0, 1.0);

// Whether the table defines `material`. An id past its end, or in a gap in it, is unknown.
fn is_known(material: u32) -> bool {
    let at = material * RECORD_VEC4S;
    return at + 3u < arrayLength(&materials) && materials[at + 3u].z != 0.0;
}

// The Material's base colour: magenta if it is unknown.
fn albedo(material: u32) -> vec3<f32> {
    if !is_known(material) {
        return UNKNOWN;
    }
    return materials[material * RECORD_VEC4S].xyz;
}

// The light the Material gives off, which is none if it is unknown.
fn emission(material: u32) -> vec3<f32> {
    if !is_known(material) {
        return vec3<f32>(0.0);
    }
    return materials[material * RECORD_VEC4S + 2u].xyz;
}

fn hash_voxel(cell: vec3<i32>) -> u32 {
    var h = bitcast<u32>(cell.x) * 0x8da6b343u ^ bitcast<u32>(cell.y) * 0xd8163841u ^ bitcast<u32>(cell.z) * 0xcb1ab31fu;
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    return h ^ (h >> 12u);
}

// The Material's colour at `voxel`: its base colour with the hue and brightness turned by up to
// the Material's jitter, by an amount that depends only on the patch of `noise_scale` Voxels
// that `voxel` is in. There is no colour stored per Voxel: this is a function of the Voxel's
// position alone, and it does not change with distance.
fn varied_albedo(material: u32, voxel: vec3<i32>) -> vec3<f32> {
    let base = albedo(material);
    if !is_known(material) {
        return base;
    }
    let params = materials[material * RECORD_VEC4S + 1u];
    let h = hash_voxel(vec3<i32>(floor(vec3<f32>(voxel) / params.x)));
    // Two independent numbers in -1..1.
    let value = f32(h & 255u) / 255.0 * 2.0 - 1.0;
    let hue = f32((h >> 8u) & 255u) / 255.0 * 2.0 - 1.0;
    // Turn the colour about the grey axis (Rodrigues' rotation) for the hue.
    let angle = hue * params.y;
    let axis = vec3<f32>(0.57735027);
    let turned = base * cos(angle) + cross(axis, base) * sin(angle) + axis * dot(axis, base) * (1.0 - cos(angle));
    return max(turned, vec3<f32>(0.0)) * (1.0 + value * params.z);
}

// A simple gradient from the horizon to the zenith, until the atmosphere replaces it. It uses
// the world's y axis as up, which is right near the +y pole and nowhere else.
fn sky(dir: vec3<f32>) -> vec3<f32> {
    return mix(vec3<f32>(0.75, 0.82, 0.92), vec3<f32>(0.25, 0.45, 0.85), clamp(dir.y, 0.0, 1.0));
}

// The ray from the camera through `pixel` of an image `size` pixels across. `offset` says where
// in the pixel, from (0, 0) at its top left to (1, 1); (0.5, 0.5) is the centre.
fn camera_ray(pixel: vec2<u32>, size: vec2<u32>, offset: vec2<f32>) -> Ray {
    let aspect = f32(size.x) / f32(size.y);
    let ndc = (vec2<f32>(pixel) + offset) / vec2<f32>(size) * 2.0 - 1.0;
    let tan_half = camera.params.x;
    var ray: Ray;
    ray.cell = camera.cell.xyz;
    ray.origin = camera.origin.xyz;
    ray.dir = normalize(camera.forward.xyz + camera.right.xyz * (ndc.x * aspect * tan_half) - camera.up.xyz * (ndc.y * tan_half));
    return ray;
}

// Linear colour to the image's sRGB-encoded values, which a plain unorm texture then clamps to
// 0..1.
fn encode(linear: vec3<f32>) -> vec3<f32> {
    return pow(linear, vec3<f32>(1.0 / 2.2));
}

// What the debug views show, as the values written to the image (already encoded). `view` is
// never 0 here: that is the lit image, which the kernels draw themselves. The colours of misses
// are black. Keep the numbers in step with DebugView in debug_view.rs.
const VIEW_NORMALS = 1u;
const VIEW_ALBEDO = 2u;
const VIEW_DEPTH = 3u;
const VIEW_SAMPLE_COUNT = 4u;
// Drawn by the upscaler (upscale.wgsl) in primary mode; in the reference image the history is its
// samples.
const VIEW_HISTORY_LENGTH = 5u;
// Voxels at which the depth view is half as bright as at the camera.
const DEPTH_HALF = 64.0;
// The sample count at which the sample count view reaches its brightest colour.
const SAMPLES_BRIGHTEST = 1024.0;

fn debug_colour(view: u32, hit: Hit, sample_index: u32) -> vec3<f32> {
    if view == VIEW_SAMPLE_COUNT || view == VIEW_HISTORY_LENGTH {
        let scale = clamp(log2(f32(sample_index + 1u)) / log2(SAMPLES_BRIGHTEST), 0.0, 1.0);
        return mix(vec3<f32>(0.10, 0.0, 0.30), vec3<f32>(1.0, 0.90, 0.20), scale);
    }
    if !hit.hit {
        return vec3<f32>(0.0);
    }
    switch view {
        case VIEW_NORMALS: { return hit.normal * 0.5 + 0.5; }
        case VIEW_ALBEDO: { return encode(albedo(hit.material)); }
        case VIEW_DEPTH: { return vec3<f32>(DEPTH_HALF / (DEPTH_HALF + hit.t)); }
        default: { return vec3<f32>(1.0, 0.0, 1.0); }
    }
}
