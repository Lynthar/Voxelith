//! GLB → `procgen::VoxelPatch`: walk the scene graph, sample every
//! triangle onto the grid, then fill the interior by a 3-axis parity
//! scan with majority vote. The file is untrusted throughout.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};
use glam::{Mat4, Quat, Vec3};

use crate::core::Voxel;
use crate::procgen::VoxelPatch;

/// Voxelize a GLB into a `VoxelPatch` starting at `(0, 0, 0)`.
/// `resolution` is the voxel count along the mesh's longest axis; the
/// other two shrink in proportion.
///
/// # Errors
/// A resolution outside 4..=256, a malformed GLB, or a file carrying no
/// triangle primitives.
pub fn voxelize_glb(bytes: &[u8], resolution: u32) -> Result<VoxelPatch> {
    if !(4..=256).contains(&resolution) {
        bail!("Resolution must be in 4..=256, got {}", resolution);
    }

    // Parse structure first, decode pixels ourselves: `import_slice`
    // decodes every embedded image eagerly with nowhere to pass a
    // `Limits`, and a failed allocation aborts the editor.
    let gltf::Gltf { document, blob } = gltf::Gltf::from_slice(bytes).context("Parsing GLB")?;
    // `None` base path: any image or buffer referencing an external
    // file URI is refused rather than read off the local disk.
    let buffers = gltf::import_buffers(&document, None, blob).context("Reading GLB buffers")?;
    let textures = decode_base_color_textures(&document, &buffers);

    // Prefer the explicit default scene; fall back to the first scene;
    // if neither exists, walk all meshes directly (some exporters
    // produce GLBs with no scene node — rare but seen in the wild).
    let mut triangles = Vec::new();
    let mut limits = WalkLimits::default();
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next());
    if let Some(scene) = scene {
        for node in scene.nodes() {
            walk_node(
                &node,
                Mat4::IDENTITY,
                &buffers,
                &textures,
                &mut triangles,
                0,
                &mut limits,
            );
        }
    } else {
        // Same budget as the scene walk: this fallback used to skip the
        // limits entirely, making "export with no scene node" the way a
        // hostile file bypassed the triangle cap.
        for mesh in document.meshes() {
            extract_from_mesh(
                &mesh,
                Mat4::IDENTITY,
                &buffers,
                &textures,
                &mut triangles,
                &mut limits,
            );
        }
    }

    if triangles.is_empty() {
        bail!("GLB has no triangle primitives");
    }

    let (aabb_min, aabb_max) = compute_aabb(&triangles);
    let extent = aabb_max - aabb_min;
    // Guard against degenerate / zero-extent meshes — a single-point
    // mesh would produce voxel_size = 0 and rasterize_triangles would
    // generate NaN cell coords.
    let max_extent = extent.max_element().max(1e-6);
    let voxel_size = max_extent / resolution as f32;

    let accumulator = rasterize_triangles(&triangles, aabb_min, aabb_max, voxel_size, resolution);
    let surface = finalize_surface(accumulator);
    let filled = fill_interior(&surface);
    let mut patch = build_patch(filled);
    if limits.exhausted {
        // The log line says which limit; this is the half the person who
        // picked the file sees. A truncated import that looks complete
        // is how a model ships with half its geometry.
        patch.notes.push(
            "only part of this file was voxelized — see the log for which limit it hit".into(),
        );
    }
    Ok(patch)
}

// -------------------- glTF extraction --------------------

struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    c0: [u8; 4],
    c1: [u8; 4],
    c2: [u8; 4],
}

struct DecodedImage {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
}

/// Per-axis ceiling on a decoded texture. 8192² is well past anything a
/// voxel import needs and still bounds one decode at 256 MiB of RGBA.
const MAX_TEXTURE_DIM: u32 = 8192;
/// Ceiling on a single decode's own allocations.
const MAX_TEXTURE_ALLOC: u64 = 256 * 1024 * 1024;
/// Ceiling on all base-color textures we keep resident at once.
const MAX_TEXTURE_BUDGET: u64 = 512 * 1024 * 1024;

/// Decode the base-color textures the materials actually sample, keyed
/// by image index. One that fails to decode is logged and skipped — the
/// model still voxelizes, from the factor or vertex colors instead.
fn decode_base_color_textures(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> HashMap<usize, DecodedImage> {
    let wanted: HashSet<usize> = document
        .materials()
        .filter_map(|m| m.pbr_metallic_roughness().base_color_texture())
        .map(|info| info.texture().source().index())
        .collect();

    let mut out = HashMap::new();
    let mut budget = MAX_TEXTURE_BUDGET;
    for image in document.images() {
        let index = image.index();
        if !wanted.contains(&index) {
            continue;
        }
        let gltf::image::Source::View { view, .. } = image.source() else {
            // An external file URI: `import_buffers(None)` refuses those
            // too, and we're not about to read local files on behalf of
            // a downloaded asset.
            log::warn!("Skipping texture {index}: external image URIs aren't read");
            continue;
        };
        let Some(buffer) = buffers.get(view.buffer().index()) else {
            log::warn!("Skipping texture {index}: buffer out of range");
            continue;
        };
        let Some(encoded) = buffer.get(view.offset()..view.offset() + view.length()) else {
            log::warn!("Skipping texture {index}: view out of range");
            continue;
        };
        match decode_texture(encoded, &mut budget) {
            Ok(decoded) => {
                out.insert(index, decoded);
            }
            Err(e) => log::warn!("Skipping texture {index}: {e:#}"),
        }
    }
    out
}

/// Decode one encoded image under explicit limits, charging its size
/// against the shared `budget`.
fn decode_texture(encoded: &[u8], budget: &mut u64) -> Result<DecodedImage> {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_TEXTURE_DIM);
    limits.max_image_height = Some(MAX_TEXTURE_DIM);
    limits.max_alloc = Some(MAX_TEXTURE_ALLOC);

    let mut reader = image::ImageReader::new(std::io::Cursor::new(encoded))
        .with_guessed_format()
        .context("Sniffing texture format")?;
    reader.limits(limits);
    let decoded = reader.decode().context("Decoding texture")?;

    // Charge the RGBA conversion before performing it — that's the
    // allocation the image crate's own limits don't cover.
    let (width, height) = (decoded.width(), decoded.height());
    let cost = u64::from(width) * u64::from(height) * 4;
    if cost > *budget {
        bail!("would exceed the {MAX_TEXTURE_BUDGET} byte texture budget");
    }
    *budget -= cost;

    let rgba = decoded.to_rgba8();
    Ok(DecodedImage {
        rgba: rgba.into_raw(),
        width,
        height,
    })
}

/// Depth cap for the node walk. Real hierarchies are a handful of
/// levels; this exists so a hand-built chain of nodes can't recurse the
/// main thread into a stack overflow.
const MAX_NODE_DEPTH: usize = 256;

/// Triangles one file may hand over. Nothing else bounds this: a mesh
/// can be referenced from any number of nodes, so a few hundred bytes
/// of JSON can ask for the same geometry a thousand times.
const MAX_TRIANGLES: usize = 2_000_000;

/// Budget tracking for [`walk_node`] — the same shape and reason as
/// `.vox`'s `FlattenLimits`: refuse to be talked into a stack overflow
/// or an out-of-memory abort by a small file.
#[derive(Default)]
struct WalkLimits {
    /// Nodes already walked. glTF requires disjoint strict trees, so a
    /// second arrival means a malformed file — and a node listing
    /// itself as its child would overflow the stack.
    visited: std::collections::HashSet<usize>,
    exhausted: bool,
}

impl WalkLimits {
    /// Abandon the rest of the walk, keeping what was extracted so
    /// far — a partial model the user can see beats a hang.
    fn stop(&mut self, why: &str) {
        if !self.exhausted {
            log::warn!("glTF: {why}; import stopped early");
            self.exhausted = true;
        }
    }
}

fn walk_node(
    node: &gltf::Node,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    textures: &HashMap<usize, DecodedImage>,
    triangles: &mut Vec<Triangle>,
    depth: usize,
    limits: &mut WalkLimits,
) {
    if limits.exhausted {
        return;
    }
    if depth > MAX_NODE_DEPTH {
        limits.stop("node hierarchy is nested too deeply");
        return;
    }
    if !limits.visited.insert(node.index()) {
        // Prune this branch rather than the whole import: one
        // malformed link shouldn't cost the geometry that was fine.
        log::warn!(
            "glTF: node {} is reachable more than once; pruning (the spec requires a tree)",
            node.index()
        );
        return;
    }

    let local = mat4_from_transform(node.transform());
    let transform = parent_transform * local;

    if let Some(mesh) = node.mesh() {
        // The budget lives inside `extract_from_mesh` now — checked per
        // primitive against the accessor's declared counts, before
        // anything is collected.
        extract_from_mesh(&mesh, transform, buffers, textures, triangles, limits);
        if limits.exhausted {
            return;
        }
    }

    for child in node.children() {
        walk_node(
            &child,
            transform,
            buffers,
            textures,
            triangles,
            depth + 1,
            limits,
        );
    }
}

fn mat4_from_transform(t: gltf::scene::Transform) -> Mat4 {
    match t {
        gltf::scene::Transform::Matrix { matrix } => Mat4::from_cols_array_2d(&matrix),
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => Mat4::from_scale_rotation_translation(
            Vec3::from_array(scale),
            Quat::from_array(rotation),
            Vec3::from_array(translation),
        ),
    }
}

fn extract_from_mesh(
    mesh: &gltf::Mesh,
    transform: Mat4,
    buffers: &[gltf::buffer::Data],
    textures: &HashMap<usize, DecodedImage>,
    triangles: &mut Vec<Triangle>,
    limits: &mut WalkLimits,
) {
    for primitive in mesh.primitives() {
        if limits.exhausted {
            return;
        }
        if primitive.mode() != gltf::mesh::Mode::Triangles {
            // Skip lines / points / triangle strips. Only triangles
            // have an interior to fill, and an imported file may well
            // contain the others — no-op the primitive, don't panic.
            continue;
        }

        // Budget the primitive before collecting it: the accessor
        // declares its counts up front, so an oversized one is refused
        // for two lookups instead of after a huge allocation.
        let vertex_count = primitive
            .get(&gltf::Semantic::Positions)
            .map(|a| a.count())
            .unwrap_or(0);
        let declared_triangles = primitive
            .indices()
            .map(|a| a.count())
            .unwrap_or(vertex_count)
            / 3;
        if triangles.len() + declared_triangles > MAX_TRIANGLES {
            limits.stop("mesh has more triangles than an import can hold");
            return;
        }

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

        let positions: Vec<[f32; 3]> = match reader.read_positions() {
            Some(p) => p.collect(),
            None => continue,
        };

        // Vertex colors are optional and, per the spec, an additional
        // linear multiplier — they multiply with the texture and factor
        // rather than replacing them.
        let vertex_colors: Option<Vec<[f32; 4]>> =
            reader.read_colors(0).map(|c| c.into_rgba_f32().collect());

        let tex_coords: Option<Vec<[f32; 2]>> =
            reader.read_tex_coords(0).map(|tc| tc.into_f32().collect());

        let material = primitive.material();
        let pbr = material.pbr_metallic_roughness();
        let base_factor = pbr.base_color_factor();
        let base_texture: Option<&DecodedImage> = pbr
            .base_color_texture()
            .and_then(|info| textures.get(&info.texture().source().index()));

        let color_at_vertex = |i: usize| -> [u8; 4] {
            let vertex = vertex_colors
                .as_ref()
                .map(|colors| colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]));
            let tex_sample = match (base_texture, tex_coords.as_ref()) {
                (Some(tex), Some(uvs)) => {
                    let uv = uvs.get(i).copied().unwrap_or([0.5, 0.5]);
                    Some(sample_texture(tex, uv[0], uv[1]))
                }
                _ => None,
            };
            compose_base_color(vertex, tex_sample, base_factor)
        };

        let world_pos =
            |i: usize| -> Vec3 { transform.transform_point3(Vec3::from_array(positions[i])) };

        // glTF triangle primitives may be indexed or unindexed; treat
        // the unindexed case as identity indices to keep emission
        // logic uniform.
        let indices: Vec<u32> = reader
            .read_indices()
            .map(|i| i.into_u32().collect())
            .unwrap_or_else(|| (0..positions.len() as u32).collect());

        for chunk in indices.chunks_exact(3) {
            let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
            if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
                continue;
            }
            triangles.push(Triangle {
                v0: world_pos(i0),
                v1: world_pos(i1),
                v2: world_pos(i2),
                c0: color_at_vertex(i0),
                c1: color_at_vertex(i1),
                c2: color_at_vertex(i2),
            });
        }
    }
}

fn sample_texture(tex: &DecodedImage, u: f32, v: f32) -> [f32; 4] {
    // Approximate REPEAT wrap (glTF default). Negative inputs wrap to
    // their positive fractional part — `rem_euclid` would be cleaner
    // but `f32::rem_euclid` is fine.
    let u = u.rem_euclid(1.0);
    let v = v.rem_euclid(1.0);
    let x = ((u * tex.width as f32) as u32).min(tex.width.saturating_sub(1));
    // No flip: glTF puts UV origin (0,0) at the image's top-left and
    // grows v downward, the same direction `to_rgba8()` lays its rows
    // out, so v maps straight onto the row index.
    let y = ((v * tex.height as f32) as u32).min(tex.height.saturating_sub(1));
    let idx = ((y * tex.width + x) * 4) as usize;
    [
        tex.rgba[idx] as f32 / 255.0,
        tex.rgba[idx + 1] as f32 / 255.0,
        tex.rgba[idx + 2] as f32 / 255.0,
        tex.rgba[idx + 3] as f32 / 255.0,
    ]
}

fn pack_rgba(c: [f32; 4]) -> [u8; 4] {
    [
        (c[0] * 255.0).clamp(0.0, 255.0) as u8,
        (c[1] * 255.0).clamp(0.0, 255.0) as u8,
        (c[2] * 255.0).clamp(0.0, 255.0) as u8,
        (c[3] * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

/// glTF base-color composition for one vertex: `factor × texture ×
/// COLOR_0`, each absent source contributing 1.0. With no source at all
/// a neutral 200-gray is returned rather than pure white.
fn compose_base_color(
    vertex: Option<[f32; 4]>,
    tex_sample: Option<[f32; 4]>,
    factor: [f32; 4],
) -> [u8; 4] {
    if vertex.is_none() && tex_sample.is_none() && factor == [1.0, 1.0, 1.0, 1.0] {
        return [200, 200, 200, 255];
    }
    let mut c = factor;
    if let Some(vc) = vertex {
        c = [c[0] * vc[0], c[1] * vc[1], c[2] * vc[2], c[3] * vc[3]];
    }
    if let Some(s) = tex_sample {
        c = [c[0] * s[0], c[1] * s[1], c[2] * s[2], c[3] * s[3]];
    }
    pack_rgba(c)
}

fn compute_aabb(triangles: &[Triangle]) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for t in triangles {
        for v in [t.v0, t.v1, t.v2] {
            min = min.min(v);
            max = max.max(v);
        }
    }
    (min, max)
}

// -------------------- voxelization --------------------

/// `(r_sum, g_sum, b_sum, a_sum, count)` per cell. Caller divides at
/// the end to get the mean color per voxel.
type ColorAccum = [u32; 5];

fn rasterize_triangles(
    triangles: &[Triangle],
    origin: Vec3,
    far_corner: Vec3,
    voxel_size: f32,
    resolution: u32,
) -> HashMap<(i32, i32, i32), ColorAccum> {
    let mut grid: HashMap<(i32, i32, i32), ColorAccum> = HashMap::new();
    // The last cell index each axis has. Geometry on the box's far face
    // divides exactly, so `floor` would put those samples one layer
    // past the model: a cell spans `[p, p+1)` and the end face is its.
    let last_cell = |extent: f32| ((extent / voxel_size).ceil() as i32 - 1).max(0);
    let last = (
        last_cell(far_corner.x - origin.x),
        last_cell(far_corner.y - origin.y),
        last_cell(far_corner.z - origin.z),
    );
    let voxel_area = voxel_size * voxel_size;
    // An honest sample grid tops out near 2·`resolution` a side, since
    // no triangle exceeds the box it sits in. The clamp matters because
    // an overflowing `f32` area saturates on the cast rather than wraps.
    let max_grid = 4 * resolution as usize;

    for tri in triangles {
        // Adaptive density: the area term gives ~4 samples per voxel
        // cell, but area underestimates slivers, so the floor is also
        // the longest edge in voxels — one sample per voxel crossed.
        let area = 0.5 * (tri.v1 - tri.v0).cross(tri.v2 - tri.v0).length();
        let target_samples = ((area / voxel_area * 4.0).ceil() as usize).max(4);
        let area_n = (target_samples as f32).sqrt().ceil() as usize;

        let longest_edge = (tri.v1 - tri.v0)
            .length()
            .max((tri.v2 - tri.v1).length())
            .max((tri.v0 - tri.v2).length());
        let edge_n = (longest_edge / voxel_size).ceil() as usize + 1;

        let grid_n = area_n.max(edge_n).min(max_grid);
        let grid_n_f = grid_n as f32;

        // Stratified grid in barycentric space. The `u + v > 1` reject
        // is half of the unit square; the remaining cells form a
        // triangular lattice over the actual triangle.
        for i in 0..grid_n {
            for j in 0..grid_n {
                let u = (i as f32 + 0.5) / grid_n_f;
                let v = (j as f32 + 0.5) / grid_n_f;
                if u + v > 1.0 {
                    continue;
                }
                let w = 1.0 - u - v;
                let pos = tri.v0 * w + tri.v1 * u + tri.v2 * v;
                let cell = (
                    (((pos.x - origin.x) / voxel_size).floor() as i32).min(last.0),
                    (((pos.y - origin.y) / voxel_size).floor() as i32).min(last.1),
                    (((pos.z - origin.z) / voxel_size).floor() as i32).min(last.2),
                );
                let entry = grid.entry(cell).or_insert([0; 5]);
                entry[0] +=
                    (tri.c0[0] as f32 * w + tri.c1[0] as f32 * u + tri.c2[0] as f32 * v) as u32;
                entry[1] +=
                    (tri.c0[1] as f32 * w + tri.c1[1] as f32 * u + tri.c2[1] as f32 * v) as u32;
                entry[2] +=
                    (tri.c0[2] as f32 * w + tri.c1[2] as f32 * u + tri.c2[2] as f32 * v) as u32;
                entry[3] +=
                    (tri.c0[3] as f32 * w + tri.c1[3] as f32 * u + tri.c2[3] as f32 * v) as u32;
                entry[4] += 1;
            }
        }
    }

    grid
}

/// Average each cell's accumulated samples into a voxel. Alpha is
/// dropped rather than averaged — every voxel that reaches the world is
/// opaque, and a transparent black texel would pack to the sentinel.
fn finalize_surface(grid: HashMap<(i32, i32, i32), ColorAccum>) -> HashMap<(i32, i32, i32), Voxel> {
    grid.into_iter()
        .map(|(pos, [r, g, b, _a, count])| {
            let count = count.max(1); // can't be 0, but be paranoid
            (
                pos,
                Voxel::from_rgb((r / count) as u8, (g / count) as u8, (b / count) as u8),
            )
        })
        .collect()
}

// -------------------- interior fill --------------------

fn fill_interior(surface: &HashMap<(i32, i32, i32), Voxel>) -> HashMap<(i32, i32, i32), Voxel> {
    if surface.is_empty() {
        return HashMap::new();
    }

    let surface_set: HashSet<(i32, i32, i32)> = surface.keys().copied().collect();

    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut min_z = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    let mut max_z = i32::MIN;
    for &(x, y, z) in surface.keys() {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        min_z = min_z.min(z);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        max_z = max_z.max(z);
    }

    // Bitmask per non-surface cell: which axes' parity scans called it
    // inside. Two or more bits fills the cell, which tolerates a single
    // axis miscounting a grazing crossing.
    let mut inside_mask: HashMap<(i32, i32, i32), u8> = HashMap::new();

    // X-axis scan
    for y in min_y..=max_y {
        for z in min_z..=max_z {
            let mut inside = false;
            let mut last_was_surface = false;
            for x in min_x..=max_x {
                let is_surface = surface_set.contains(&(x, y, z));
                if is_surface && !last_was_surface {
                    inside = !inside;
                }
                if !is_surface && inside {
                    *inside_mask.entry((x, y, z)).or_insert(0) |= 0b001;
                }
                last_was_surface = is_surface;
            }
        }
    }

    // Y-axis scan
    for x in min_x..=max_x {
        for z in min_z..=max_z {
            let mut inside = false;
            let mut last_was_surface = false;
            for y in min_y..=max_y {
                let is_surface = surface_set.contains(&(x, y, z));
                if is_surface && !last_was_surface {
                    inside = !inside;
                }
                if !is_surface && inside {
                    *inside_mask.entry((x, y, z)).or_insert(0) |= 0b010;
                }
                last_was_surface = is_surface;
            }
        }
    }

    // Z-axis scan
    for x in min_x..=max_x {
        for y in min_y..=max_y {
            let mut inside = false;
            let mut last_was_surface = false;
            for z in min_z..=max_z {
                let is_surface = surface_set.contains(&(x, y, z));
                if is_surface && !last_was_surface {
                    inside = !inside;
                }
                if !is_surface && inside {
                    *inside_mask.entry((x, y, z)).or_insert(0) |= 0b100;
                }
                last_was_surface = is_surface;
            }
        }
    }

    // Default interior color = mean of all surface colors. Users
    // rarely see interior voxels (only when they remove surface cells)
    // but the mean keeps post-edit colors visually consistent.
    let (r_sum, g_sum, b_sum, count) = surface
        .values()
        .fold((0u64, 0u64, 0u64, 0u64), |(r, g, b, c), v| {
            (r + v.r as u64, g + v.g as u64, b + v.b as u64, c + 1)
        });
    // `NonZeroU64` rather than a `count > 0` guard: the divisor's
    // invariant then lives in the type instead of in a check three
    // lines above the three divisions that depend on it.
    let fill_voxel = match std::num::NonZeroU64::new(count) {
        Some(count) => Voxel::from_rgb(
            (r_sum / count) as u8,
            (g_sum / count) as u8,
            (b_sum / count) as u8,
        ),
        // Nothing on the surface to average — a mid gray keeps an
        // interior that shows through from reading as a color choice.
        None => Voxel::from_rgb(180, 180, 180),
    };

    let mut result = surface.clone();
    for (pos, mask) in inside_mask {
        if (mask as u32).count_ones() >= 2 {
            result.entry(pos).or_insert(fill_voxel);
        }
    }
    result
}

fn build_patch(voxels: HashMap<(i32, i32, i32), Voxel>) -> VoxelPatch {
    if voxels.is_empty() {
        return VoxelPatch::new();
    }
    // Re-anchor at (0, 0, 0) so the patch's footprint starts at the
    // origin regardless of where the GLB's mesh sat in glTF world
    // space. Caller can move via the selection / clipboard tools.
    let min_x = voxels.keys().map(|&(x, _, _)| x).min().unwrap();
    let min_y = voxels.keys().map(|&(_, y, _)| y).min().unwrap();
    let min_z = voxels.keys().map(|&(_, _, z)| z).min().unwrap();
    // Sorted so the patch is reproducible for a given input; `HashMap`
    // order is randomized per instance. Positions are map keys, so
    // sorting can't disturb any same-cell write order.
    let mut cells: Vec<_> = voxels.into_iter().collect();
    cells.sort_unstable_by_key(|&(pos, _)| pos);
    let mut patch = VoxelPatch::with_capacity(cells.len());
    for ((x, y, z), v) in cells {
        patch.set(x - min_x, y - min_y, z - min_z, v);
    }
    patch
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voxel(r: u8, g: u8, b: u8) -> Voxel {
        Voxel::from_rgb(r, g, b)
    }

    #[test]
    fn voxelize_rejects_extreme_resolutions() {
        let glb: &[u8] = &[];
        // Out of range
        assert!(voxelize_glb(glb, 0).is_err());
        assert!(voxelize_glb(glb, 1).is_err());
        assert!(voxelize_glb(glb, 257).is_err());
        // Empty GLB also fails (but only after the range check passes)
        let _ = voxelize_glb(glb, 64);
    }

    #[test]
    fn voxelize_handles_malformed_bytes() {
        let bytes: &[u8] = b"not a glb";
        let result = voxelize_glb(bytes, 64);
        assert!(result.is_err());
    }

    /// Two rows, red on top and blue below. glTF's v grows downward
    /// from a top-left origin, so v = 0 must land on red — every
    /// assertion here fails with the axis inverted.
    #[test]
    fn texture_v_axis_runs_top_down_like_gltf_says() {
        let tex = DecodedImage {
            rgba: vec![
                255, 0, 0, 255, // row 0 — red
                0, 0, 255, 255, // row 1 — blue
            ],
            width: 1,
            height: 2,
        };

        let top = sample_texture(&tex, 0.5, 0.0);
        assert_eq!(top[0], 1.0, "v = 0 must sample the top row");
        assert_eq!(top[2], 0.0);

        let bottom = sample_texture(&tex, 0.5, 0.99);
        assert_eq!(bottom[2], 1.0, "v near 1 must sample the bottom row");
        assert_eq!(bottom[0], 0.0);
    }

    /// v = 1.0 exactly is a legal UV and wraps to 0.0 under REPEAT, so
    /// it samples the *top* row — and the clamp keeps any value that
    /// slips through from indexing past the last row.
    #[test]
    fn texture_sampling_stays_in_bounds_at_the_edges() {
        let tex = DecodedImage {
            rgba: vec![255, 0, 0, 255, 0, 0, 255, 255],
            width: 1,
            height: 2,
        };

        assert_eq!(sample_texture(&tex, 0.0, 1.0)[0], 1.0);
        // Far outside on both axes: wraps, never panics.
        let _ = sample_texture(&tex, -3.7, 42.9);
    }

    #[test]
    fn pack_rgba_clamps_to_byte_range() {
        // baseColorFactor can be > 1 in glTF (HDR materials, rare but
        // legal); we clamp rather than wrap.
        assert_eq!(pack_rgba([2.0, -0.5, 0.5, 1.0]), [255, 0, 127, 255]);
    }

    #[test]
    fn compose_base_color_multiplies_all_present_sources() {
        // glTF rule: factor × texture × COLOR_0. R = 0.5·0.5·0.5.
        let out = compose_base_color(
            Some([0.5, 1.0, 1.0, 1.0]),
            Some([0.5, 0.5, 1.0, 1.0]),
            [0.5, 1.0, 1.0, 1.0],
        );
        assert_eq!(out[0], (0.5 * 0.5 * 0.5 * 255.0) as u8); // 31
        assert_eq!(out[1], (0.5 * 255.0) as u8); // 127 (only texture ≠ 1)
        assert_eq!(out[2], 255);
        assert_eq!(out[3], 255);
    }

    #[test]
    fn compose_base_color_white_vertex_plus_texture_keeps_texture() {
        // The regressed case: a white COLOR_0 beside a texture must not
        // wash the texture out (old code picked vertex *or* texture).
        let tex = [0.2, 0.4, 0.6, 1.0];
        let out = compose_base_color(Some([1.0, 1.0, 1.0, 1.0]), Some(tex), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(out, pack_rgba(tex));
    }

    #[test]
    fn compose_base_color_gray_only_when_nothing_to_color() {
        // No COLOR_0, no texture, default-white factor → neutral gray.
        assert_eq!(
            compose_base_color(None, None, [1.0, 1.0, 1.0, 1.0]),
            [200, 200, 200, 255]
        );
        // An explicit non-default factor is honored, not overridden by gray.
        assert_eq!(
            compose_base_color(None, None, [0.8, 0.2, 0.1, 1.0]),
            pack_rgba([0.8, 0.2, 0.1, 1.0])
        );
    }

    #[test]
    fn rasterize_thin_triangle_covers_its_full_length() {
        // A long thin sliver: near-zero area but spans ~20 voxels along X.
        // Area-only sampling would take the 4-sample floor and leave gaps;
        // the edge-based floor must sample every voxel it crosses.
        let c = [255u8, 255, 255, 255];
        let tri = Triangle {
            v0: Vec3::new(0.0, 0.0, 0.0),
            v1: Vec3::new(20.0, 0.0, 0.0),
            v2: Vec3::new(20.0, 0.05, 0.0), // 0.05 tall → tiny area
            c0: c,
            c1: c,
            c2: c,
        };
        let grid = rasterize_triangles(&[tri], Vec3::ZERO, Vec3::new(20.0, 1.0, 1.0), 1.0, 20);
        for x in 0..20 {
            assert!(
                grid.keys().any(|&(gx, _, _)| gx == x),
                "no sample landed in the x={} column — the sliver has a hole",
                x
            );
        }
    }

    /// `resolution` counts voxels along the longest axis. Geometry on
    /// the far face divides exactly, so `floor` would put those samples
    /// a layer past the model — which an axis-aligned cube always hits.
    #[test]
    fn geometry_on_the_far_face_lands_in_the_last_cell_not_past_it() {
        let c = [255u8, 255, 255, 255];
        let tri = Triangle {
            v0: Vec3::new(0.0, 0.0, 0.0),
            v1: Vec3::new(10.0, 0.0, 0.0),
            v2: Vec3::new(10.0, 10.0, 0.0),
            c0: c,
            c1: c,
            c2: c,
        };
        let grid = rasterize_triangles(&[tri], Vec3::ZERO, Vec3::new(10.0, 10.0, 0.0), 1.0, 10);
        let widest = grid.keys().map(|&(x, _, _)| x).max().unwrap();
        let tallest = grid.keys().map(|&(_, y, _)| y).max().unwrap();
        assert_eq!(widest, 9, "10 cells across means indices 0..=9");
        assert_eq!(tallest, 9);
    }

    /// A node listing itself as its own child. The `gltf` crate only
    /// validates that indices are in range, so without the visited set
    /// a 112-byte file recurses until the stack runs out.
    #[test]
    fn a_self_referencing_node_is_pruned_rather_than_followed() {
        let json = br#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"children":[0]}]}"#;
        let mut chunk = json.to_vec();
        while !chunk.len().is_multiple_of(4) {
            chunk.push(b' ');
        }
        let mut glb = Vec::new();
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&((12 + 8 + chunk.len()) as u32).to_le_bytes());
        glb.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F_534Au32.to_le_bytes()); // "JSON"
        glb.extend_from_slice(&chunk);

        // The walk terminates; there is no geometry in it, so the
        // ordinary "nothing to voxelize" refusal is what comes back.
        let error = voxelize_glb(&glb, 32).expect_err("this file has no triangles");
        assert!(
            error.to_string().contains("no triangle primitives"),
            "unexpected error: {error}"
        );
    }

    /// Vertices around 2e10 overflow the `f32` cross product to
    /// infinity, and `f32 → usize` saturates: without the clamp the
    /// sample grid is 4.29 billion a side, on the drawing thread.
    #[test]
    fn a_triangle_with_overflowing_coordinates_samples_a_bounded_grid() {
        let c = [255u8, 255, 255, 255];
        let tri = Triangle {
            v0: Vec3::ZERO,
            v1: Vec3::new(2.0e10, 0.0, 0.0),
            v2: Vec3::new(0.0, 2.0e10, 0.0),
            c0: c,
            c1: c,
            c2: c,
        };
        let resolution = 32u32;
        let voxel_size = 2.0e10 / resolution as f32;
        assert!(
            (0.5 * (tri.v1 - tri.v0).cross(tri.v2 - tri.v0).length()).is_infinite(),
            "precondition: this triangle's area overflows f32"
        );

        let grid = rasterize_triangles(
            &[tri],
            Vec3::ZERO,
            Vec3::new(2.0e10, 2.0e10, 0.0),
            voxel_size,
            resolution,
        );
        let max_grid = 4 * resolution as usize;
        assert!(
            grid.len() <= max_grid * max_grid,
            "sampled {} cells; the grid is capped at {}²",
            grid.len(),
            max_grid
        );
    }

    #[test]
    fn fill_interior_marks_enclosed_cells_inside_a_hollow_box() {
        // 5×5×5 hollow box (surface only) — every voxel except the
        // interior 3×3×3 is on the surface. After fill_interior,
        // every interior cell should be filled with the mean color.
        let mut surface: HashMap<(i32, i32, i32), Voxel> = HashMap::new();
        let red = voxel(255, 0, 0);
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..5 {
                    let on_surface = x == 0 || x == 4 || y == 0 || y == 4 || z == 0 || z == 4;
                    if on_surface {
                        surface.insert((x, y, z), red);
                    }
                }
            }
        }

        let filled = fill_interior(&surface);

        // Interior 3×3×3 should now be filled.
        for x in 1..4 {
            for y in 1..4 {
                for z in 1..4 {
                    assert!(
                        filled.contains_key(&(x, y, z)),
                        "interior cell ({},{},{}) should be filled",
                        x,
                        y,
                        z
                    );
                }
            }
        }
        // Surface count unchanged (98 cells for a 5×5×5 hollow shell).
        let surface_count = filled
            .iter()
            .filter(|(p, _)| surface.contains_key(p))
            .count();
        assert_eq!(surface_count, surface.len());
    }

    #[test]
    fn fill_interior_does_not_fill_an_open_l_shape() {
        // L-shape made of 2 cells — no enclosed interior. fill_interior
        // shouldn't invent any cells.
        let mut surface: HashMap<(i32, i32, i32), Voxel> = HashMap::new();
        surface.insert((0, 0, 0), voxel(255, 0, 0));
        surface.insert((1, 0, 0), voxel(255, 0, 0));

        let filled = fill_interior(&surface);
        assert_eq!(filled.len(), 2, "open shape shouldn't grow new cells");
    }

    #[test]
    fn build_patch_translates_to_origin_aligned_aabb() {
        // Input has cells at negative / non-origin coords; patch should
        // start at (0, 0, 0) so it lands cleanly at world origin.
        let mut voxels: HashMap<(i32, i32, i32), Voxel> = HashMap::new();
        voxels.insert((-2, 5, -3), voxel(255, 0, 0));
        voxels.insert((0, 7, -1), voxel(0, 255, 0));

        let patch = build_patch(voxels);
        let positions: HashSet<_> = patch.voxels.iter().map(|(p, _)| *p).collect();
        // Translated by (+2, -5, +3) so min becomes (0, 0, 0).
        assert!(positions.contains(&(0, 0, 0)));
        assert!(positions.contains(&(2, 2, 2)));
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn finalize_surface_averages_colors_at_overlapping_samples() {
        // Two samples land on the same cell with different colors;
        // the cell should end up at their per-channel mean.
        let mut grid: HashMap<(i32, i32, i32), ColorAccum> = HashMap::new();
        grid.insert((1, 2, 3), [200 + 100, 100, 100 + 200, 255 + 255, 2]);
        let surface = finalize_surface(grid);
        let v = surface.get(&(1, 2, 3)).expect("cell exists");
        assert_eq!(v.r, 150);
        assert_eq!(v.g, 50);
        assert_eq!(v.b, 150);
        assert_eq!(v.a, 255);
    }

    #[test]
    fn finalize_surface_ignores_texture_alpha() {
        // Voxels in the world are always opaque, so a transparent texel
        // must not carry its alpha through. Transparent black is the
        // dangerous case — it packs to the mesher's "no face" sentinel.
        let mut grid: HashMap<(i32, i32, i32), ColorAccum> = HashMap::new();
        grid.insert((0, 0, 0), [120, 130, 140, 40, 1]);
        grid.insert((1, 0, 0), [0, 0, 0, 0, 1]);
        let surface = finalize_surface(grid);
        assert_eq!(surface[&(0, 0, 0)].a, 255);
        let black = surface[&(1, 0, 0)];
        assert_eq!(black.color(), [0, 0, 0, 255]);
        assert!(black.is_solid());
    }
}
