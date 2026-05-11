//! GLB → `procgen::VoxelPatch` conversion.
//!
//! Pipeline:
//! 1. Parse GLB via the `gltf` crate.
//! 2. Walk the scene graph, applying cumulative node transforms.
//! 3. Extract triangles with vertex positions + per-vertex colors.
//!    Color priority: COLOR_0 vertex attribute → baseColorTexture
//!    sampled at the vertex's UV → baseColorFactor uniform → light-
//!    gray default.
//! 4. Voxelize the surface by grid-sampling each triangle (sample
//!    density tied to triangle area / voxel area so even very large
//!    triangles fill their full voxel footprint).
//! 5. Fill the interior by a 3-axis parity scan + majority vote — a
//!    cell is "inside" when ≥ 2 of the 3 axis scans say so. Robust to
//!    minor mesh defects (a single missed surface crossing on one
//!    axis won't leak the whole interior).
//! 6. Translate to non-negative coordinates and emit as `VoxelPatch`
//!    so the caller can apply it via `Command::set_voxels` and the
//!    result lands at the world origin (downstream tools can Move it).
//!
//! Phase 3 is the first place this gets called (Phase 2 only
//! downloaded the GLB and dropped the bytes); Phase 4 will polish
//! placement, auto-select, and recent-prompts MRU.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};
use glam::{Mat4, Quat, Vec3};

use crate::core::Voxel;
use crate::procgen::VoxelPatch;

/// Voxelize a GLB binary into a `VoxelPatch` whose coordinates start
/// at `(0, 0, 0)`. `resolution` is the number of voxels along the
/// mesh's longest axis; the other two axes shrink in proportion.
///
/// Bails on:
/// - resolution out of range (4..=256)
/// - malformed GLB
/// - GLB with no triangle primitives
pub fn voxelize_glb(bytes: &[u8], resolution: u32) -> Result<VoxelPatch> {
    if !(4..=256).contains(&resolution) {
        bail!("Resolution must be in 4..=256, got {}", resolution);
    }

    let (document, buffers, images) =
        gltf::import_slice(bytes).context("Parsing GLB")?;

    let textures: Vec<DecodedImage> = images
        .iter()
        .map(decode_image)
        .collect::<Result<Vec<_>>>()?;

    // Prefer the explicit default scene; fall back to the first scene;
    // if neither exists, walk all meshes directly (some exporters
    // produce GLBs with no scene node — rare but seen in the wild).
    let mut triangles = Vec::new();
    let scene = document.default_scene().or_else(|| document.scenes().next());
    if let Some(scene) = scene {
        for node in scene.nodes() {
            walk_node(
                &node,
                Mat4::IDENTITY,
                &buffers,
                &textures,
                &mut triangles,
            );
        }
    } else {
        for mesh in document.meshes() {
            extract_from_mesh(&mesh, Mat4::IDENTITY, &buffers, &textures, &mut triangles);
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

    let accumulator = rasterize_triangles(&triangles, aabb_min, voxel_size);
    let surface = finalize_surface(accumulator);
    let filled = fill_interior(&surface);
    Ok(build_patch(filled))
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

fn decode_image(image: &gltf::image::Data) -> Result<DecodedImage> {
    use gltf::image::Format::*;
    let (width, height) = (image.width, image.height);
    let pixel_count = (width as usize) * (height as usize);
    let rgba = match image.format {
        R8G8B8A8 => image.pixels.clone(),
        R8G8B8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in image.pixels.chunks_exact(3) {
                out.extend_from_slice(chunk);
                out.push(255);
            }
            out
        }
        R8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for &g in &image.pixels {
                out.extend_from_slice(&[g, g, g, 255]);
            }
            out
        }
        R8G8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in image.pixels.chunks_exact(2) {
                out.extend_from_slice(&[chunk[0], chunk[0], chunk[0], chunk[1]]);
            }
            out
        }
        other => {
            // 16-bit and float formats are technically allowed by glTF
            // but extremely rare for baseColorTexture. Falling back to
            // a neutral gray avoids failing the whole pipeline on an
            // unexpected texture — user sees a gray model and can
            // tell us if it matters.
            log::warn!("Unsupported texture format {:?}; using gray fallback", other);
            vec![200; pixel_count * 4]
        }
    };
    Ok(DecodedImage { rgba, width, height })
}

fn walk_node(
    node: &gltf::Node,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    textures: &[DecodedImage],
    triangles: &mut Vec<Triangle>,
) {
    let local = mat4_from_transform(node.transform());
    let transform = parent_transform * local;

    if let Some(mesh) = node.mesh() {
        extract_from_mesh(&mesh, transform, buffers, textures, triangles);
    }

    for child in node.children() {
        walk_node(&child, transform, buffers, textures, triangles);
    }
}

fn mat4_from_transform(t: gltf::scene::Transform) -> Mat4 {
    match t {
        gltf::scene::Transform::Matrix { matrix } => {
            Mat4::from_cols_array_2d(&matrix)
        }
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
    textures: &[DecodedImage],
    triangles: &mut Vec<Triangle>,
) {
    for primitive in mesh.primitives() {
        if primitive.mode() != gltf::mesh::Mode::Triangles {
            // Skip lines / points / triangle strips — Hunyuan3D V3
            // doesn't emit them, but a future provider might and we'd
            // rather no-op a primitive than panic.
            continue;
        }
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

        let positions: Vec<[f32; 3]> = match reader.read_positions() {
            Some(p) => p.collect(),
            None => continue,
        };

        // Vertex colors (optional). When present they take priority
        // over textures per the glTF spec's "factor * vertex * texture"
        // composition, but for AI 3D-gen output they're usually empty
        // and we end up sampling the baseColorTexture instead.
        let vertex_colors: Option<Vec<[f32; 4]>> = reader
            .read_colors(0)
            .map(|c| c.into_rgba_f32().collect());

        let tex_coords: Option<Vec<[f32; 2]>> = reader
            .read_tex_coords(0)
            .map(|tc| tc.into_f32().collect());

        let material = primitive.material();
        let pbr = material.pbr_metallic_roughness();
        let base_factor = pbr.base_color_factor();
        let base_texture: Option<&DecodedImage> = pbr
            .base_color_texture()
            .and_then(|info| textures.get(info.texture().source().index()));

        let color_at_vertex = |i: usize| -> [u8; 4] {
            // Source the per-vertex color, then multiply by the
            // material's baseColorFactor (glTF composition rule).
            if let Some(colors) = vertex_colors.as_ref() {
                let c = colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]);
                pack_rgba([
                    c[0] * base_factor[0],
                    c[1] * base_factor[1],
                    c[2] * base_factor[2],
                    c[3] * base_factor[3],
                ])
            } else if let (Some(tex), Some(uvs)) = (base_texture, tex_coords.as_ref()) {
                let uv = uvs.get(i).copied().unwrap_or([0.5, 0.5]);
                let sampled = sample_texture(tex, uv[0], uv[1]);
                pack_rgba([
                    sampled[0] * base_factor[0],
                    sampled[1] * base_factor[1],
                    sampled[2] * base_factor[2],
                    sampled[3] * base_factor[3],
                ])
            } else {
                pack_rgba(base_factor)
            }
        };

        let world_pos = |i: usize| -> Vec3 {
            transform.transform_point3(Vec3::from_array(positions[i]))
        };

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
    // glTF UVs use bottom-left origin; image rows are top-down. Flip Y.
    let y = (((1.0 - v) * tex.height as f32) as u32)
        .min(tex.height.saturating_sub(1));
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
    voxel_size: f32,
) -> HashMap<(i32, i32, i32), ColorAccum> {
    let mut grid: HashMap<(i32, i32, i32), ColorAccum> = HashMap::new();
    let voxel_area = voxel_size * voxel_size;

    for tri in triangles {
        // Triangle area for adaptive sampling density. Small triangles
        // get the minimum 4 samples; large ones get enough to cover
        // every voxel they touch (≈ 4 samples per voxel-cell area).
        let area = 0.5 * (tri.v1 - tri.v0).cross(tri.v2 - tri.v0).length();
        let target_samples =
            ((area / voxel_area * 4.0).ceil() as usize).max(4);
        let grid_n = (target_samples as f32).sqrt().ceil() as usize;
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
                    ((pos.x - origin.x) / voxel_size).floor() as i32,
                    ((pos.y - origin.y) / voxel_size).floor() as i32,
                    ((pos.z - origin.z) / voxel_size).floor() as i32,
                );
                let entry = grid.entry(cell).or_insert([0; 5]);
                entry[0] +=
                    (tri.c0[0] as f32 * w + tri.c1[0] as f32 * u + tri.c2[0] as f32 * v)
                        as u32;
                entry[1] +=
                    (tri.c0[1] as f32 * w + tri.c1[1] as f32 * u + tri.c2[1] as f32 * v)
                        as u32;
                entry[2] +=
                    (tri.c0[2] as f32 * w + tri.c1[2] as f32 * u + tri.c2[2] as f32 * v)
                        as u32;
                entry[3] +=
                    (tri.c0[3] as f32 * w + tri.c1[3] as f32 * u + tri.c2[3] as f32 * v)
                        as u32;
                entry[4] += 1;
            }
        }
    }

    grid
}

fn finalize_surface(
    grid: HashMap<(i32, i32, i32), ColorAccum>,
) -> HashMap<(i32, i32, i32), Voxel> {
    grid.into_iter()
        .map(|(pos, [r, g, b, a, count])| {
            let count = count.max(1); // can't be 0, but be paranoid
            (
                pos,
                Voxel::from_rgba(
                    (r / count) as u8,
                    (g / count) as u8,
                    (b / count) as u8,
                    (a / count) as u8,
                ),
            )
        })
        .collect()
}

// -------------------- interior fill --------------------

fn fill_interior(
    surface: &HashMap<(i32, i32, i32), Voxel>,
) -> HashMap<(i32, i32, i32), Voxel> {
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

    // Bitmask per non-surface cell: which axes' parity scans flagged
    // it as "inside". A cell with ≥ 2 bits set is voted inside and
    // gets filled; this tolerates a single axis miscount (e.g. from a
    // grazing surface crossing).
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
    let (r_sum, g_sum, b_sum, count) =
        surface
            .values()
            .fold((0u64, 0u64, 0u64, 0u64), |(r, g, b, c), v| {
                (r + v.r as u64, g + v.g as u64, b + v.b as u64, c + 1)
            });
    let fill_voxel = if count > 0 {
        Voxel::from_rgb(
            (r_sum / count) as u8,
            (g_sum / count) as u8,
            (b_sum / count) as u8,
        )
    } else {
        Voxel::from_rgb(180, 180, 180)
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
    let mut patch = VoxelPatch::with_capacity(voxels.len());
    for ((x, y, z), v) in voxels {
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

    #[test]
    fn pack_rgba_clamps_to_byte_range() {
        // baseColorFactor can be > 1 in glTF (HDR materials, rare but
        // legal); we clamp rather than wrap.
        assert_eq!(pack_rgba([2.0, -0.5, 0.5, 1.0]), [255, 0, 127, 255]);
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
        let surface_count =
            filled.iter().filter(|(p, _)| surface.contains_key(p)).count();
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
        grid.insert((1, 2, 3), [200 + 100, 0 + 100, 100 + 200, 255 + 255, 2]);
        let surface = finalize_surface(grid);
        let v = surface.get(&(1, 2, 3)).expect("cell exists");
        assert_eq!(v.r, 150);
        assert_eq!(v.g, 50);
        assert_eq!(v.b, 150);
        assert_eq!(v.a, 255);
    }
}
