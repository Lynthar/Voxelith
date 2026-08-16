//! Marching Cubes for export-time smoothing. Samples a density field
//! at voxel centers, optionally blurs it, and marches it into one
//! world-space mesh. Export only — never on the render path.

mod tables;

use std::collections::HashMap;

use tables::{EDGE_TABLE, TRI_TABLE};
use thiserror::Error;

use crate::core::{Voxel, World};
use crate::mesh::{ChunkMesh, Vertex};

/// Cap on the dense density field, in samples. 64 Mi cells is roughly
/// a 400³ bounding box and holds the field to 256 MiB — 512 MiB while
/// the blur's second buffer is alive.
const MAX_DENSITY_CELLS: usize = 64 * 1024 * 1024;

/// Why a smoothed mesh couldn't be produced.
#[derive(Debug, Error)]
pub enum SmoothMeshError {
    #[error(
        "scene spans {min:?} to {max:?}, which needs a {cells}-cell density \
         field (limit {MAX_DENSITY_CELLS}); delete stray far-away voxels, or \
         export without smoothing"
    )]
    SceneTooLarge {
        min: (i32, i32, i32),
        max: (i32, i32, i32),
        cells: u128,
    },
}

/// Density above which a sample counts as inside the surface. With
/// voxel-centered density this puts the surface midway between a solid
/// voxel and an air one.
const ISO_LEVEL: f32 = 0.5;

/// Marching-Cubes mesh of the whole world as one flat `ChunkMesh`.
/// `smooth` adds a 3×3×3 blur first: without it surfaces are rounded
/// cubes, with it clay — thin features shrink or vanish.
///
/// # Errors
/// [`SmoothMeshError::SceneTooLarge`] when the scene's bounding box
/// would need more than `MAX_DENSITY_CELLS` samples.
pub fn mesh_world_smoothed(world: &World, smooth: bool) -> Result<ChunkMesh, SmoothMeshError> {
    use crate::core::ChunkPos;
    let Some((bbox_min, bbox_max)) = world.scene_aabb() else {
        return Ok(ChunkMesh::new(ChunkPos::ZERO));
    };

    // Sampled at every integer position in [min, max + 1] so boundary
    // cubes see a zero gradient outside, plus ±1 padding for the blur
    // kernel — hence extent + 3 per axis.
    let pad = 1;
    let min = (bbox_min.0 - pad, bbox_min.1 - pad, bbox_min.2 - pad);
    let max = (
        bbox_max.0 + 1 + pad,
        bbox_max.1 + 1 + pad,
        bbox_max.2 + 1 + pad,
    );
    let size = (
        (max.0 - min.0 + 1) as usize,
        (max.1 - min.1 + 1) as usize,
        (max.2 - min.2 + 1) as usize,
    );
    // Cost follows the bounding box, not the voxel count. A failed
    // `vec!` aborts the process, so refuse up front; `u128` because the
    // product can wrap a `usize` at extreme coordinates.
    let cells = size.0 as u128 * size.1 as u128 * size.2 as u128;
    if cells > MAX_DENSITY_CELLS as u128 {
        return Err(SmoothMeshError::SceneTooLarge {
            min: bbox_min,
            max: bbox_max,
            cells,
        });
    }
    let total = size.0 * size.1 * size.2;

    // Raw density: 1.0 if the voxel at the sample point is solid,
    // 0.0 if air. Sampling at integer positions means each density
    // sample IS a voxel — no extra averaging needed for the raw pass.
    let mut density = vec![0.0_f32; total];
    let idx =
        |dx: usize, dy: usize, dz: usize| -> usize { dx + dy * size.0 + dz * size.0 * size.1 };
    for dz in 0..size.2 {
        for dy in 0..size.1 {
            for dx in 0..size.0 {
                let wx = min.0 + dx as i32;
                let wy = min.1 + dy as i32;
                let wz = min.2 + dz as i32;
                if !world.get_voxel(wx, wy, wz).is_air() {
                    density[idx(dx, dy, dz)] = 1.0;
                }
            }
        }
    }

    if smooth {
        density = box_blur_3x3x3(&density, size);
    }

    // March every cube whose corners stay within the field bounds.
    // A cube at (gx, gy, gz) uses corners (gx..gx+1, gy..gy+1,
    // gz..gz+1), so the cube range stops 1 short of `size` per axis.
    let mut mesh = ChunkMesh::new(ChunkPos::ZERO);
    let mut shared: HashMap<EdgeKey, u32> = HashMap::new();
    for gz in 0..size.2 - 1 {
        for gy in 0..size.1 - 1 {
            for gx in 0..size.0 - 1 {
                march_one_cube(
                    &density,
                    size,
                    &idx,
                    gx,
                    gy,
                    gz,
                    min,
                    world,
                    &mut mesh,
                    &mut shared,
                );
            }
        }
    }
    Ok(mesh)
}

/// Process a single MC cube at field-local index `(gx, gy, gz)`.
/// Samples the 8 corners' densities, looks up the triangulation
/// from the standard MC tables, and emits triangles into `mesh`.
#[allow(clippy::too_many_arguments)]
fn march_one_cube(
    density: &[f32],
    size: (usize, usize, usize),
    idx: &dyn Fn(usize, usize, usize) -> usize,
    gx: usize,
    gy: usize,
    gz: usize,
    field_min: (i32, i32, i32),
    world: &World,
    mesh: &mut ChunkMesh,
    shared: &mut HashMap<EdgeKey, u32>,
) {
    // Corner numbering follows Bourke's convention so the EDGE_TABLE
    // and TRI_TABLE indices line up: 0-3 walk the y=gy face as
    // (x,z) = (0,0) (1,0) (1,1) (0,1), and 4-7 repeat it at gy+1.
    let corners_local: [(usize, usize, usize); 8] = [
        (gx, gy, gz),
        (gx + 1, gy, gz),
        (gx + 1, gy, gz + 1),
        (gx, gy, gz + 1),
        (gx, gy + 1, gz),
        (gx + 1, gy + 1, gz),
        (gx + 1, gy + 1, gz + 1),
        (gx, gy + 1, gz + 1),
    ];
    let densities: [f32; 8] = [
        density[idx(corners_local[0].0, corners_local[0].1, corners_local[0].2)],
        density[idx(corners_local[1].0, corners_local[1].1, corners_local[1].2)],
        density[idx(corners_local[2].0, corners_local[2].1, corners_local[2].2)],
        density[idx(corners_local[3].0, corners_local[3].1, corners_local[3].2)],
        density[idx(corners_local[4].0, corners_local[4].1, corners_local[4].2)],
        density[idx(corners_local[5].0, corners_local[5].1, corners_local[5].2)],
        density[idx(corners_local[6].0, corners_local[6].1, corners_local[6].2)],
        density[idx(corners_local[7].0, corners_local[7].1, corners_local[7].2)],
    ];

    // Build the 8-bit cube index: bit i set iff corner i is "inside"
    // (density >= ISO_LEVEL). EDGE_TABLE[index] tells us which of the
    // 12 edges intersect the surface.
    let mut cube_index: usize = 0;
    for (i, &density) in densities.iter().enumerate() {
        if density >= ISO_LEVEL {
            cube_index |= 1 << i;
        }
    }
    let edges = EDGE_TABLE[cube_index];
    if edges == 0 {
        return; // entirely inside or outside — no surface here
    }

    // World-space corner positions for emit time.
    let corners_world: [[f32; 3]; 8] = [
        [
            (field_min.0 + gx as i32) as f32,
            (field_min.1 + gy as i32) as f32,
            (field_min.2 + gz as i32) as f32,
        ],
        [
            (field_min.0 + gx as i32 + 1) as f32,
            (field_min.1 + gy as i32) as f32,
            (field_min.2 + gz as i32) as f32,
        ],
        [
            (field_min.0 + gx as i32 + 1) as f32,
            (field_min.1 + gy as i32) as f32,
            (field_min.2 + gz as i32 + 1) as f32,
        ],
        [
            (field_min.0 + gx as i32) as f32,
            (field_min.1 + gy as i32) as f32,
            (field_min.2 + gz as i32 + 1) as f32,
        ],
        [
            (field_min.0 + gx as i32) as f32,
            (field_min.1 + gy as i32 + 1) as f32,
            (field_min.2 + gz as i32) as f32,
        ],
        [
            (field_min.0 + gx as i32 + 1) as f32,
            (field_min.1 + gy as i32 + 1) as f32,
            (field_min.2 + gz as i32) as f32,
        ],
        [
            (field_min.0 + gx as i32 + 1) as f32,
            (field_min.1 + gy as i32 + 1) as f32,
            (field_min.2 + gz as i32 + 1) as f32,
        ],
        [
            (field_min.0 + gx as i32) as f32,
            (field_min.1 + gy as i32 + 1) as f32,
            (field_min.2 + gz as i32 + 1) as f32,
        ],
    ];

    // The 12 potential edge vertices, filled lazily — only those
    // flagged in `edges` are used. Edge i connects the corner pair at
    // EDGE_VERTEX_PAIRS[i].
    type EdgeVertex = (([f32; 3], [f32; 3], [f32; 4]), bool);
    let mut edge_vertices: [EdgeVertex; 12] = [(([0.0; 3], [0.0; 3], [0.0; 4]), false); 12];
    // Field-global identity of each edge, so neighbouring cubes reuse
    // one vertex instead of each emitting their own copy.
    let mut edge_keys: [EdgeKey; 12] = [((0, 0, 0), 0); 12];
    for e in 0..12 {
        if edges & (1 << e) == 0 {
            continue;
        }
        let (a, b) = EDGE_VERTEX_PAIRS[e];
        edge_keys[e] = edge_key(corners_local[a], corners_local[b]);
        let pos = interp_edge(
            corners_world[a],
            corners_world[b],
            densities[a],
            densities[b],
        );
        // Shift +0.5 on every axis: MC's surface is voxel-centered, so
        // this lands it on the same [n, n+1] cell the greedy mesher and
        // socket placement use. `edge_color` keeps the unshifted coords.
        let pos = [pos[0] + 0.5, pos[1] + 0.5, pos[2] + 0.5];
        let normal = density_gradient(
            density,
            size,
            idx,
            corners_local,
            a,
            b,
            densities[a],
            densities[b],
        );
        let color = edge_color(world, corners_world[a], corners_world[b]);
        edge_vertices[e] = ((pos, normal, color), true);
    }

    // TRI_TABLE rows are -1-terminated triples of edge indices. A few
    // configurations come out wound CW from outside, so every triangle
    // is checked against the outward normal below and flipped if so.
    let row = TRI_TABLE[cube_index];
    let mut i = 0;
    while i < row.len() && row[i] != -1 {
        let e0 = row[i] as usize;
        let e1 = row[i + 1] as usize;
        let e2 = row[i + 2] as usize;
        let v0 = Vertex::new(
            edge_vertices[e0].0 .0,
            edge_vertices[e0].0 .1,
            edge_vertices[e0].0 .2,
        );
        let v1 = Vertex::new(
            edge_vertices[e1].0 .0,
            edge_vertices[e1].0 .1,
            edge_vertices[e1].0 .2,
        );
        let v2 = Vertex::new(
            edge_vertices[e2].0 .0,
            edge_vertices[e2].0 .1,
            edge_vertices[e2].0 .2,
        );

        // Cross product (v1-v0) × (v2-v0).
        let e_a = [
            v1.position[0] - v0.position[0],
            v1.position[1] - v0.position[1],
            v1.position[2] - v0.position[2],
        ];
        let e_b = [
            v2.position[0] - v0.position[0],
            v2.position[1] - v0.position[1],
            v2.position[2] - v0.position[2],
        ];
        let cross = [
            e_a[1] * e_b[2] - e_a[2] * e_b[1],
            e_a[2] * e_b[0] - e_a[0] * e_b[2],
            e_a[0] * e_b[1] - e_a[1] * e_b[0],
        ];
        let dot = cross[0] * v0.normal[0] + cross[1] * v0.normal[1] + cross[2] * v0.normal[2];

        let i0 = intern_edge_vertex(mesh, shared, edge_keys[e0], v0);
        let i1 = intern_edge_vertex(mesh, shared, edge_keys[e1], v1);
        let i2 = intern_edge_vertex(mesh, shared, edge_keys[e2], v2);
        if dot < 0.0 {
            // Swap the indices, not the vertices — those belong to
            // other triangles too.
            mesh.indices.extend_from_slice(&[i0, i2, i1]);
        } else {
            mesh.indices.extend_from_slice(&[i0, i1, i2]);
        }
        i += 3;
    }
}

/// Identity of a marching-cubes edge within the density field: its low
/// corner plus the axis it runs along. Two cubes that share an edge
/// produce the same key.
type EdgeKey = ((usize, usize, usize), u8);

fn edge_key(a: (usize, usize, usize), b: (usize, usize, usize)) -> EdgeKey {
    let axis = if a.0 != b.0 {
        0
    } else if a.1 != b.1 {
        1
    } else {
        2
    };
    ((a.0.min(b.0), a.1.min(b.1), a.2.min(b.2)), axis)
}

/// The vertex index for `key`, creating it on first use. Every MC
/// vertex attribute is a property of the edge alone, so sharing is
/// lossless — and the first writer fixes the position for everyone.
fn intern_edge_vertex(
    mesh: &mut ChunkMesh,
    shared: &mut HashMap<EdgeKey, u32>,
    key: EdgeKey,
    vertex: Vertex,
) -> u32 {
    if let Some(&index) = shared.get(&key) {
        return index;
    }
    let index = mesh.vertices.len() as u32;
    mesh.vertices.push(vertex);
    shared.insert(key, index);
    index
}

/// Linear interpolation of an edge crossing: the surface sits at
/// `t = (ISO_LEVEL - da) / (db - da)`. Two densities too close to
/// divide by fall back to the midpoint.
fn interp_edge(a: [f32; 3], b: [f32; 3], da: f32, db: f32) -> [f32; 3] {
    let denom = db - da;
    let t = if denom.abs() < 1e-6 {
        0.5
    } else {
        ((ISO_LEVEL - da) / denom).clamp(0.0, 1.0)
    };
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Surface normal at a vertex: gradient of the density field at the
/// vertex, normalized. Linearly interpolating the gradients at the
/// two corners gives a smooth normal across the edge.
#[allow(clippy::too_many_arguments)]
fn density_gradient(
    density: &[f32],
    size: (usize, usize, usize),
    idx: &dyn Fn(usize, usize, usize) -> usize,
    corners_local: [(usize, usize, usize); 8],
    a: usize,
    b: usize,
    da: f32,
    db: f32,
) -> [f32; 3] {
    let g_a = sample_gradient(density, size, idx, corners_local[a]);
    let g_b = sample_gradient(density, size, idx, corners_local[b]);
    let denom = db - da;
    let t = if denom.abs() < 1e-6 {
        0.5
    } else {
        ((ISO_LEVEL - da) / denom).clamp(0.0, 1.0)
    };
    let nx = g_a[0] + t * (g_b[0] - g_a[0]);
    let ny = g_a[1] + t * (g_b[1] - g_a[1]);
    let nz = g_a[2] + t * (g_b[2] - g_a[2]);
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-6 {
        // Both endpoint gradients vanished. Derive the outward normal
        // from the edge instead — solid endpoint toward air — which is
        // a real direction, so the winding check still fires on it.
        let (ca, cb) = (corners_local[a], corners_local[b]);
        let edge = [
            cb.0 as f32 - ca.0 as f32,
            cb.1 as f32 - ca.1 as f32,
            cb.2 as f32 - ca.2 as f32,
        ];
        let elen = (edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]).sqrt();
        if elen < 1e-6 {
            return [0.0, 1.0, 0.0]; // a == b (never a real edge) — stay finite
        }
        // da >= db → corner a is the solid side, so solid→air runs a→b.
        let sign = if da >= db { 1.0 } else { -1.0 };
        [
            sign * edge[0] / elen,
            sign * edge[1] / elen,
            sign * edge[2] / elen,
        ]
    } else {
        // Negative because the gradient of a "solid=1, air=0" field
        // points INTO the solid; we want the outward-facing normal.
        [-nx / len, -ny / len, -nz / len]
    }
}

/// Central-difference gradient of the density field at a sample point.
/// Out-of-range neighbors read as 0, which the padding layer makes rare.
fn sample_gradient(
    density: &[f32],
    size: (usize, usize, usize),
    idx: &dyn Fn(usize, usize, usize) -> usize,
    p: (usize, usize, usize),
) -> [f32; 3] {
    let sample = |x: i32, y: i32, z: i32| -> f32 {
        if x < 0
            || y < 0
            || z < 0
            || x as usize >= size.0
            || y as usize >= size.1
            || z as usize >= size.2
        {
            0.0
        } else {
            density[idx(x as usize, y as usize, z as usize)]
        }
    };
    let dx = sample(p.0 as i32 + 1, p.1 as i32, p.2 as i32)
        - sample(p.0 as i32 - 1, p.1 as i32, p.2 as i32);
    let dy = sample(p.0 as i32, p.1 as i32 + 1, p.2 as i32)
        - sample(p.0 as i32, p.1 as i32 - 1, p.2 as i32);
    let dz = sample(p.0 as i32, p.1 as i32, p.2 as i32 + 1)
        - sample(p.0 as i32, p.1 as i32, p.2 as i32 - 1);
    [dx, dy, dz]
}

/// Color for a vertex on the edge between corners `a` and `b`: the
/// average of whichever endpoint voxels are solid. After blurring both
/// can be empty, so the search widens to the 2×2×2 block before white.
fn edge_color(world: &World, a: [f32; 3], b: [f32; 3]) -> [f32; 4] {
    let ai = (
        a[0].round() as i32,
        a[1].round() as i32,
        a[2].round() as i32,
    );
    let bi = (
        b[0].round() as i32,
        b[1].round() as i32,
        b[2].round() as i32,
    );

    if let Some(c) = average_solid(world, [ai, bi].into_iter()) {
        return c;
    }

    // Blurred field: widen to the block spanned by the two endpoints
    // (2 cells along the edge axis, 2 on each perpendicular axis).
    let lo = (ai.0.min(bi.0), ai.1.min(bi.1), ai.2.min(bi.2));
    let block = (0..2).flat_map(move |dx| {
        (0..2).flat_map(move |dy| (0..2).map(move |dz| (lo.0 + dx, lo.1 + dy, lo.2 + dz)))
    });
    average_solid(world, block).unwrap_or([1.0, 1.0, 1.0, 1.0])
}

/// Mean color of the solid voxels among `positions`, or `None` if none
/// of them are solid.
fn average_solid(
    world: &World,
    positions: impl Iterator<Item = (i32, i32, i32)>,
) -> Option<[f32; 4]> {
    let mut sum = [0.0_f32; 4];
    let mut count = 0u32;
    for (x, y, z) in positions {
        let v: Voxel = world.get_voxel(x, y, z);
        if !v.is_air() {
            let c = v.color_f32();
            for i in 0..4 {
                sum[i] += c[i];
            }
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let n = count as f32;
    Some([sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n])
}

/// 3×3×3 box blur over a density field, turning 0/1 densities into a
/// continuous one. Boundary cells blur against zero: the in-bounds sum
/// is always divided by the full 27.
fn box_blur_3x3x3(input: &[f32], size: (usize, usize, usize)) -> Vec<f32> {
    let idx =
        |dx: usize, dy: usize, dz: usize| -> usize { dx + dy * size.0 + dz * size.0 * size.1 };
    let mut out = vec![0.0_f32; input.len()];
    for z in 0..size.2 {
        for y in 0..size.1 {
            for x in 0..size.0 {
                let mut sum = 0.0;
                for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let nz = z as i32 + dz;
                            if nx >= 0
                                && ny >= 0
                                && nz >= 0
                                && (nx as usize) < size.0
                                && (ny as usize) < size.1
                                && (nz as usize) < size.2
                            {
                                sum += input[idx(nx as usize, ny as usize, nz as usize)];
                            }
                        }
                    }
                }
                // Always divide by the full 27, treating out-of-bounds
                // neighbors as 0. The in-bounds count instead lets a
                // bottom-pad cell read 0.5 and erases the bottom face.
                out[idx(x, y, z)] = sum / 27.0;
            }
        }
    }
    out
}

/// Edge → (corner_a, corner_b) lookup. Same numbering as Paul
/// Bourke's MC reference: each cube has 12 edges, 4 around the
/// bottom, 4 around the top, and 4 vertical pillars.
const EDGE_VERTEX_PAIRS: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_empty_world_no_geometry() {
        let world = World::new();
        let mesh = mesh_world_smoothed(&world, true).expect("scene is small");
        assert!(mesh.is_empty());
    }

    #[test]
    fn test_single_voxel_produces_geometry() {
        // On raw 0/1 density a lone voxel has eight corner samples at
        // 1.0 against neighbors at 0.0, so MC closes a surface around
        // it — at least 12 triangles, the rounded-cube case.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, false).expect("scene is small");
        assert!(
            !mesh.is_empty(),
            "isolated voxel should still produce a closed surface"
        );
        assert!(
            mesh.triangle_count() >= 8,
            "expected at least an octahedron-ish surface"
        );
    }

    #[test]
    fn test_smoothing_reduces_or_keeps_surface_extent() {
        // Smoothing rounds off corners but shouldn't make a small
        // model balloon outward — count of triangles for a small
        // shape should stay bounded.
        let mut world = World::new();
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    world.set_voxel(x, y, z, Voxel::from_rgb(100, 100, 100));
                }
            }
        }
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, true).expect("scene is small");
        // Smoothed 3³ block is a roundish blob; expect non-zero,
        // bounded triangle count.
        assert!(!mesh.is_empty());
        assert!(
            mesh.triangle_count() < 1000,
            "too many triangles: {}",
            mesh.triangle_count()
        );
    }

    #[test]
    fn test_winding_outward_for_isolated_voxel() {
        // A flipped table or two swapped edge indices would export
        // inside-out into Blender or Unity. Build an isolated voxel and
        // assert every triangle's cross product points outward.
        let mut world = World::new();
        world.set_voxel(5, 5, 5, Voxel::from_rgb(200, 100, 50));
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, false).expect("scene is small");
        assert!(!mesh.is_empty(), "expected MC mesh for isolated voxel");

        // Ground truth is each triangle's vertex normals, not the voxel
        // center: near a corner the cell-outward direction and the
        // surface normal differ enough to report a false positive.
        let mut outward_count = 0;
        let mut inward_count = 0;
        let mut zero_count = 0;
        let tol = 1e-4_f32;
        for tri in 0..mesh.indices.len() / 3 {
            let i0 = mesh.indices[tri * 3] as usize;
            let i1 = mesh.indices[tri * 3 + 1] as usize;
            let i2 = mesh.indices[tri * 3 + 2] as usize;
            let v0 = mesh.vertices[i0].position;
            let v1 = mesh.vertices[i1].position;
            let v2 = mesh.vertices[i2].position;
            // Average vertex normal — represents the surface's
            // outward direction across the triangle.
            let avg_normal = [
                (mesh.vertices[i0].normal[0]
                    + mesh.vertices[i1].normal[0]
                    + mesh.vertices[i2].normal[0])
                    / 3.0,
                (mesh.vertices[i0].normal[1]
                    + mesh.vertices[i1].normal[1]
                    + mesh.vertices[i2].normal[1])
                    / 3.0,
                (mesh.vertices[i0].normal[2]
                    + mesh.vertices[i1].normal[2]
                    + mesh.vertices[i2].normal[2])
                    / 3.0,
            ];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let dot =
                cross[0] * avg_normal[0] + cross[1] * avg_normal[1] + cross[2] * avg_normal[2];
            if dot > tol {
                outward_count += 1;
            } else if dot < -tol {
                inward_count += 1;
            } else {
                zero_count += 1;
            }
        }
        let total = outward_count + inward_count + zero_count;
        eprintln!(
            "MC isolated-voxel winding (vs vertex normal): {} outward, {} inward, {} zero (out of {})",
            outward_count, inward_count, zero_count, total
        );
        assert_eq!(
            inward_count, 0,
            "MC winding correction failed: {} inward triangles vs vertex normal (expected 0); {} outward, {} zero out of {}",
            inward_count, outward_count, zero_count, total
        );
    }

    #[test]
    fn test_normals_are_unit_length() {
        let mut world = World::new();
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    world.set_voxel(x, y, z, Voxel::from_rgb(200, 100, 50));
                }
            }
        }
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, false).expect("scene is small");
        for v in &mesh.vertices {
            let len =
                (v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] + v.normal[2] * v.normal[2])
                    .sqrt();
            assert!(
                (len - 1.0).abs() < 1e-3,
                "non-unit normal: {:?} (length {})",
                v.normal,
                len
            );
        }
    }

    #[test]
    fn test_mc_mesh_uses_cell_convention_not_voxel_centered() {
        // A voxel at integer (2,2,2) must span the cell [2,3]³ after the
        // +0.5 shift — the "voxel n occupies [n, n+1)" convention the
        // rest of the app uses — so its center sits at 2.5, not 2.0.
        let mut world = World::new();
        world.set_voxel(2, 2, 2, Voxel::from_rgb(200, 100, 50));
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, false).expect("scene is small");
        assert!(!mesh.is_empty());
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for v in &mesh.vertices {
            for a in 0..3 {
                lo[a] = lo[a].min(v.position[a]);
                hi[a] = hi[a].max(v.position[a]);
            }
        }
        for a in 0..3 {
            let center = (lo[a] + hi[a]) * 0.5;
            assert!(
                (center - 2.5).abs() < 0.1,
                "axis {} surface center {} (expected ~2.5, the cell center of [2,3])",
                a,
                center
            );
        }
    }

    #[test]
    fn test_thin_wall_normals_stay_finite_and_unit() {
        // Two 1-cell walls with a 1-cell gap is where both endpoint
        // gradients vanish. The fallback must still give finite,
        // unit-length normals derived from the edge direction.
        let mut world = World::new();
        for y in 0..4 {
            for z in 0..4 {
                world.set_voxel(0, y, z, Voxel::from_rgb(180, 180, 180));
                world.set_voxel(2, y, z, Voxel::from_rgb(180, 180, 180));
            }
        }
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, false).expect("scene is small");
        assert!(!mesh.is_empty());
        for v in &mesh.vertices {
            assert!(
                v.normal.iter().all(|c| c.is_finite()),
                "non-finite normal {:?}",
                v.normal
            );
            let len =
                (v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] + v.normal[2] * v.normal[2])
                    .sqrt();
            assert!(
                (len - 1.0).abs() < 1e-3,
                "non-unit normal {:?} (len {})",
                v.normal,
                len
            );
        }
    }

    #[test]
    fn test_box_blur_divides_by_27_against_zero_padding() {
        // The blur always divides by the full 27, treating out-of-bounds
        // neighbors as 0. Dividing by the in-bounds count instead lets a
        // pad cell read 0.5 and dissolves the model's bottom face.
        let size = (5, 5, 5);
        let input = vec![1.0_f32; 125];
        let out = box_blur_3x3x3(&input, size);
        let at = |x: usize, y: usize, z: usize| out[x + y * size.0 + z * size.0 * size.1];
        // Interior: all 27 neighbors in-bounds → 27/27 = 1.0.
        assert!((at(2, 2, 2) - 1.0).abs() < 1e-6);
        // Face center: one plane out of bounds → 18 in-bounds.
        assert!((at(0, 2, 2) - 18.0 / 27.0).abs() < 1e-6);
        // Edge: two planes out → 12 in-bounds.
        assert!((at(0, 0, 2) - 12.0 / 27.0).abs() < 1e-6);
        // Corner: three planes out → 8 in-bounds.
        assert!((at(0, 0, 0) - 8.0 / 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_box_blur_smooths_step() {
        // A sharp 0/1 step should diffuse outward into the 0 region
        // by one cell on each side after the blur.
        let size = (5, 5, 5);
        let mut input = vec![0.0_f32; 125];
        // Set the center cell to 1.
        let i = |x: usize, y: usize, z: usize| -> usize { x + y * size.0 + z * size.0 * size.1 };
        input[i(2, 2, 2)] = 1.0;
        let out = box_blur_3x3x3(&input, size);
        // (2,2,2) and (1,2,2) both include the lone center → 1/27 each;
        // (0,2,2) doesn't reach it → 0.
        assert!(out[i(2, 2, 2)] > 0.0 && out[i(2, 2, 2)] < 1.0);
        assert!(out[i(1, 2, 2)] > 0.0);
        assert_eq!(out[i(0, 2, 2)], 0.0);
    }

    #[test]
    fn test_vertex_color_comes_from_the_voxel_the_surface_belongs_to() {
        // On the raw field every crossing edge runs from one solid voxel
        // to air, so each vertex takes that voxel's color exactly and
        // nothing invents a third one.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(1, 0, 0, Voxel::from_rgb(0, 0, 255));
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, false).expect("scene is small");
        assert!(!mesh.is_empty());

        let mut saw_red = false;
        let mut saw_blue = false;
        for v in &mesh.vertices {
            let c = [v.color[0], v.color[1], v.color[2]];
            let is_red = c[0] > 0.99 && c[1] < 0.01 && c[2] < 0.01;
            let is_blue = c[0] < 0.01 && c[1] < 0.01 && c[2] > 0.99;
            assert!(
                is_red || is_blue,
                "vertex color {c:?} matches neither source voxel"
            );
            saw_red |= is_red;
            saw_blue |= is_blue;
        }
        assert!(saw_red && saw_blue, "both voxels should contribute surface");
    }

    #[test]
    fn test_isolated_voxel_surface_is_entirely_its_own_color() {
        // Regression: a sampler reading only the edge's low-coordinate
        // side never contains the high endpoint, so −X/−Y/−Z faces of a
        // convex shape sample air and fall through to white.
        let mut world = World::new();
        world.set_voxel(5, 5, 5, Voxel::from_rgb(200, 100, 50));
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, false).expect("scene is small");
        assert!(!mesh.is_empty());

        let expected = Voxel::from_rgb(200, 100, 50).color_f32();
        for v in &mesh.vertices {
            for a in 0..3 {
                assert!(
                    (v.color[a] - expected[a]).abs() < 1e-3,
                    "vertex at {:?} has color {:?}, expected the voxel's own {:?}",
                    v.position,
                    v.color,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_smoothed_mesh_shares_vertices_between_triangles() {
        // Every MC vertex belongs to a cube edge and all its attributes
        // are properties of that edge, so cubes sharing an edge must
        // share the vertex rather than emit three per triangle.
        let mut world = World::new();
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    world.set_voxel(x, y, z, Voxel::from_rgb(120, 120, 120));
                }
            }
        }
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, false).expect("scene is small");
        assert!(mesh.triangle_count() > 0);
        assert!(
            mesh.vertex_count() < mesh.triangle_count() * 3,
            "no sharing: {} vertices for {} triangles",
            mesh.vertex_count(),
            mesh.triangle_count()
        );
        // Sharing must be exact, not approximate: no two vertices may
        // sit at the same position.
        let mut seen = HashSet::new();
        for v in &mesh.vertices {
            let key = (
                v.position[0].to_bits(),
                v.position[1].to_bits(),
                v.position[2].to_bits(),
            );
            assert!(seen.insert(key), "duplicate vertex at {:?}", v.position);
        }
    }

    #[test]
    fn test_far_apart_voxels_error_instead_of_aborting() {
        // The field is dense over the bounding box, so two stray voxels
        // far apart ask for gigabytes. A failed allocation aborts the
        // process, so this must be a clean error instead.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(1, 2, 3));
        world.set_voxel(2000, 2000, 2000, Voxel::from_rgb(1, 2, 3));
        world.clear_dirty_flags();
        assert!(matches!(
            mesh_world_smoothed(&world, false),
            Err(SmoothMeshError::SceneTooLarge { .. })
        ));

        // A far-apart pair on a *single* axis is only a thin slab, and
        // must still succeed.
        let mut thin = World::new();
        thin.set_voxel(0, 0, 0, Voxel::from_rgb(1, 2, 3));
        thin.set_voxel(2000, 0, 0, Voxel::from_rgb(1, 2, 3));
        thin.clear_dirty_flags();
        assert!(mesh_world_smoothed(&thin, false).is_ok());
    }
}
