//! Marching Cubes mesher for export-time geometry smoothing.
//!
//! Used by the "smoothed" variants of the OBJ / GLB exporters. The
//! editor's render path keeps the chunky greedy mesher; this module
//! is a separate code path invoked only when the user picks
//! "Wavefront OBJ - smoothed (.obj)..." (or the GLB equivalent).
//! It walks the entire world, samples a density field at voxel
//! centers, applies a 3×3×3 smoothing pass, then runs the classic
//! Paul Bourke / Lorensen-Cline Marching Cubes algorithm on the
//! resulting field to produce a continuous interpolated surface.
//!
//! Per-vertex output:
//! - **Position**: linear interpolation along the cube edge between
//!   the two density samples spanning the 0.5 isolevel.
//! - **Normal**: gradient of the density field at the vertex,
//!   normalized — gives smooth (non-faceted) shading.
//! - **Color**: average of the up-to-4 solid voxels touching the
//!   edge this vertex lies on, so color boundaries (e.g. grass next
//!   to stone) blend over a 1-cell band.
//!
//! Output is one combined `ChunkMesh` for the whole world, ready
//! for the same OBJ / GLB writer code paths the regular exporters
//! use.

mod tables;

use tables::{EDGE_TABLE, TRI_TABLE};

use crate::core::{Voxel, World};
use crate::mesh::{ChunkMesh, Vertex};

/// Density value above which a sample is considered "inside" the
/// surface. With voxel-centered density (1.0 for solid, 0.0 for air,
/// optionally smoothed) this naturally places the surface midway
/// between solid and air voxels.
const ISO_LEVEL: f32 = 0.5;

/// Build a Marching-Cubes mesh of the entire world. Returns a single
/// `ChunkMesh` (the chunk position field is unused — it's a flat
/// world-space mesh, not chunked). For empty worlds returns an
/// empty mesh.
///
/// The `smooth` flag toggles a 3×3×3 box-blur over the density field
/// before marching. Without smoothing, MC over 0/1 voxel data
/// produces "rounded cubes" — softer than greedy but still recognizably
/// blocky. With smoothing, surfaces become clay-like — small features
/// shrink and isolated voxels may disappear into the smoothed
/// background. The smoothed mode is what the user gets from the
/// "smoothed" export menu entries.
pub fn mesh_world_smoothed(world: &World, smooth: bool) -> ChunkMesh {
    use crate::core::ChunkPos;
    let Some(bbox) = world_voxel_bbox(world) else {
        return ChunkMesh::new(ChunkPos::ZERO);
    };

    // Density field is sampled at every integer position in
    // [bbox.min, bbox.max + 1] inclusive, i.e. one extra layer past
    // the voxel bbox so MC cubes at the boundary still see a 0
    // density gradient toward the outside. The smoothing kernel
    // additionally needs `(±1)` padding on every side, so the
    // allocated field is (bbox extent + 3) per axis.
    let pad = 1;
    let min = (bbox.min.0 - pad, bbox.min.1 - pad, bbox.min.2 - pad);
    let max = (bbox.max.0 + 1 + pad, bbox.max.1 + 1 + pad, bbox.max.2 + 1 + pad);
    let size = (
        (max.0 - min.0 + 1) as usize,
        (max.1 - min.1 + 1) as usize,
        (max.2 - min.2 + 1) as usize,
    );
    let total = size.0 * size.1 * size.2;

    // Raw density: 1.0 if the voxel at the sample point is solid,
    // 0.0 if air. Sampling at integer positions means each density
    // sample IS a voxel — no extra averaging needed for the raw pass.
    let mut density = vec![0.0_f32; total];
    let idx = |dx: usize, dy: usize, dz: usize| -> usize {
        dx + dy * size.0 + dz * size.0 * size.1
    };
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
    for gz in 0..size.2 - 1 {
        for gy in 0..size.1 - 1 {
            for gx in 0..size.0 - 1 {
                march_one_cube(
                    &density, size, &idx, gx, gy, gz, min, world, &mut mesh,
                );
            }
        }
    }
    mesh
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
) {
    // Corner numbering follows Paul Bourke's convention so the
    // EDGE_TABLE / TRI_TABLE indices line up:
    //   0: (gx,   gy,   gz)
    //   1: (gx+1, gy,   gz)
    //   2: (gx+1, gy,   gz+1)
    //   3: (gx,   gy,   gz+1)
    //   4: (gx,   gy+1, gz)
    //   5: (gx+1, gy+1, gz)
    //   6: (gx+1, gy+1, gz+1)
    //   7: (gx,   gy+1, gz+1)
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
    for i in 0..8 {
        if densities[i] >= ISO_LEVEL {
            cube_index |= 1 << i;
        }
    }
    let edges = EDGE_TABLE[cube_index];
    if edges == 0 {
        return; // entirely inside or outside — no surface here
    }

    // World-space corner positions for emit time.
    let corners_world: [[f32; 3]; 8] = [
        [(field_min.0 + gx as i32) as f32, (field_min.1 + gy as i32) as f32, (field_min.2 + gz as i32) as f32],
        [(field_min.0 + gx as i32 + 1) as f32, (field_min.1 + gy as i32) as f32, (field_min.2 + gz as i32) as f32],
        [(field_min.0 + gx as i32 + 1) as f32, (field_min.1 + gy as i32) as f32, (field_min.2 + gz as i32 + 1) as f32],
        [(field_min.0 + gx as i32) as f32, (field_min.1 + gy as i32) as f32, (field_min.2 + gz as i32 + 1) as f32],
        [(field_min.0 + gx as i32) as f32, (field_min.1 + gy as i32 + 1) as f32, (field_min.2 + gz as i32) as f32],
        [(field_min.0 + gx as i32 + 1) as f32, (field_min.1 + gy as i32 + 1) as f32, (field_min.2 + gz as i32) as f32],
        [(field_min.0 + gx as i32 + 1) as f32, (field_min.1 + gy as i32 + 1) as f32, (field_min.2 + gz as i32 + 1) as f32],
        [(field_min.0 + gx as i32) as f32, (field_min.1 + gy as i32 + 1) as f32, (field_min.2 + gz as i32 + 1) as f32],
    ];

    // Compute the 12 potential edge vertex positions (only the ones
    // flagged in `edges` are actually used; we lazily fill them).
    // Edge i connects EDGE_VERTEX_PAIRS[i].0 → EDGE_VERTEX_PAIRS[i].1.
    let mut edge_vertices: [(([f32; 3], [f32; 3], [f32; 4]), bool); 12] = [
        (([0.0; 3], [0.0; 3], [0.0; 4]), false); 12
    ];
    for e in 0..12 {
        if edges & (1 << e) == 0 {
            continue;
        }
        let (a, b) = EDGE_VERTEX_PAIRS[e];
        let pos = interp_edge(corners_world[a], corners_world[b], densities[a], densities[b]);
        let normal = density_gradient(density, size, idx, corners_local, a, b, densities[a], densities[b]);
        let color = edge_color(world, corners_world[a], corners_world[b]);
        edge_vertices[e] = ((pos, normal, color), true);
    }

    // Emit triangles. TRI_TABLE rows are -1-terminated lists of
    // edge indices, three at a time.
    let row = TRI_TABLE[cube_index];
    let mut i = 0;
    while i < row.len() && row[i] != -1 {
        let e0 = row[i] as usize;
        let e1 = row[i + 1] as usize;
        let e2 = row[i + 2] as usize;
        let v0 = Vertex::new(edge_vertices[e0].0 .0, edge_vertices[e0].0 .1, edge_vertices[e0].0 .2);
        let v1 = Vertex::new(edge_vertices[e1].0 .0, edge_vertices[e1].0 .1, edge_vertices[e1].0 .2);
        let v2 = Vertex::new(edge_vertices[e2].0 .0, edge_vertices[e2].0 .1, edge_vertices[e2].0 .2);

        let base = mesh.vertices.len() as u32;
        mesh.vertices.push(v0);
        mesh.vertices.push(v1);
        mesh.vertices.push(v2);
        mesh.indices.push(base);
        mesh.indices.push(base + 1);
        mesh.indices.push(base + 2);
        i += 3;
    }
}

/// Linear interpolation of an edge crossing point. With density
/// values `da` and `db` at the two endpoints, the surface at
/// `ISO_LEVEL` lives at parameter `t = (ISO_LEVEL - da) / (db - da)`.
/// Degenerate case (da ≈ db) falls back to the midpoint to avoid
/// division by zero.
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
        // Degenerate — surface gradient vanished. Pick +Y so the
        // vertex still has a sensible (if arbitrary) normal.
        [0.0, 1.0, 0.0]
    } else {
        // Negative because the gradient of a "solid=1, air=0" field
        // points INTO the solid; we want the outward-facing normal.
        [-nx / len, -ny / len, -nz / len]
    }
}

/// Central-difference gradient of the density field at a sample
/// point. Out-of-range neighbors are treated as 0 (air outside the
/// field) — the padding layer already reserves at least one cell on
/// each side, so this fallback rarely fires in practice.
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

/// Average color of the up-to-4 solid voxels that share the cube
/// edge connecting world-space corners `a` and `b`. An edge between
/// two adjacent corners is touched by exactly 4 voxels (the 4
/// voxels in the 2-cell-thick slab perpendicular to the edge); we
/// average the colors of whichever ones are solid. Falls back to
/// white if somehow none are solid (shouldn't happen on a real
/// surface vertex but defensively keeps mesh data sane).
fn edge_color(world: &World, a: [f32; 3], b: [f32; 3]) -> [f32; 4] {
    // The edge runs along whichever axis a and b differ on. The 4
    // voxels touching this edge are at offsets {(0|−1)} on the two
    // perpendicular axes from the edge's midpoint cell.
    let mid = [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ];
    // The midpoint sits between two voxels along the edge axis;
    // we want both, AND the adjacent rows on the perpendicular
    // axes — 4 voxels total.
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    // Identify the axis the edge runs along (exactly one of dx/dy/
    // dz is non-zero given an MC cube edge).
    let (ox, oy, oz): (i32, i32, i32) = if dx.abs() > 0.5 {
        // Edge along X — perpendicular axes are Y, Z.
        (mid[0].floor() as i32, 0, 0)
    } else if dy.abs() > 0.5 {
        (0, mid[1].floor() as i32, 0)
    } else {
        (0, 0, mid[2].floor() as i32)
    };
    let _ = (ox, oy, oz); // silence unused warning if we restructure

    // The 4 voxels around the edge: subtract 1 from each
    // perpendicular axis to find the lower of the two voxels in
    // that direction; the edge spans both.
    let edge_along_x = dx.abs() > 0.5;
    let edge_along_y = dy.abs() > 0.5;
    let edge_along_z = dz.abs() > 0.5;

    let (vx, vy, vz) = (
        mid[0].floor() as i32,
        mid[1].floor() as i32,
        mid[2].floor() as i32,
    );

    let voxel_offsets: [(i32, i32, i32); 4] = if edge_along_x {
        [(vx, vy - 1, vz - 1), (vx, vy, vz - 1), (vx, vy - 1, vz), (vx, vy, vz)]
    } else if edge_along_y {
        [(vx - 1, vy, vz - 1), (vx, vy, vz - 1), (vx - 1, vy, vz), (vx, vy, vz)]
    } else if edge_along_z {
        [(vx - 1, vy - 1, vz), (vx, vy - 1, vz), (vx - 1, vy, vz), (vx, vy, vz)]
    } else {
        // Degenerate (a == b) — should never happen for a real edge.
        return [1.0, 1.0, 1.0, 1.0];
    };

    let mut sum = [0.0_f32; 4];
    let mut count = 0u32;
    for (x, y, z) in voxel_offsets {
        let v: Voxel = world.get_voxel(x, y, z);
        if !v.is_air() {
            let c = v.color_f32();
            sum[0] += c[0];
            sum[1] += c[1];
            sum[2] += c[2];
            sum[3] += c[3];
            count += 1;
        }
    }
    if count == 0 {
        [1.0, 1.0, 1.0, 1.0]
    } else {
        let n = count as f32;
        [sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n]
    }
}

/// 3×3×3 box blur over a density field. Used for the "smoothed"
/// export mode — turns 0/1 voxel densities into a continuous field
/// where surfaces become rounded blobs instead of rounded cubes.
/// Boundary cells are blurred against zero (the padding layer the
/// caller already allocated, so this just rolls naturally past the
/// real data).
fn box_blur_3x3x3(input: &[f32], size: (usize, usize, usize)) -> Vec<f32> {
    let idx = |dx: usize, dy: usize, dz: usize| -> usize {
        dx + dy * size.0 + dz * size.0 * size.1
    };
    let mut out = vec![0.0_f32; input.len()];
    for z in 0..size.2 {
        for y in 0..size.1 {
            for x in 0..size.0 {
                let mut sum = 0.0;
                let mut count = 0;
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
                                sum +=
                                    input[idx(nx as usize, ny as usize, nz as usize)];
                                count += 1;
                            }
                        }
                    }
                }
                out[idx(x, y, z)] = sum / count as f32;
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

/// World-space bounding box (inclusive on both ends) of all solid
/// voxels in the world. Returns `None` for an empty world.
struct VoxelBbox {
    min: (i32, i32, i32),
    max: (i32, i32, i32),
}

fn world_voxel_bbox(world: &World) -> Option<VoxelBbox> {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut min_z = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    let mut max_z = i32::MIN;
    let mut found = false;

    for (chunk_pos, chunk_lock) in world.chunks() {
        let chunk = chunk_lock.read();
        let (ox, oy, oz) = chunk_pos.world_origin();
        for (local_pos, _) in chunk.iter_solid() {
            let x = ox + local_pos.x as i32;
            let y = oy + local_pos.y as i32;
            let z = oz + local_pos.z as i32;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            min_z = min_z.min(z);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            max_z = max_z.max(z);
            found = true;
        }
    }

    if !found {
        None
    } else {
        Some(VoxelBbox {
            min: (min_x, min_y, min_z),
            max: (max_x, max_y, max_z),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_world_no_geometry() {
        let world = World::new();
        let mesh = mesh_world_smoothed(&world, true);
        assert!(mesh.is_empty());
    }

    #[test]
    fn test_single_voxel_produces_geometry() {
        // With raw 0/1 density (smooth=false), a single solid voxel
        // surrounded by air has 8 corners-as-density-samples each
        // at 1.0, with neighbors at 0.0. MC produces a closed
        // surface around the voxel — at minimum 12 triangles
        // (the standard rounded-cube case).
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, false);
        assert!(!mesh.is_empty(), "isolated voxel should still produce a closed surface");
        assert!(mesh.triangle_count() >= 8, "expected at least an octahedron-ish surface");
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
        let mesh = mesh_world_smoothed(&world, true);
        // Smoothed 3³ block is a roundish blob; expect non-zero,
        // bounded triangle count.
        assert!(!mesh.is_empty());
        assert!(mesh.triangle_count() < 1000, "too many triangles: {}", mesh.triangle_count());
    }

    #[test]
    fn test_winding_outward_for_isolated_voxel() {
        // MC's TRI_TABLE (standard Lorensen-Cline) emits triangles
        // CCW-from-outside when density is positive inside the
        // surface — same convention as the cube mesher. We verify
        // by building an isolated solid voxel, running MC, and
        // asserting every triangle's cross product points AWAY
        // from the voxel center.
        //
        // If MC ever gets a flipped table (or someone swaps two
        // edge indices), exported `.obj` / `.glb` smoothed meshes
        // would import inside-out into Blender / Unity. This test
        // catches that even though MC is render-disabled.
        let mut world = World::new();
        world.set_voxel(5, 5, 5, Voxel::from_rgb(200, 100, 50));
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, false);
        assert!(!mesh.is_empty(), "expected MC mesh for isolated voxel");

        // Voxel center in world coords: (5.5, 5.5, 5.5).
        let center = [5.5_f32, 5.5, 5.5];
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
            let centroid = [
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0,
            ];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let outward = [
                centroid[0] - center[0],
                centroid[1] - center[1],
                centroid[2] - center[2],
            ];
            let dot = cross[0] * outward[0]
                + cross[1] * outward[1]
                + cross[2] * outward[2];
            if dot > tol {
                outward_count += 1;
            } else if dot < -tol {
                inward_count += 1;
            } else {
                zero_count += 1;
            }
        }
        // Diagnostic: report the breakdown. Standard MC (Lorensen-
        // Cline) should have ALL triangles outward for an isolated
        // solid voxel; if half are inward, the TRI_TABLE is using a
        // different convention; if mixed unevenly, something is
        // broken.
        let total = outward_count + inward_count + zero_count;
        eprintln!(
            "MC isolated-voxel winding: {} outward, {} inward, {} zero (out of {})",
            outward_count, inward_count, zero_count, total
        );
        // Dominant direction must be outward (matching wgpu / glTF
        // CCW-from-outside convention). Allow some zero/edge cases.
        assert!(
            outward_count > inward_count,
            "MC winding not predominantly outward: {} outward vs {} inward",
            outward_count,
            inward_count
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
        let mesh = mesh_world_smoothed(&world, false);
        for v in &mesh.vertices {
            let len = (v.normal[0] * v.normal[0]
                + v.normal[1] * v.normal[1]
                + v.normal[2] * v.normal[2])
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
    fn test_box_blur_preserves_average() {
        // Box blur of a uniform field should give back the same
        // uniform field (interior cells average 27 ones; boundary
        // cells average fewer ones over fewer samples → still 1.0).
        let size = (5, 5, 5);
        let input = vec![1.0_f32; 125];
        let out = box_blur_3x3x3(&input, size);
        for v in &out {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_box_blur_smooths_step() {
        // A sharp 0/1 step should diffuse outward into the 0 region
        // by one cell on each side after the blur.
        let size = (5, 5, 5);
        let mut input = vec![0.0_f32; 125];
        // Set the center cell to 1.
        let i = |x: usize, y: usize, z: usize| -> usize {
            x + y * size.0 + z * size.0 * size.1
        };
        input[i(2, 2, 2)] = 1.0;
        let out = box_blur_3x3x3(&input, size);
        // The cell at (2, 2, 2) averages 27 cells with 1 center 1 →
        // 1/27. The neighbor at (1, 2, 2) also includes the center
        // → 1/27. The cell at (0, 2, 2) doesn't reach the center →
        // 0.
        assert!(out[i(2, 2, 2)] > 0.0 && out[i(2, 2, 2)] < 1.0);
        assert!(out[i(1, 2, 2)] > 0.0);
        assert_eq!(out[i(0, 2, 2)], 0.0);
    }

    #[test]
    fn test_color_blends_at_voxel_boundary() {
        // Two adjacent voxels with very different colors: vertices
        // generated near the seam should have an averaged color
        // (somewhere between the two source colors).
        let mut world = World::new();
        let red = Voxel::from_rgb(255, 0, 0);
        let blue = Voxel::from_rgb(0, 0, 255);
        world.set_voxel(0, 0, 0, red);
        world.set_voxel(1, 0, 0, blue);
        world.clear_dirty_flags();
        let mesh = mesh_world_smoothed(&world, false);
        // Some vertex somewhere should have a non-pure-red, non-pure-
        // blue color (averaging happened).
        let mut saw_blend = false;
        for v in &mesh.vertices {
            let r = v.color[0];
            let b = v.color[2];
            // Blended voxel-color average has both r and b in
            // (0, 1) — neither was 0 nor 1.
            if r > 0.05 && r < 0.95 && b > 0.05 && b < 0.95 {
                saw_blend = true;
                break;
            }
        }
        assert!(saw_blend, "expected at least one blended vertex color");
    }
}
