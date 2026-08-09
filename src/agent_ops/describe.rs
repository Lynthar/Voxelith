//! Feedback: what the document contains, and what one plane of it
//! looks like.
//!
//! An agent editing blind needs a cheap way to check its own work. Two
//! views, both text: [`describe`] is the summary (how much, how big,
//! what colors), [`slice`] is one plane as ASCII art, which is what
//! actually catches "the door is one cell too high".

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::core::World;
use crate::editor::{Selection, Socket};
use crate::procgen::PipelineGraph;

use super::schema::{Aabb, AxisSpec};
use super::{ErrorCode, OpsError};

/// Distinct colors reported exactly; the rest are summarized. Sixteen
/// covers a hand-built palette without letting a photo-derived import
/// dump thousands of near-duplicates into the report.
const TOP_COLORS: usize = 16;

/// Side limit for one slice. A wider view stops being readable long
/// before it stops being cheap.
const MAX_SLICE_SIDE: i32 = 128;

/// Summary of the document.
#[derive(Debug, Clone, Serialize)]
pub struct Description {
    pub voxel_count: u64,
    pub chunk_count: usize,
    pub world_aabb: Option<Aabb>,
    /// `world_aabb` as extents, since "how big is it" is the question
    /// the AABB is usually standing in for.
    ///
    /// `i64` because it is a *difference* of two coordinates: everything
    /// this tool writes stays inside ±[`MAX_COORD`](super::MAX_COORD),
    /// but a `.vxlt` is an external file, and a span wider than `i32`
    /// used to overflow here rather than be reported. The extra width
    /// costs nothing on the wire — every real document's extents fit in
    /// three digits.
    pub size: Option<[i64; 3]>,
    /// The most common colors, most first.
    pub colors: Vec<ColorCount>,
    /// Voxels whose color didn't make the list, and how many distinct
    /// colors that covers.
    pub other_color_voxels: u64,
    pub other_color_kinds: usize,
    pub emissive: u64,
    pub metallic: u64,
    /// Voxel count per tint zone, indexed 0..=3.
    pub tint_zones: [u64; 4],
    pub sockets: Vec<SocketInfo>,
    pub selection: Option<Aabb>,
    pub undo_depth: usize,
    pub redo_depth: usize,
    /// Deterministic structural measurements, or `null` when the
    /// document is too big to measure cheaply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structure: Option<Structure>,
    /// The document's pipeline graph, whole, when it has one.
    ///
    /// Whole rather than summarized because it is small (a graph is
    /// capped at 64 nodes) and because reading it back is what makes
    /// editing one possible: an agent can take this, change a parameter
    /// or a wire, and send it straight back as a `graph` op.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<PipelineGraph>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColorCount {
    pub rgb: [u8; 3],
    pub count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SocketInfo {
    pub name: String,
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Which plane to render, and how.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub struct SliceRequest {
    /// Axis the plane is perpendicular to. `y` (the default) is the
    /// top-down view.
    #[serde(default)]
    pub axis: AxisSpec,
    /// Coordinate of the plane along `axis`.
    pub index: i32,
    /// Window to render. Defaults to the whole scene.
    #[serde(default)]
    pub region: Option<Aabb>,
    #[serde(default)]
    pub mode: SliceMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub enum SliceMode {
    /// `#` solid, `.` empty.
    #[default]
    Solid,
    /// A letter per color, with a legend.
    Color,
}

/// Loose parts listed individually. Past a handful the list stops
/// telling an agent anything the count didn't.
const MAX_LOOSE_PARTS: usize = 8;

/// Bounding-box cells above which the air sweep is skipped.
///
/// Unlike the solid passes, this one's cost tracks the *bounding box*,
/// not the model: two voxels a thousand cells apart are a cheap model
/// and a billion-cell box. Bounded separately for that reason.
const MAX_AIR_CELLS: u64 = 8_000_000;

/// Voxels above which the structural pass is skipped.
///
/// It is three linear passes with hash lookups per neighbor, which is
/// nothing on a model and real work on a scene. `describe` is answered
/// on the editor's main thread when an agent is driving the in-editor
/// bridge, so a document big enough to stutter it reports `null` and
/// says why rather than freezing the person watching.
const MAX_STRUCTURE_VOXELS: u64 = 2_000_000;

/// Deterministic measurements of the shape itself — what an agent can't
/// reliably read off a rendered view.
///
/// Every number here is a measurement, not a verdict. Three connected
/// components is wrong for a sword and right for a forest; a floating
/// part is a bug in a chair and the whole point of a tree canopy. The
/// judgment belongs to whoever asked for the model. What this removes
/// is the guessing: "are those two towers actually joined" has an exact
/// answer, and no isometric render reliably gives it.
#[derive(Debug, Clone, Serialize)]
pub struct Structure {
    /// Connected components under 6-connectivity (faces only — two
    /// voxels touching at a corner are not joined, which is what the
    /// mesher, the flood fill and a physical print all agree on).
    pub components: usize,
    /// Voxels in the largest component.
    pub largest_component: u64,
    /// The rest, biggest first, capped at eight.
    pub loose_parts: Vec<LoosePart>,
    /// Components that don't reach the scene's lowest layer. The
    /// "floating island" of print preparation, and the usual shape of
    /// "I meant to attach that".
    pub floating_components: usize,
    /// Solid voxels whose six face neighbors are all solid: the
    /// interior `hollow` would remove, and dead weight in an export.
    pub enclosed: u64,
    /// Mirror mismatch per axis, measured across the scene bounding
    /// box's own midplane.
    pub symmetry: [SymmetryCheck; 3],
    /// Solid cells resting on the lowest layer of the scene — how much
    /// of the model actually touches the ground.
    ///
    /// What tells an arch from a wall. Both can be one connected piece
    /// with nothing floating and the same bounding box; the arch stands
    /// on two piers and the wall stands on everything.
    pub footprint: u64,
    /// Air the model completely encloses: sealed rooms, and bubbles
    /// that would ship inside an exported mesh. `None` when the
    /// bounding box is too large to sweep.
    ///
    /// Distinct from `enclosed`, which counts *solid* cells with no
    /// exposed face. A hollow crate has no enclosed solids and one
    /// cavity; a solid crate has enclosed solids and no cavity. And
    /// distinct again from open space like the gap under an arch, which
    /// reaches the outside and is therefore not a cavity at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cavities: Option<Cavities>,
}

/// Enclosed air, as counts.
#[derive(Debug, Clone, Serialize)]
pub struct Cavities {
    pub count: usize,
    pub voxels: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoosePart {
    pub voxels: u64,
    pub aabb: Aabb,
}

/// How close the model is to symmetric about one axis.
///
/// Cell semantics, not point semantics: a voxel is the cell `[p, p+1)`,
/// so the reflection of `p` across a box is `min + max - p`. Using the
/// point formula would report every even-sized model as asymmetric by
/// its whole width — the same off-by-one `io::vox::rotate_cell` exists
/// to avoid.
#[derive(Debug, Clone, Serialize)]
pub struct SymmetryCheck {
    pub axis: &'static str,
    /// Solid voxels whose mirror image is not also solid.
    pub mismatched: u64,
    /// `mismatched` as a fraction of the model, rounded to 4 places.
    /// 0.0 is exact symmetry.
    pub ratio: f64,
}

/// Measure a world. `None` when it is too big to do cheaply — the
/// answer says so rather than making the caller wait.
pub fn structure(world: &World, voxel_count: u64) -> Option<Structure> {
    if voxel_count == 0 || voxel_count > MAX_STRUCTURE_VOXELS {
        return None;
    }
    let solids: HashSet<(i32, i32, i32)> = world
        .chunks()
        .flat_map(|(pos, chunk)| {
            let base = pos.world_origin();
            chunk
                .read()
                .iter_solid()
                .map(|(local, _)| {
                    (
                        base.0 + local.x as i32,
                        base.1 + local.y as i32,
                        base.2 + local.z as i32,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let (min, max) = world.scene_aabb()?;

    let mut components = components_of(&solids);
    // Biggest first, then by position so equal-sized parts don't
    // reorder between runs (the set iteration is a HashSet). Both
    // corners, because two parts can share a `min` — an L and its
    // mirror image do — and a key that ties is a key that lets the
    // hash order through.
    components.sort_unstable_by(|a, b| {
        b.voxels
            .cmp(&a.voxels)
            .then(a.aabb.min.cmp(&b.aabb.min))
            .then(a.aabb.max.cmp(&b.aabb.max))
    });
    let component_count = components.len();
    let largest_component = components.first().map(|c| c.voxels).unwrap_or(0);
    let floating_components = components
        .iter()
        .filter(|part| part.aabb.min[1] > min.1)
        .count();
    // The list is capped; the count is not. A model in 300 pieces has
    // to say 300, or the cap turns a disaster into a tidy-looking eight.
    let loose_parts: Vec<LoosePart> = components
        .into_iter()
        .skip(1)
        .take(MAX_LOOSE_PARTS)
        .collect();

    let enclosed = solids
        .iter()
        .filter(|&&pos| super::compile::is_enclosed(world, pos))
        .count() as u64;

    let axes = [("x", 0usize), ("y", 1), ("z", 2)];
    let bounds = [(min.0, max.0), (min.1, max.1), (min.2, max.2)];
    let symmetry = axes.map(|(name, index)| {
        let (low, high) = bounds[index];
        // `min + max - p` in i64. A `.vxlt` is an external file and can
        // hold coordinates the ops path would have refused, and two of
        // them added together overflowed i32 here — in the editor, on
        // the thread that draws. A reflection that lands outside i32
        // names no cell, so it counts as a mismatch, which is the same
        // answer as landing on air.
        let reflect = |p: i32| i32::try_from(low as i64 + high as i64 - p as i64).ok();
        let mismatched = solids
            .iter()
            .filter(|&&pos| {
                let mut mirrored = [pos.0, pos.1, pos.2];
                match reflect(mirrored[index]) {
                    Some(reflected) => {
                        mirrored[index] = reflected;
                        !solids.contains(&(mirrored[0], mirrored[1], mirrored[2]))
                    }
                    None => true,
                }
            })
            .count() as u64;
        SymmetryCheck {
            axis: name,
            mismatched,
            // Six places, not four: the structural pass runs on up to
            // two million voxels, and at four a single mismatched cell
            // in more than ten thousand rounds to `0.0` — a number an
            // agent reads as "symmetric" when `mismatched` says
            // otherwise two lines up. Six covers the whole range this
            // pass will ever measure.
            ratio: ((mismatched as f64 / voxel_count as f64) * 1_000_000.0).round()
                / 1_000_000.0,
        }
    });

    let footprint = solids.iter().filter(|pos| pos.1 == min.1).count() as u64;
    let cavities = sealed_air(&solids, min, max);

    Some(Structure {
        components: component_count,
        largest_component,
        loose_parts,
        floating_components,
        enclosed,
        symmetry,
        footprint,
        cavities,
    })
}

/// Air inside the bounding box that never reaches its surface.
///
/// The sweep runs inside the box only, so "reaches the outside" is
/// "reaches a face of the box" — the gap under an arch qualifies and is
/// correctly not a cavity, while a sealed room does not and is.
///
/// `None` when the box is too big to walk: the cost here is the box,
/// not the model, so a sparse scene spanning a thousand cells is
/// expensive in a way its voxel count never shows.
fn sealed_air(
    solids: &HashSet<(i32, i32, i32)>,
    min: (i32, i32, i32),
    max: (i32, i32, i32),
) -> Option<Cavities> {
    let span = |lo: i32, hi: i32| (hi as i64 - lo as i64 + 1).max(0) as u64;
    let volume = span(min.0, max.0)
        .saturating_mul(span(min.1, max.1))
        .saturating_mul(span(min.2, max.2));
    if volume > MAX_AIR_CELLS {
        return None;
    }

    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut queue: VecDeque<(i32, i32, i32)> = VecDeque::new();
    let mut count = 0usize;
    let mut voxels = 0u64;

    for x in min.0..=max.0 {
        for y in min.1..=max.1 {
            for z in min.2..=max.2 {
                let start = (x, y, z);
                if solids.contains(&start) || !seen.insert(start) {
                    continue;
                }
                let mut size = 0u64;
                let mut escapes = false;
                queue.push_back(start);
                while let Some(pos) = queue.pop_front() {
                    size += 1;
                    escapes |= pos.0 == min.0
                        || pos.0 == max.0
                        || pos.1 == min.1
                        || pos.1 == max.1
                        || pos.2 == min.2
                        || pos.2 == max.2;
                    for delta in super::compile::FACE_NEIGHBORS {
                        let Some(next) = super::compile::face_neighbor(pos, delta) else {
                            continue;
                        };
                        let inside = (min.0..=max.0).contains(&next.0)
                            && (min.1..=max.1).contains(&next.1)
                            && (min.2..=max.2).contains(&next.2);
                        if inside && !solids.contains(&next) && seen.insert(next) {
                            queue.push_back(next);
                        }
                    }
                }
                if !escapes {
                    count += 1;
                    voxels += size;
                }
            }
        }
    }
    Some(Cavities { count, voxels })
}

/// Flood fill every solid cell into components, 6-connected.
fn components_of(solids: &HashSet<(i32, i32, i32)>) -> Vec<LoosePart> {
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::with_capacity(solids.len());
    let mut parts = Vec::new();
    let mut queue: VecDeque<(i32, i32, i32)> = VecDeque::new();

    for &start in solids {
        if !seen.insert(start) {
            continue;
        }
        let mut voxels = 0u64;
        let mut min = start;
        let mut max = start;
        queue.push_back(start);
        while let Some(pos) = queue.pop_front() {
            voxels += 1;
            min = (min.0.min(pos.0), min.1.min(pos.1), min.2.min(pos.2));
            max = (max.0.max(pos.0), max.1.max(pos.1), max.2.max(pos.2));
            for delta in super::compile::FACE_NEIGHBORS {
                let Some(next) = super::compile::face_neighbor(pos, delta) else {
                    continue;
                };
                if solids.contains(&next) && seen.insert(next) {
                    queue.push_back(next);
                }
            }
        }
        parts.push(LoosePart {
            voxels,
            aabb: Aabb::from_pair((min, max)),
        });
    }
    parts
}

pub(super) fn solid_voxel_count(world: &World) -> u64 {
    world
        .chunks()
        .map(|(_, chunk)| chunk.read().solid_count() as u64)
        .sum()
}

/// The parts of a document a description is built from, borrowed from
/// whoever owns them.
///
/// [`AgentSession`](super::AgentSession) keeps all four together; the
/// editor keeps them on three different structs, because a selection and
/// an undo stack were its own long before an agent had any use for them.
/// Describing takes a view rather than a session so neither host has to
/// pretend to be the other — the same reason
/// [`run_batch`](super::run_batch) takes a world instead of owning one.
pub struct DocumentView<'a> {
    pub world: &'a World,
    pub selection: Option<Selection>,
    pub sockets: &'a [Socket],
    pub undo_depth: usize,
    pub redo_depth: usize,
    pub graph: &'a PipelineGraph,
}

pub fn describe(view: DocumentView<'_>) -> Description {
    let world = view.world;
    let mut histogram: HashMap<[u8; 3], u64> = HashMap::new();
    let mut voxel_count = 0u64;
    let mut emissive = 0u64;
    let mut metallic = 0u64;
    let mut tint_zones = [0u64; 4];

    for (_, chunk) in world.chunks() {
        for (_, voxel) in chunk.read().iter_solid() {
            voxel_count += 1;
            *histogram.entry([voxel.r, voxel.g, voxel.b]).or_insert(0) += 1;
            if voxel.is_emissive() {
                emissive += 1;
            }
            if voxel.is_metallic() {
                metallic += 1;
            }
            if let Some(slot) = tint_zones.get_mut(voxel.tint_zone() as usize) {
                *slot += 1;
            }
        }
    }

    let mut ranked: Vec<([u8; 3], u64)> = histogram.into_iter().collect();
    // Count descending, then color, so equal counts don't reorder
    // between runs (the histogram is a HashMap).
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let other = ranked.split_off(ranked.len().min(TOP_COLORS));

    let world_aabb = world.scene_aabb().map(Aabb::from_pair);
    Description {
        voxel_count,
        chunk_count: world.chunk_count(),
        world_aabb,
        size: world_aabb.map(|b| {
            let span = |lo: i32, hi: i32| hi as i64 - lo as i64 + 1;
            [
                span(b.min[0], b.max[0]),
                span(b.min[1], b.max[1]),
                span(b.min[2], b.max[2]),
            ]
        }),
        colors: ranked
            .into_iter()
            .map(|(rgb, count)| ColorCount { rgb, count })
            .collect(),
        other_color_voxels: other.iter().map(|(_, count)| count).sum(),
        other_color_kinds: other.len(),
        emissive,
        metallic,
        tint_zones,
        sockets: view
            .sockets
            .iter()
            .map(|socket| SocketInfo {
                name: socket.name.clone(),
                position: socket.position,
                normal: socket.normal,
            })
            .collect(),
        selection: view.selection.map(Aabb::from),
        structure: structure(world, voxel_count),
        graph: (!view.graph.nodes.is_empty()).then(|| view.graph.clone()),
        undo_depth: view.undo_depth,
        redo_depth: view.redo_depth,
    }
}

pub fn slice(world: &World, request: &SliceRequest) -> Result<String, OpsError> {
    let region = match request.region {
        Some(aabb) => aabb.to_selection(),
        None => match world.scene_aabb() {
            Some((min, max)) => Selection::from_corners(min, max),
            None => return Ok("(no solid voxels)".to_string()),
        },
    };

    let plane = request.axis.index();
    let (across, down) = plane_axes(plane);
    let (left, right) = axis_range(region, across);
    let (near, far) = axis_range(region, down);

    // i64, and only then compared: `region` is a bare `[i32; 3]` pair
    // that reaches here without passing `check_coord` (the ops path's
    // ceiling lives in `compile`, and a region defaulted from the scene
    // comes out of the file), so a span of four billion cells used to
    // overflow this subtraction *before* the limit below could refuse
    // it — and in the editor's bridge that panic lands on the frame
    // loop's own thread.
    let width = right as i64 - left as i64 + 1;
    let height = far as i64 - near as i64 + 1;
    if width > MAX_SLICE_SIDE as i64 || height > MAX_SLICE_SIDE as i64 {
        return Err(OpsError::new(
            ErrorCode::SliceTooLarge,
            format!(
                "slice is {width}×{height}; at most {MAX_SLICE_SIDE} per side — pass a smaller \"region\""
            ),
        ));
    }

    // Vertical views read like an elevation drawing: the top row is the
    // highest Y. A top-down view reads like a map: the first row is the
    // lowest Z. Either way the header states the row order rather than
    // leaving the agent to guess it.
    // The vertical axis of the image is Y for an elevation and Z for a
    // map. `down == 1` means the rows run along Y, which is the
    // elevation case — the one that reads top-down, highest row first.
    let elevation = down == 1;
    let rows: Vec<i32> = if elevation {
        (near..=far).rev().collect()
    } else {
        (near..=far).collect()
    };

    let mut legend: Vec<[u8; 3]> = Vec::new();
    let mut grid = String::with_capacity(rows.len() * (width as usize + 1));
    for &row in &rows {
        for column in left..=right {
            let mut pos = [0i32; 3];
            pos[plane] = request.index;
            pos[across] = column;
            pos[down] = row;
            let voxel = world.get_voxel(pos[0], pos[1], pos[2]);
            grid.push(if voxel.is_air() {
                '.'
            } else {
                match request.mode {
                    SliceMode::Solid => '#',
                    SliceMode::Color => color_key(&mut legend, [voxel.r, voxel.g, voxel.b]),
                }
            });
        }
        grid.push('\n');
    }

    let name = ["x", "y", "z"];
    let mut out = format!(
        "{}={}  {}={}..{} (left->right)  {}={}..{} (top->bottom)\n",
        name[plane],
        request.index,
        name[across],
        left,
        right,
        name[down],
        rows.first().copied().unwrap_or(near),
        rows.last().copied().unwrap_or(far),
    );
    if request.mode == SliceMode::Color {
        for (i, rgb) in legend.iter().enumerate() {
            out.push_str(&format!(
                "{} = rgb({}, {}, {})\n",
                letter(i),
                rgb[0],
                rgb[1],
                rgb[2]
            ));
        }
    }
    out.push_str(&grid);
    Ok(out)
}

/// The two axes spanning the plane perpendicular to `axis`, as
/// `(across, down)` component indices.
fn plane_axes(axis: usize) -> (usize, usize) {
    match axis {
        0 => (2, 1), // looking along X: Z across, Y down
        1 => (0, 2), // looking down Y: X across, Z down
        _ => (0, 1), // looking along Z: X across, Y down
    }
}

fn axis_range(region: Selection, axis: usize) -> (i32, i32) {
    let min = [region.min.0, region.min.1, region.min.2];
    let max = [region.max.0, region.max.1, region.max.2];
    (min[axis], max[axis])
}

/// Assign letters to colors in the order they first appear.
fn color_key(legend: &mut Vec<[u8; 3]>, rgb: [u8; 3]) -> char {
    match legend.iter().position(|entry| *entry == rgb) {
        Some(i) => letter(i),
        None => {
            legend.push(rgb);
            letter(legend.len() - 1)
        }
    }
}

/// `A`..`Z`, then `?` — past 26 distinct colors a letter map has
/// stopped being readable anyway.
fn letter(index: usize) -> char {
    if index < 26 {
        (b'A' + index as u8) as char
    } else {
        '?'
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_ops::AgentSession;
    use crate::core::Voxel;
    use crate::editor::Socket;

    fn request(json: &str) -> SliceRequest {
        serde_json::from_str(json).expect("slice request should parse")
    }

    // -------- structure --------

    fn filled(cells: impl IntoIterator<Item = (i32, i32, i32)>) -> World {
        let mut world = World::new();
        for (x, y, z) in cells {
            world.set_voxel(x, y, z, Voxel::from_rgb(120, 120, 120));
        }
        world
    }

    fn measure(world: &World) -> Structure {
        structure(world, solid_voxel_count(world)).expect("this world is measurable")
    }

    /// Two cubes a cell apart: the failure an agent can't see in a
    /// render and can't reliably catch in a slice.
    #[test]
    fn a_gap_of_one_cell_reads_as_two_components() {
        let touching = filled((0..4).flat_map(|x| (0..2).map(move |y| (x, y, 0))));
        assert_eq!(measure(&touching).components, 1);

        let split = filled(
            (0..2)
                .chain(3..5)
                .flat_map(|x| (0..2).map(move |y| (x, y, 0))),
        );
        let report = measure(&split);
        assert_eq!(report.components, 2);
        assert_eq!(report.largest_component, 4);
        assert_eq!(report.loose_parts.len(), 1);
        assert_eq!(report.loose_parts[0].voxels, 4);
    }

    #[test]
    fn corner_contact_is_not_a_connection() {
        // 6-connectivity, stated: two voxels meeting only at a corner
        // are two parts to the mesher, the flood fill and a printer.
        let report = measure(&filled([(0, 0, 0), (1, 1, 0)]));
        assert_eq!(report.components, 2);
    }

    #[test]
    fn a_part_that_never_reaches_the_lowest_layer_is_floating() {
        // The ground slab plus a block hanging four cells above it.
        let world = filled(
            (0..3)
                .flat_map(|x| (0..3).map(move |z| (x, 0, z)))
                .chain([(1, 4, 1)]),
        );
        let report = measure(&world);
        assert_eq!(report.components, 2);
        assert_eq!(report.floating_components, 1);
        // …and it is a measurement, not a verdict: the same number is
        // right for a tree canopy.
    }

    #[test]
    fn enclosed_counts_the_interior_hollow_would_remove() {
        // A solid 3³ block has exactly one fully-surrounded cell.
        let cube = filled((0..3).flat_map(|x| {
            (0..3).flat_map(move |y| (0..3).map(move |z| (x, y, z)))
        }));
        assert_eq!(measure(&cube).enclosed, 1);
        // A 3×3 slab has none — every cell is on a face.
        let slab = filled((0..3).flat_map(|x| (0..3).map(move |z| (x, 0, z))));
        assert_eq!(measure(&slab).enclosed, 0);
    }

    /// `enclosed`, `cavities` and open space are three different things
    /// that all get called "hollow", and a case that reaches for the
    /// wrong one fails correct work. Stated as one test so the contrast
    /// is in one place.
    #[test]
    fn enclosed_solids_sealed_air_and_open_space_are_three_different_readings() {
        let cube = |n: i32| {
            (0..n).flat_map(move |x| {
                (0..n).flat_map(move |y| (0..n).map(move |z| (x, y, z)))
            })
        };

        // Solid 3³: one solid cell has no exposed face, and no air is
        // trapped anywhere.
        let solid = measure(&filled(cube(3)));
        assert_eq!(solid.enclosed, 1);
        assert_eq!(solid.cavities.as_ref().unwrap().count, 0);

        // Hollow it: the interior cell becomes trapped air instead. No
        // solid is enclosed any more — the readings swap.
        let mut shell = filled(cube(3));
        shell.set_voxel(1, 1, 1, Voxel::AIR);
        let shell = measure(&shell);
        assert_eq!(shell.enclosed, 0);
        let sealed = shell.cavities.as_ref().unwrap();
        assert_eq!(sealed.count, 1);
        assert_eq!(sealed.voxels, 1);

        // Open the shell to the outside and the cavity stops being one:
        // the air reaches the bounding box surface, exactly like the gap
        // under an arch.
        let mut opened = filled(cube(3));
        opened.set_voxel(1, 1, 1, Voxel::AIR);
        opened.set_voxel(1, 1, 0, Voxel::AIR);
        assert_eq!(measure(&opened).cavities.as_ref().unwrap().count, 0);
    }

    /// The reading that tells an arch from a wall. Both are one piece,
    /// nothing floating, same bounding box — only the ground contact
    /// differs.
    #[test]
    fn footprint_separates_a_span_from_a_wall() {
        // A wall: 6 wide, 4 tall, resting on all six cells.
        let wall = filled((0..6).flat_map(|x| (0..4).map(move |y| (x, y, 0))));
        assert_eq!(measure(&wall).footprint, 6);

        // A span with the same box: two legs and a deck. Two cells on
        // the ground, and the space under it is open, not a cavity.
        let span = filled(
            [(0, 0, 0), (0, 1, 0), (0, 2, 0), (5, 0, 0), (5, 1, 0), (5, 2, 0)]
                .into_iter()
                .chain((0..6).map(|x| (x, 3, 0))),
        );
        let span = measure(&span);
        assert_eq!(span.footprint, 2);
        assert_eq!(span.components, 1);
        assert_eq!(span.cavities.as_ref().unwrap().count, 0);
    }

    #[test]
    fn symmetry_uses_cell_semantics_not_point_semantics() {
        // An even-sized symmetric shape is the case that separates the
        // two: reflecting `p` to `-p` would call this 100% asymmetric.
        let world = filled((0..4).map(|x| (x, 0, 0)));
        let report = measure(&world);
        let x = &report.symmetry[0];
        assert_eq!(x.axis, "x");
        assert_eq!(x.mismatched, 0, "a 4-wide bar is symmetric about x");
        assert_eq!(x.ratio, 0.0);

        // Same bar with a two-cell bump on one side only: exactly the
        // two voxels whose mirror image is missing are counted.
        let lopsided = filled((0..4).map(|x| (x, 0, 0)).chain([(0, 1, 0), (1, 1, 0)]));
        let report = measure(&lopsided);
        assert_eq!(report.symmetry[0].mismatched, 2, "x is the lopsided axis");
        assert_eq!(report.symmetry[2].mismatched, 0, "z is a single layer");
    }

    #[test]
    fn an_empty_world_has_nothing_to_measure() {
        assert!(structure(&World::new(), 0).is_none());
    }

    /// A 3-wide, 2-tall wall at z = 0 with a marked corner.
    fn wall() -> AgentSession {
        let mut session = AgentSession::new();
        for x in 0..3 {
            for y in 0..2 {
                session.world.set_voxel(x, y, 0, Voxel::from_rgb(100, 100, 100));
            }
        }
        session.world.set_voxel(0, 0, 0, Voxel::from_rgb(200, 0, 0));
        session
    }

    #[test]
    fn describe_counts_what_is_there() {
        let mut session = wall();
        let mut lamp = Voxel::from_rgb(255, 255, 0);
        lamp.set_emissive(true);
        lamp.set_tint_zone(1);
        session.world.set_voxel(5, 0, 0, lamp);
        session.sockets.push(Socket::new("muzzle", [1.0, 2.0, 0.5], [0.0, 1.0, 0.0]));
        session.selection = Some(Selection::from_corners((0, 0, 0), (2, 1, 0)));

        let description = session.describe();
        assert_eq!(description.voxel_count, 7);
        assert_eq!(description.world_aabb, Some(Aabb { min: [0, 0, 0], max: [5, 1, 0] }));
        assert_eq!(description.size, Some([6, 2, 1]));
        assert_eq!(description.emissive, 1);
        assert_eq!(description.metallic, 0);
        assert_eq!(description.tint_zones, [6, 1, 0, 0]);
        assert_eq!(description.sockets.len(), 1);
        assert_eq!(description.sockets[0].name, "muzzle");
        assert_eq!(
            description.selection,
            Some(Aabb { min: [0, 0, 0], max: [2, 1, 0] })
        );

        // Colors, most common first.
        assert_eq!(description.colors[0].rgb, [100, 100, 100]);
        assert_eq!(description.colors[0].count, 5);
        assert_eq!(description.colors.len(), 3);
        assert_eq!(description.other_color_kinds, 0);
    }

    #[test]
    fn describe_reports_history_depth() {
        let mut session = AgentSession::new();
        let batch = serde_json::from_str(
            r#"{"version":1,"ops":[{"op":"box","min":[0,0,0],"max":[1,1,1],"voxel":{"rgb":[1,2,3]}}]}"#,
        )
        .unwrap();
        session.apply_ops(&batch).unwrap();
        assert_eq!(session.describe().undo_depth, 1);
        session.undo();
        let after = session.describe();
        assert_eq!(after.undo_depth, 0);
        assert_eq!(after.redo_depth, 1);
        assert_eq!(after.voxel_count, 0);
    }

    #[test]
    fn a_front_view_reads_like_an_elevation_drawing() {
        // Looking along Z: X to the right, Y down the page from the top
        // — the top row is the highest Y, the way a person draws a wall.
        let session = wall();
        let art = session.slice(&request(r#"{"axis":"z","index":0}"#)).unwrap();
        let lines: Vec<&str> = art.lines().collect();
        assert!(lines[0].contains("z=0"), "header names the plane: {}", lines[0]);
        assert!(
            lines[0].contains("y=1..0"),
            "header states the row order, got: {}",
            lines[0]
        );
        assert_eq!(&lines[1..], ["###", "###"]);
    }

    #[test]
    fn a_top_down_view_runs_z_downward() {
        let session = wall();
        let art = session
            .slice(&request(r#"{"axis":"y","index":0,"region":{"min":[0,0,0],"max":[2,0,1]}}"#))
            .unwrap();
        let lines: Vec<&str> = art.lines().collect();
        assert!(lines[0].contains("z=0..1"), "got: {}", lines[0]);
        assert_eq!(&lines[1..], ["###", "..."]);
    }

    #[test]
    fn color_mode_labels_each_color_and_prints_a_legend() {
        let session = wall();
        let art = session
            .slice(&request(r#"{"axis":"z","index":0,"mode":"color"}"#))
            .unwrap();
        let lines: Vec<&str> = art.lines().collect();
        // Row order is top (y=1) first, so gray is seen before red.
        assert_eq!(lines[1], "A = rgb(100, 100, 100)");
        assert_eq!(lines[2], "B = rgb(200, 0, 0)");
        assert_eq!(&lines[3..], ["AAA", "BAA"]);
    }

    #[test]
    fn an_empty_world_slices_to_a_plain_answer() {
        let session = AgentSession::new();
        assert_eq!(
            session.slice(&request(r#"{"index":0}"#)).unwrap(),
            "(no solid voxels)"
        );
    }

    #[test]
    fn an_unreadably_wide_slice_is_refused_with_a_way_out() {
        let session = wall();
        let error = session
            .slice(&request(
                r#"{"axis":"y","index":0,"region":{"min":[0,0,0],"max":[500,0,10]}}"#,
            ))
            .expect_err("500 columns is past the limit");
        assert_eq!(error.code, ErrorCode::SliceTooLarge);
        assert!(
            error.message.contains("region"),
            "the message should say how to narrow it, got: {}",
            error.message
        );
    }

    /// The width is a difference of two `i32`s that never went through
    /// `check_coord`, so a region four billion cells wide has to be
    /// *refused* rather than wrap into a small positive number and be
    /// waved through — the wrap used to panic on the subtraction first.
    #[test]
    fn a_slice_wider_than_i32_is_refused_rather_than_wrapping() {
        let session = wall();
        let error = session
            .slice(&request(
                r#"{"axis":"y","index":0,"region":{"min":[-2000000000,0,0],"max":[2000000000,0,0]}}"#,
            ))
            .expect_err("four billion columns is past the limit");
        assert_eq!(error.code, ErrorCode::SliceTooLarge);
    }

    // -------- coordinates a `.vxlt` can hold and the ops path cannot --------

    /// `MAX_COORD` bounds what an *op* may write; it says nothing about
    /// what an opened file holds. Every measurement below reads the
    /// world as it found it, and each used to do its arithmetic in i32
    /// — which the editor's bridge runs on the frame loop's own thread,
    /// so the panic took the editor and its unsaved work with it.
    #[test]
    fn a_span_wider_than_i32_is_reported_rather_than_wrapped() {
        let mut session = AgentSession::new();
        session.world.set_voxel(-2_000_000_000, 0, 0, Voxel::from_rgb(120, 120, 120));
        session.world.set_voxel(2_000_000_000, 0, 0, Voxel::from_rgb(120, 120, 120));

        let description = session.describe();
        assert_eq!(description.size, Some([4_000_000_001, 1, 1]));
        let structure = description.structure.expect("two voxels is measurable");
        assert_eq!(structure.components, 2);
    }

    /// The reflection is `min + max - p`: the *result* always lands back
    /// inside the box, but the sum of two coordinates need not fit i32.
    #[test]
    fn symmetry_survives_a_pair_of_coordinates_that_sum_past_i32() {
        let mut session = AgentSession::new();
        session.world.set_voxel(1_500_000_000, 0, 0, Voxel::from_rgb(120, 120, 120));
        session.world.set_voxel(2_000_000_000, 0, 0, Voxel::from_rgb(120, 120, 120));

        let structure = session.describe().structure.expect("measurable");
        let x = structure.symmetry.iter().find(|s| s.axis == "x").unwrap();
        // The two cells are each other's mirror image about the box's
        // midpoint, so this is a symmetric model — a wrapped sum used to
        // be the only reason it wasn't.
        assert_eq!(x.mismatched, 0);
    }

    /// A cell sitting on `i32::MAX` has no neighbor past it. Three
    /// separate walks step to a neighbor here — the component flood, the
    /// enclosure test and the sealed-air flood — and all three read "no
    /// such cell" rather than overflowing to find out.
    #[test]
    fn a_cell_at_the_edge_of_i32_has_no_neighbour_past_it() {
        let mut session = AgentSession::new();
        session.world.set_voxel(i32::MAX, 0, 0, Voxel::from_rgb(120, 120, 120));
        session.world.set_voxel(i32::MAX, 2, 0, Voxel::from_rgb(120, 120, 120));

        let structure = session.describe().structure.expect("two voxels is measurable");
        assert_eq!(structure.components, 2);
        // Neither cell is walled in — the +x side isn't a cell at all.
        assert_eq!(structure.enclosed, 0);
        // The air cell between them reaches the box wall, so it is not
        // a cavity; the point is that the flood got to say so.
        assert_eq!(structure.cavities.map(|c| c.count), Some(0));
    }
}
