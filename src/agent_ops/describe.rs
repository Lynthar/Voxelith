//! Feedback: what the document contains, and what one plane of it
//! looks like.
//!
//! An agent editing blind needs a cheap way to check its own work. Two
//! views, both text: [`describe`] is the summary (how much, how big,
//! what colors), [`slice`] is one plane as ASCII art, which is what
//! actually catches "the door is one cell too high".

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::core::World;
use crate::editor::{Selection, Socket};

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
    pub size: Option<[i32; 3]>,
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
            [
                b.max[0] - b.min[0] + 1,
                b.max[1] - b.min[1] + 1,
                b.max[2] - b.min[2] + 1,
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

    let width = right - left + 1;
    let height = far - near + 1;
    if width > MAX_SLICE_SIDE || height > MAX_SLICE_SIDE {
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
    let top_down = down == 1;
    let rows: Vec<i32> = if top_down {
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
}
