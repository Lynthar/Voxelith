//! Off-screen views of a world, rendered on the CPU.
//!
//! This is the agent's eye. `describe` gives it counts and a bounding
//! box, `slice` gives it one plane as text — neither answers "does this
//! look like a hut?". A picture does, and an agent that can see its own
//! model can correct it.
//!
//! Distinct from [`crate::render`] in every way that matters: that one
//! is the interactive viewport, needs a GPU and a window, and lives
//! behind the `gui` feature. This one is pure CPU ray casting over the
//! voxel grid, so it works in a container with no display — which is
//! where an agent actually runs.
//!
//! Orthographic, not perspective. Six axis views plus an isometric one,
//! all parallel projections, because the job is reading a model rather
//! than admiring it: no foreshortening means a wall that looks straight
//! *is* straight, and equal cells stay equal size wherever they sit.
//! [`Framing`] then reports exactly what the image covers, so "the door
//! is one cell too high" can be turned back into coordinates.

use glam::Vec3;
use rayon::prelude::*;

use crate::core::{Voxel, World};

/// Default image edge, in pixels. Enough to read the silhouette and the
/// colors of a typical model; a client turns an image this size into
/// well under a hundred tokens, so a seven-view sweep stays affordable.
pub const DEFAULT_SIZE: u32 = 256;

/// Largest image edge accepted. A refusal, not a clamp — see
/// [`ViewError`].
pub const MAX_SIZE: u32 = 1024;

/// Backstop on the walk. Rays stop when they leave the scene box, so
/// this only fires on a bounding box already too big to draw — and it
/// fires as a blank pixel rather than a hang. [`View::truncated`] then
/// says it happened, because a blank pixel nobody explains reads as
/// "the model is gone".
const MAX_STEPS: u32 = 8192;

/// Direction the scene is lit from. Off-axis on all three so no face of
/// an axis-aligned box comes out unlit, and — the part that's easy to
/// get wrong — **no two components equal**, or the two faces they light
/// come out the same tone and the isometric view of a cube reads as a
/// flat hexagon. A test pins that.
///
/// See [`key_light`]: this is the direction for a view that looks at the
/// lit side, and it's mirrored for the views that don't.
const LIGHT: Vec3 = Vec3::new(0.34, 0.82, 0.55);

/// Floor on the lambert term. Faces pointing away from the light stay
/// legible instead of going black — the picture is a diagram, and an
/// unlit face still has a color the agent asked for.
const AMBIENT: f32 = 0.35;

/// How much a fully-enclosed corner darkens. Enough to read concavity,
/// gentle enough that it can't be mistaken for a different material.
const AO_STRENGTH: f32 = 0.45;

/// Background gray. Mid-tone on purpose: light models and dark models
/// both stand out against it, which a white or black field can't manage.
const BACKGROUND: [u8; 3] = [58, 58, 64];

/// One of the seven canonical viewpoints.
///
/// Named for where the camera *is*, not where it looks: `Top` is the
/// view from above. `Front` looks down −Z with +X to the right and +Y
/// up, matching the editor's own starting camera.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ViewKind {
    /// From +Z. Right is +X, up is +Y.
    Front,
    /// From −Z. Right is −X, up is +Y.
    Back,
    /// From −X. Right is +Z, up is +Y.
    Left,
    /// From +X. Right is −Z, up is +Y.
    Right,
    /// From +Y, looking down. Right is +X, up in the image is −Z.
    Top,
    /// From −Y, looking up. Right is +X, up in the image is +Z.
    Bottom,
    /// From (+1, +1, +1). Three faces of every cube are visible, which
    /// is why this is the default: one image, all three dimensions.
    Iso,
}

impl ViewKind {
    /// Every viewpoint, in the order a contact sheet reads best.
    pub const ALL: [ViewKind; 7] = [
        ViewKind::Iso,
        ViewKind::Front,
        ViewKind::Back,
        ViewKind::Left,
        ViewKind::Right,
        ViewKind::Top,
        ViewKind::Bottom,
    ];

    /// The name used on the wire — same string `serde` produces.
    pub fn as_str(self) -> &'static str {
        match self {
            ViewKind::Front => "front",
            ViewKind::Back => "back",
            ViewKind::Left => "left",
            ViewKind::Right => "right",
            ViewKind::Top => "top",
            ViewKind::Bottom => "bottom",
            ViewKind::Iso => "iso",
        }
    }

    /// Look up a wire name. `None` for anything else — callers turn that
    /// into an error naming the alternatives rather than guessing.
    ///
    /// Deliberately not `FromStr`: that trait's `from_str` returns a
    /// `Result`, and the error type would exist only to be discarded by
    /// [`ViewKind::parse_list`], which is the one caller that has
    /// anything useful to say about a bad name.
    pub fn from_name(name: &str) -> Option<Self> {
        ViewKind::ALL
            .into_iter()
            .find(|kind| kind.as_str() == name)
    }

    /// Parse a comma-separated list, where `all` means every view.
    ///
    /// Lives here rather than in the CLI so the names an agent can type
    /// come from one place — the same list [`ViewKind::ALL`] and the
    /// serde representation are built from.
    pub fn parse_list(spec: &str) -> Result<Vec<ViewKind>, String> {
        let mut kinds = Vec::new();
        for name in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if name.eq_ignore_ascii_case("all") {
                return Ok(ViewKind::ALL.to_vec());
            }
            let kind = ViewKind::from_name(&name.to_ascii_lowercase()).ok_or_else(|| {
                let known: Vec<&str> = ViewKind::ALL.iter().map(|k| k.as_str()).collect();
                format!("unknown view {name:?}; pick from {} or all", known.join(", "))
            })?;
            // Rendering the same view twice would just write the same
            // file twice — quietly drop the repeat rather than fail on
            // it; `all,iso` is a reasonable thing to type.
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        }
        if kinds.is_empty() {
            return Err("name at least one view".to_string());
        }
        Ok(kinds)
    }

    /// `(forward, right, up)` — an orthonormal camera basis, where
    /// `forward` is the direction rays travel.
    fn basis(self) -> (Vec3, Vec3, Vec3) {
        match self {
            ViewKind::Front => (-Vec3::Z, Vec3::X, Vec3::Y),
            ViewKind::Back => (Vec3::Z, -Vec3::X, Vec3::Y),
            ViewKind::Left => (Vec3::X, Vec3::Z, Vec3::Y),
            ViewKind::Right => (-Vec3::X, -Vec3::Z, Vec3::Y),
            ViewKind::Top => (-Vec3::Y, Vec3::X, -Vec3::Z),
            ViewKind::Bottom => (Vec3::Y, Vec3::X, Vec3::Z),
            ViewKind::Iso => {
                // Down the (1,1,1) diagonal. `right` is horizontal (no Y
                // component) so the image doesn't sit at a tilt, and
                // `up` completes the right-handed set.
                let forward = Vec3::new(-1.0, -1.0, -1.0).normalize();
                let right = forward.cross(Vec3::Y).normalize();
                let up = right.cross(forward).normalize();
                (forward, right, up)
            }
        }
    }
}

/// Why a view couldn't be rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewError {
    /// The requested edge is outside `1..=MAX_SIZE`.
    SizeOutOfRange(u32),
}

impl std::fmt::Display for ViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViewError::SizeOutOfRange(size) => write!(
                f,
                "size {size} is outside 1..={MAX_SIZE} pixels; ask for a smaller image"
            ),
        }
    }
}

impl std::error::Error for ViewError {}

/// Inclusive `(min, max)` cell bounds — the shape `World::scene_aabb`
/// speaks in.
pub type CellBounds = ((i32, i32, i32), (i32, i32, i32));

/// What an image actually covers, so a pixel can be turned back into
/// cells.
///
/// Without this a render is a pretty picture an agent can't act on: it
/// can see the door is too high but has no way to say by how much. With
/// it, `bounds` plus `cells_per_pixel` plus the axes is enough to do the
/// arithmetic.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Framing {
    /// Cell bounds of the scene this view was fitted to. `None` for an
    /// empty world.
    pub bounds: Option<CellBounds>,
    /// Edge length of one pixel, in cells.
    pub cells_per_pixel: f32,
    /// World direction that is right in the image.
    pub right: [f32; 3],
    /// World direction that is up in the image.
    pub up: [f32; 3],
    /// World direction the rays travel — into the screen.
    pub forward: [f32; 3],
}

/// A rendered view: the PNG plus what it covers.
#[derive(Debug, Clone)]
pub struct View {
    pub kind: ViewKind,
    pub size: u32,
    /// PNG bytes, RGB8.
    pub png: Vec<u8>,
    pub framing: Framing,
    /// Set when the world held nothing to draw, so a caller can say so
    /// instead of handing over a rectangle of background and letting the
    /// agent conclude its model vanished.
    pub empty: bool,
    /// Set when at least one ray spent its whole step budget without
    /// reaching the scene or leaving it — so those pixels are
    /// background because the walk gave up, not because nothing is
    /// there.
    ///
    /// `empty` covers the same mistake from the other side, and covers
    /// it only for a world with no voxels at all: a scene whose extent
    /// along the view direction is past what the walk can cross draws
    /// as a full rectangle of background while `empty` is false and the
    /// bounds say there is plenty to see. An agent reading that
    /// concludes its model vanished and rebuilds it.
    pub truncated: bool,
}

/// Render one view of `world`.
///
/// Cost is `size²` rays, each walking the grid; the rows are split
/// across the rayon pool. Measured on a ~1100-voxel model in release:
/// **11 ms** at 256², **480 ms** at 1024² — so a seven-view sweep at the
/// default size costs about as much as one frame of anything else, and
/// the largest size is a deliberate wait.
pub fn render(world: &World, kind: ViewKind, size: u32) -> Result<View, ViewError> {
    if size == 0 || size > MAX_SIZE {
        return Err(ViewError::SizeOutOfRange(size));
    }
    let (forward, right, up) = kind.basis();

    let Some((min, max)) = world.scene_aabb() else {
        let blank: Vec<u8> = BACKGROUND
            .iter()
            .copied()
            .cycle()
            .take((size * size * 3) as usize)
            .collect();
        return Ok(View {
            kind,
            size,
            png: encode_png(&blank, size),
            framing: Framing {
                bounds: None,
                cells_per_pixel: 0.0,
                right: right.to_array(),
                up: up.to_array(),
                forward: forward.to_array(),
            },
            empty: true,
            truncated: false,
        });
    };

    // A cell spans [p, p+1), so the solid the camera sees runs to
    // `max + 1` — the same off-by-one-cell rule the `.vox` rotation and
    // the mesher both live by.
    let low = Vec3::new(min.0 as f32, min.1 as f32, min.2 as f32);
    let high = Vec3::new(max.0 as f32 + 1.0, max.1 as f32 + 1.0, max.2 as f32 + 1.0);
    let center = (low + high) * 0.5;

    // Fit the projected scene to the frame. The extent along an axis of
    // the image is the box's half-extent projected onto it, which for an
    // axis view is just half a side and for the isometric view picks up
    // the diagonal.
    let half = (high - low) * 0.5;
    let extent_right = half.x * right.x.abs() + half.y * right.y.abs() + half.z * right.z.abs();
    let extent_up = half.x * up.x.abs() + half.y * up.y.abs() + half.z * up.z.abs();
    // Square pixels and a square image: one scale for both axes, plus a
    // small margin so the model doesn't touch the border.
    let cells_per_pixel = (extent_right.max(extent_up) * 2.05) / size as f32;

    // Start every ray just outside the box, measured *along the view
    // direction* rather than by the box's diagonal. Both put the origin
    // outside the scene, which is all the walk needs — but the diagonal
    // charges a wide scene for width the camera is not looking through,
    // and the walk gives up after `MAX_STEPS`. A scene 100,000 cells
    // wide and one cell thick used to start its front-view rays 50,000
    // cells away from a wall one cell deep, run out of steps, and hand
    // back a picture of the background. This is the box's support
    // distance in `forward`: for an axis view, half the thickness the
    // camera actually looks through.
    let depth = half.x * forward.x.abs()
        + half.y * forward.y.abs()
        + half.z * forward.z.abs()
        + 2.0;
    let origin = center - forward * depth;

    let light_dir = key_light(forward);
    let mut pixels = vec![0u8; (size * size * 3) as usize];
    let row_bytes = (size * 3) as usize;
    let ran_out = std::sync::atomic::AtomicBool::new(false);
    pixels
        .par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(y, row)| {
            // Image y grows downward; the world's up axis grows the
            // other way.
            let offset_up = (size as f32 * 0.5 - (y as f32 + 0.5)) * cells_per_pixel;
            for x in 0..size as usize {
                let offset_right = ((x as f32 + 0.5) - size as f32 * 0.5) * cells_per_pixel;
                let from = origin + right * offset_right + up * offset_up;
                let color = match cast(world, from, forward, min, max) {
                    Trace::Hit(hit) => shade(world, &hit, forward, light_dir),
                    Trace::Missed => BACKGROUND,
                    Trace::OutOfSteps => {
                        ran_out.store(true, std::sync::atomic::Ordering::Relaxed);
                        BACKGROUND
                    }
                };
                row[x * 3..x * 3 + 3].copy_from_slice(&color);
            }
        });

    Ok(View {
        kind,
        size,
        png: encode_png(&pixels, size),
        framing: Framing {
            bounds: Some((min, max)),
            cells_per_pixel,
            right: right.to_array(),
            up: up.to_array(),
            forward: forward.to_array(),
        },
        empty: false,
        truncated: ran_out.into_inner(),
    })
}

/// Smallest share of the key light that must come from behind the
/// camera. An axis view of a convex model sees exactly one face, so that
/// face's brightness *is* the picture — leave it to the raw dot product
/// and `left` comes back at barely above ambient while `top` is fully
/// lit, for no reason the reader can see.
const MIN_FACING: f32 = 0.55;

/// The key light for one view.
///
/// [`LIGHT`] as written, then bent twice: mirrored across the view plane
/// if it would otherwise come from behind the model, and tilted toward
/// the camera until [`MIN_FACING`] of it falls on what the camera can
/// see. The final re-normalize then shortens that share a little — the
/// dimmest view comes out around 0.50 rather than the 0.55 asked for,
/// which is close enough for a diagram and cheaper than solving for
/// the exact tilt.
///
/// Both exist because these are diagrams, not renders. A fixed
/// world-space light makes `back`, `left` and `bottom` silhouettes in
/// flat ambient — correct lighting, useless picture, and those are the
/// views someone asks for precisely to inspect that side. The mirror is
/// across the plane rather than a negation so the slant survives and a
/// box's top stays the brightest face in every view: an agent reading
/// six images shouldn't have to work out which way is up in each one.
fn key_light(forward: Vec3) -> Vec3 {
    let mut light = LIGHT.normalize();
    if light.dot(-forward) < 0.0 {
        light = light - 2.0 * light.dot(forward) * forward;
    }
    let facing = light.dot(-forward);
    if facing < MIN_FACING {
        light -= forward * (MIN_FACING - facing);
    }
    light.normalize()
}

/// What one ray found.
enum Trace {
    /// The first solid cell along the ray.
    Hit(Hit),
    /// The ray left the scene box without touching anything. Background,
    /// and the model is what the picture says it is.
    Missed,
    /// The ray spent its whole step budget without either. Background
    /// too — but for a reason that has nothing to do with the model,
    /// which is why [`View::truncated`] exists to say so.
    OutOfSteps,
}

/// Where a ray stopped: the cell it entered and the face it came through.
struct Hit {
    cell: (i32, i32, i32),
    voxel: Voxel,
    /// Outward normal of the face the ray crossed, as a unit axis.
    normal: (i32, i32, i32),
}

/// Walk the voxel grid along a ray until it hits something solid.
///
/// Amanatides–Woo: track, per axis, the distance to the next cell
/// boundary and the distance between boundaries, then always step the
/// axis that is closest. Every cell the ray touches is visited exactly
/// once and in order, which is what makes the first solid cell the
/// visible one.
///
/// `box_min` / `box_max` are the scene's inclusive cell bounds, and they
/// are what makes this affordable: a ray that has passed the box, or
/// that runs parallel to it and outside it, can never hit anything, so
/// it stops there. Without that test every background pixel — four in
/// ten of an isometric view — walked the full [`MAX_STEPS`]: measured,
/// that was 183 ms for a 256² frame against 11 ms with it.
fn cast(
    world: &World,
    from: Vec3,
    direction: Vec3,
    box_min: (i32, i32, i32),
    box_max: (i32, i32, i32),
) -> Trace {
    let mut cell = (
        from.x.floor() as i32,
        from.y.floor() as i32,
        from.z.floor() as i32,
    );
    let step = (
        sign(direction.x),
        sign(direction.y),
        sign(direction.z),
    );
    // An axis the ray is parallel to never moves. If it already sits
    // outside the box on that axis, no amount of stepping brings it
    // back — which is most of the frame in an axis view, where two of
    // the three axes are frozen.
    if (step.0 == 0 && (cell.0 < box_min.0 || cell.0 > box_max.0))
        || (step.1 == 0 && (cell.1 < box_min.1 || cell.1 > box_max.1))
        || (step.2 == 0 && (cell.2 < box_min.2 || cell.2 > box_max.2))
    {
        return Trace::Missed;
    }
    // A ray exactly parallel to an axis never crosses that axis's
    // boundaries: infinity keeps it from ever being chosen as the
    // nearest, without a special case in the loop.
    let delta = Vec3::new(
        safe_inverse(direction.x),
        safe_inverse(direction.y),
        safe_inverse(direction.z),
    );
    let mut next = Vec3::new(
        boundary(from.x, direction.x, cell.0),
        boundary(from.y, direction.y, cell.1),
        boundary(from.z, direction.z, cell.2),
    );

    // The ray starts outside the scene, so the first cell can't be a
    // hit; the loop tests after stepping, and the normal is always the
    // face just crossed.
    for _ in 0..MAX_STEPS {
        let axis = if next.x <= next.y && next.x <= next.z {
            0
        } else if next.y <= next.z {
            1
        } else {
            2
        };
        let mut normal = (0, 0, 0);
        match axis {
            0 => {
                cell.0 += step.0;
                next.x += delta.x;
                normal.0 = -step.0;
            }
            1 => {
                cell.1 += step.1;
                next.y += delta.y;
                normal.1 = -step.1;
            }
            _ => {
                cell.2 += step.2;
                next.z += delta.z;
                normal.2 = -step.2;
            }
        }
        // Past the box on the axis just stepped, and heading away: the
        // ray is done. Only the stepped axis can have changed, so this
        // is one comparison rather than six.
        let left_the_box = match axis {
            0 => (step.0 > 0 && cell.0 > box_max.0) || (step.0 < 0 && cell.0 < box_min.0),
            1 => (step.1 > 0 && cell.1 > box_max.1) || (step.1 < 0 && cell.1 < box_min.1),
            _ => (step.2 > 0 && cell.2 > box_max.2) || (step.2 < 0 && cell.2 < box_min.2),
        };
        if left_the_box {
            return Trace::Missed;
        }

        let voxel = world.get_voxel(cell.0, cell.1, cell.2);
        if !voxel.is_air() {
            return Trace::Hit(Hit {
                cell,
                voxel,
                normal,
            });
        }
    }
    Trace::OutOfSteps
}

/// Light one hit: lambert against a fixed key light, floored at
/// `AMBIENT`, darkened by how boxed-in the face is.
///
/// Emissive voxels skip the lighting entirely — they are the material
/// that says "this glows", and shading one like ordinary paint hides
/// the very flag the agent set.
fn shade(world: &World, hit: &Hit, forward: Vec3, light_dir: Vec3) -> [u8; 3] {
    let base = [
        hit.voxel.r as f32,
        hit.voxel.g as f32,
        hit.voxel.b as f32,
    ];
    if hit.voxel.is_emissive() {
        return [base[0] as u8, base[1] as u8, base[2] as u8];
    }

    let normal = Vec3::new(
        hit.normal.0 as f32,
        hit.normal.1 as f32,
        hit.normal.2 as f32,
    );
    let lambert = normal.dot(light_dir).max(0.0);
    let mut light = AMBIENT + (1.0 - AMBIENT) * lambert;
    // Metallic reads as a sharper falloff toward grazing angles — not
    // physically a specular model, just enough contrast that the flag
    // is visible in the picture at all.
    if hit.voxel.is_metallic() {
        let facing = (-forward).dot(normal).abs();
        light *= 0.75 + 0.35 * facing;
    }
    light *= ambient_occlusion(world, hit);

    [
        (base[0] * light).clamp(0.0, 255.0) as u8,
        (base[1] * light).clamp(0.0, 255.0) as u8,
        (base[2] * light).clamp(0.0, 255.0) as u8,
    ]
}

/// Fraction of light reaching a face: 1.0 in the open, less as the eight
/// cells ringing it on the outside fill in.
///
/// Sampled on the empty side of the face — the cells that would actually
/// block light — which is what makes an inside corner darken and a flat
/// wall stay even.
fn ambient_occlusion(world: &World, hit: &Hit) -> f32 {
    let (nx, ny, nz) = hit.normal;
    // The two axes lying in the face's plane.
    let (u, v) = match (nx, ny, nz) {
        (0, 0, _) => ((1, 0, 0), (0, 1, 0)),
        (0, _, 0) => ((1, 0, 0), (0, 0, 1)),
        _ => ((0, 1, 0), (0, 0, 1)),
    };
    let front = (hit.cell.0 + nx, hit.cell.1 + ny, hit.cell.2 + nz);
    let mut blocked = 0;
    for du in -1..=1 {
        for dv in -1..=1 {
            if du == 0 && dv == 0 {
                continue;
            }
            let x = front.0 + u.0 * du + v.0 * dv;
            let y = front.1 + u.1 * du + v.1 * dv;
            let z = front.2 + u.2 * du + v.2 * dv;
            if !world.get_voxel(x, y, z).is_air() {
                blocked += 1;
            }
        }
    }
    1.0 - AO_STRENGTH * (blocked as f32 / 8.0)
}

/// −1 / 0 / +1, where 0 means "parallel to this axis".
fn sign(v: f32) -> i32 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

/// Distance along the ray to cross one whole cell on this axis.
fn safe_inverse(v: f32) -> f32 {
    if v == 0.0 {
        f32::INFINITY
    } else {
        (1.0 / v).abs()
    }
}

/// Distance along the ray from `position` to the next cell boundary on
/// this axis.
fn boundary(position: f32, direction: f32, cell: i32) -> f32 {
    if direction > 0.0 {
        (cell as f32 + 1.0 - position) / direction
    } else if direction < 0.0 {
        (position - cell as f32) / -direction
    } else {
        f32::INFINITY
    }
}

/// Encode an RGB8 buffer as PNG.
///
/// Infallible in practice — the only writer is an in-memory `Vec` and
/// the buffer length is ours — so a failure here is a bug, not a
/// condition a caller could handle; it yields an empty image rather than
/// taking the process down.
fn encode_png(pixels: &[u8], size: u32) -> Vec<u8> {
    let mut png = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut png);
    match image::ImageEncoder::write_image(
        encoder,
        pixels,
        size,
        size,
        image::ExtendedColorType::Rgb8,
    ) {
        Ok(()) => png,
        Err(e) => {
            log::error!("PNG encoding failed for a {size}x{size} view: {e}");
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode back to raw pixels so a test can assert on what was drawn.
    fn pixels_of(view: &View) -> Vec<u8> {
        let decoded = image::load_from_memory(&view.png).expect("a valid PNG");
        assert_eq!(decoded.width(), view.size);
        assert_eq!(decoded.height(), view.size);
        decoded.to_rgb8().into_raw()
    }

    fn count_non_background(view: &View) -> usize {
        pixels_of(view)
            .chunks(3)
            .filter(|p| p != &BACKGROUND)
            .count()
    }

    fn cube(half: i32, color: Voxel) -> World {
        let mut world = World::new();
        for x in -half..=half {
            for y in -half..=half {
                for z in -half..=half {
                    world.set_voxel(x, y, z, color);
                }
            }
        }
        world
    }

    #[test]
    fn an_empty_world_renders_a_background_and_says_it_was_empty() {
        // Not an error: "I looked and there is nothing there" is a real
        // answer. But it has to be labelled, or an agent reads a blank
        // rectangle as its model having disappeared.
        let view = render(&World::new(), ViewKind::Iso, 32).unwrap();
        assert!(view.empty);
        assert!(view.framing.bounds.is_none());
        assert_eq!(count_non_background(&view), 0);
    }

    /// A scene far wider than it is deep. The ray origin used to be set
    /// by the box's *diagonal*, so a front view — looking through one
    /// cell of depth — started 50,000 cells out, ran out of steps, and
    /// came back as a rectangle of background with `empty: false` beside
    /// bounds promising a model. Measuring the start along the view
    /// direction is what the camera is actually looking through.
    #[test]
    fn a_wide_thin_scene_is_still_drawn_from_the_front() {
        let mut world = World::new();
        let red = Voxel::from_rgb(200, 60, 60);
        world.set_voxel(0, 0, 0, red);
        world.set_voxel(100_000, 0, 0, red);

        let view = render(&world, ViewKind::Front, 64).unwrap();
        assert!(!view.empty);
        assert!(!view.truncated, "the walk had one cell of depth to cross");
        // The picture can still be blank, and honestly so: `framing`
        // says each pixel covers about 1600 cells, and a one-cell voxel
        // sampled at pixel centers is smaller than that. Aliasing an
        // agent can compute from the numbers it was given is a
        // different thing from a walk that quietly stopped.
    }

    /// The case the step budget genuinely can't cross: an isometric view
    /// looks along the diagonal, so a scene this wide is past it whatever
    /// the origin. The picture is still all background — the point is
    /// that it now says why, instead of letting an agent read it as
    /// "my model vanished" and build it again.
    #[test]
    fn a_scene_too_big_to_trace_says_the_walk_gave_up() {
        let mut world = World::new();
        let red = Voxel::from_rgb(200, 60, 60);
        world.set_voxel(0, 0, 0, red);
        world.set_voxel(100_000, 0, 0, red);

        let view = render(&world, ViewKind::Iso, 32).unwrap();
        assert!(!view.empty, "the world does hold voxels");
        assert_eq!(count_non_background(&view), 0);
        assert!(view.truncated, "background that nothing explains");
    }

    #[test]
    fn a_cube_fills_the_frame_without_touching_the_border() {
        let world = cube(3, Voxel::from_rgb(200, 60, 60));
        let view = render(&world, ViewKind::Front, 64).unwrap();
        let pixels = pixels_of(&view);

        let drawn = count_non_background(&view);
        let total = (view.size * view.size) as usize;
        assert!(
            drawn > total / 2,
            "a cube head-on should cover most of the frame, covered {drawn}/{total}"
        );
        // Every border pixel is background — that's the 2.05 margin.
        let last = (view.size - 1) as usize;
        for i in 0..view.size as usize {
            for (x, y) in [(i, 0), (i, last), (0, i), (last, i)] {
                let at = (y * view.size as usize + x) * 3;
                assert_eq!(
                    &pixels[at..at + 3],
                    &BACKGROUND,
                    "pixel ({x},{y}) on the border should be background"
                );
            }
        }
    }

    #[test]
    fn the_six_axis_views_see_the_faces_their_names_promise() {
        // Two voxels apart on +X, told apart by color rather than by
        // position: the projection is always centred on the scene, so
        // where the *pair* sits says nothing — which of the two lands on
        // the right says everything. This is the test that catches a
        // flipped basis vector, the kind of bug that silently mirrors
        // every render an agent ever reads.
        const ORIGIN: [u8; 3] = [220, 40, 40];
        const PLUS_X: [u8; 3] = [40, 80, 220];
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(ORIGIN[0], ORIGIN[1], ORIGIN[2]));
        world.set_voxel(4, 0, 0, Voxel::from_rgb(PLUS_X[0], PLUS_X[1], PLUS_X[2]));

        // Which half of the image a color's pixels fall in: <0 left,
        // >0 right. Shading moves the exact values, so match on the
        // dominant channel instead of the literal color.
        let side_of = |kind: ViewKind, red_wins: bool| {
            let view = render(&world, kind, 64).unwrap();
            let pixels = pixels_of(&view);
            let (mut sum, mut count) = (0.0f32, 0.0f32);
            for y in 0..view.size as usize {
                for x in 0..view.size as usize {
                    let at = (y * view.size as usize + x) * 3;
                    let p = &pixels[at..at + 3];
                    if p == BACKGROUND {
                        continue;
                    }
                    if (p[0] > p[2]) == red_wins {
                        sum += x as f32;
                        count += 1.0;
                    }
                }
            }
            assert!(count > 0.0, "{kind:?}: expected to see that voxel at all");
            (sum / count) - view.size as f32 / 2.0
        };

        // Front puts +X on the right, Back mirrors it, Top keeps it on
        // the right. Each is asserted on the blue voxel — the one at +X.
        assert!(side_of(ViewKind::Front, false) > 0.0, "front: +X is right");
        assert!(side_of(ViewKind::Back, false) < 0.0, "back: +X is left");
        assert!(side_of(ViewKind::Top, false) > 0.0, "top: +X is right");
        // …and the red one has to be on the other side, or the test
        // would pass on an image that drew only one voxel.
        assert!(side_of(ViewKind::Front, true) < 0.0, "front: origin is left");

        // Left looks along +X and the two are in line, so the near one
        // — the red at the origin — is all that's visible.
        let from_left = render(&world, ViewKind::Left, 64).unwrap();
        assert!(
            pixels_of(&from_left)
                .chunks(3)
                .filter(|p| p != &BACKGROUND)
                .all(|p| p[0] > p[2]),
            "left: the origin voxel is nearer, so it hides the one at +X"
        );
    }

    #[test]
    fn the_near_face_is_lit_and_the_shape_is_shaded() {
        // A flat wall must not come out as one flat color: without
        // lambert + AO an agent can't tell a box from a plane.
        let world = cube(4, Voxel::from_rgb(180, 180, 180));
        let view = render(&world, ViewKind::Iso, 96).unwrap();
        let distinct: std::collections::HashSet<[u8; 3]> = pixels_of(&view)
            .chunks(3)
            .map(|p| [p[0], p[1], p[2]])
            .filter(|p| p != &BACKGROUND)
            .collect();
        assert!(
            distinct.len() >= 3,
            "the three visible faces of a cube should differ, got {} tones",
            distinct.len()
        );
    }

    #[test]
    fn every_view_lights_the_side_it_looks_at() {
        // The bug this pins: with a light fixed in world space, `back`,
        // `left` and `bottom` return a silhouette in flat ambient. Those
        // are exactly the views someone asks for when they want to see
        // that side, so "correctly lit and unreadable" is a failure.
        let world = cube(3, Voxel::from_rgb(200, 200, 200));
        for kind in ViewKind::ALL {
            let view = render(&world, kind, 48).unwrap();
            let brightest = pixels_of(&view)
                .chunks(3)
                .filter(|p| p != &BACKGROUND)
                .map(|p| p[0])
                .max()
                .expect("the cube should be visible");
            let ambient_only = (200.0 * AMBIENT) as u8;
            assert!(
                brightest > ambient_only + 60,
                "{kind:?}: brightest pixel {brightest} is barely above the \
                 ambient floor {ambient_only} — this view is unlit"
            );
        }
    }

    #[test]
    fn an_emissive_voxel_keeps_its_full_color() {
        // The flag says "this glows"; shading it like paint would hide
        // the one thing it was set for.
        let mut voxel = Voxel::from_rgb(250, 200, 40);
        voxel.set_emissive(true);
        let world = cube(1, voxel);
        let view = render(&world, ViewKind::Front, 32).unwrap();
        assert!(
            pixels_of(&view)
                .chunks(3)
                .any(|p| p == [250, 200, 40]),
            "an emissive voxel should reach the image unshaded"
        );
    }

    #[test]
    fn framing_maps_the_image_back_to_cells() {
        // The whole reason a render is actionable: the agent has to be
        // able to turn "two pixels too high" into "one cell too high".
        let world = cube(5, Voxel::from_rgb(100, 100, 100));
        let view = render(&world, ViewKind::Front, 128).unwrap();
        let (min, max) = view.framing.bounds.unwrap();
        assert_eq!(min, (-5, -5, -5));
        assert_eq!(max, (5, 5, 5));
        // The model is 11 cells wide and the frame adds the 2.5% margin,
        // so the image covers 11 × 1.025 cells however many pixels that
        // is — the number an agent multiplies a pixel offset by.
        let spanned = view.framing.cells_per_pixel * view.size as f32;
        assert!(
            (spanned - 11.275).abs() < 0.01,
            "the frame should span the model plus the margin, spans {spanned}"
        );
        assert_eq!(view.framing.right, [1.0, 0.0, 0.0]);
        assert_eq!(view.framing.up, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn a_size_outside_the_range_is_refused_rather_than_clamped() {
        // Same rule as the ops limits: a silently smaller image is a
        // wrong answer to the question that was asked.
        assert_eq!(
            render(&World::new(), ViewKind::Iso, 0).unwrap_err(),
            ViewError::SizeOutOfRange(0)
        );
        assert_eq!(
            render(&World::new(), ViewKind::Iso, MAX_SIZE + 1).unwrap_err(),
            ViewError::SizeOutOfRange(MAX_SIZE + 1)
        );
        assert!(render(&World::new(), ViewKind::Iso, MAX_SIZE).is_ok());
    }

    #[test]
    fn view_names_round_trip() {
        for kind in ViewKind::ALL {
            assert_eq!(ViewKind::from_name(kind.as_str()), Some(kind));
            // The wire name and serde's name are the same string, so a
            // CLI flag and a JSON field can't drift apart.
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, format!("\"{}\"", kind.as_str()));
        }
        assert_eq!(ViewKind::from_name("sideways"), None);
    }
}
