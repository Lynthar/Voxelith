//! The wire format: what an agent sends, as serde types.
//!
//! Every type here is `deny_unknown_fields`. That is the opposite of
//! the `#[serde(default)]` forward-compat discipline `prefs.ron` and
//! `.vxlt` follow, and deliberately so: those formats read files an
//! *older* build wrote, while this one reads a request a language
//! model just invented. A hallucinated field that silently does
//! nothing is the worst outcome available — the agent believes it set
//! something, the world disagrees, and nothing in the report says why.
//! Rejecting it names the mistake.
//!
//! Coordinates are world cell coordinates, Y up. Regions are inclusive
//! on both ends (`min` and `max` are cells *in* the region), matching
//! [`Selection`] and the drag-shape tools.
//!
//! With the `mcp` feature these types also derive `JsonSchema`, because
//! an MCP client learns this format from the tool schema and nowhere
//! else. Generating it from the types is the same discipline the
//! generator registry follows — a hand-written copy is a second source
//! of truth that starts drifting the day after it's written. Only
//! [`VoxelSpec`] needs a hand-written schema, since it also needs a
//! hand-written `Deserialize`; a test pins the two together.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

use crate::core::Voxel;
use crate::editor::{Axis, Quarter, Selection};

use super::{ErrorCode, OpsError};

/// A batch of ops plus its envelope.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub struct OpsBatch {
    /// Must equal [`SCHEMA_VERSION`](super::SCHEMA_VERSION).
    pub version: u32,
    pub ops: Vec<Op>,
    #[serde(default)]
    pub options: BatchOptions,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub struct BatchOptions {
    /// Run everything, report what would happen, commit nothing.
    #[serde(default)]
    pub dry_run: bool,
}

/// Inclusive integer box. `min`/`max` may arrive in either order — they
/// are normalized to opposite corners on the way in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub struct Aabb {
    pub min: [i32; 3],
    pub max: [i32; 3],
}

impl Aabb {
    pub fn to_selection(self) -> Selection {
        Selection::from_corners(
            (self.min[0], self.min[1], self.min[2]),
            (self.max[0], self.max[1], self.max[2]),
        )
    }

    pub(super) fn from_pair(pair: ((i32, i32, i32), (i32, i32, i32))) -> Self {
        let ((x0, y0, z0), (x1, y1, z1)) = pair;
        Self {
            min: [x0, y0, z0],
            max: [x1, y1, z1],
        }
    }
}

impl From<Selection> for Aabb {
    fn from(sel: Selection) -> Self {
        Self::from_pair((sel.min, sel.max))
    }
}

/// World axis, lowercase on the wire (`"x"` / `"y"` / `"z"`).
///
/// A wire-format twin of [`Axis`] rather than serde derives on `Axis`
/// itself: the JSON spelling is this protocol's business, and pinning
/// it on the shared editor type would make every future serialization
/// of an axis inherit this format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub enum AxisSpec {
    X,
    #[default]
    Y,
    Z,
}

impl AxisSpec {
    pub fn to_axis(self) -> Axis {
        match self {
            AxisSpec::X => Axis::X,
            AxisSpec::Y => Axis::Y,
            AxisSpec::Z => Axis::Z,
        }
    }

    /// Component index into an `[i32; 3]` / `(i32, i32, i32)`.
    pub fn index(self) -> usize {
        match self {
            AxisSpec::X => 0,
            AxisSpec::Y => 1,
            AxisSpec::Z => 2,
        }
    }
}

/// Which cells a write is allowed to land on.
///
/// `only_solid` is how you repaint without changing the silhouette —
/// there's no separate `paint` op because `box` + `only_solid` already
/// is one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub enum WriteMode {
    /// Write unconditionally.
    #[default]
    Replace,
    /// Only write where the cell is currently air (build around what's
    /// already there).
    OnlyAir,
    /// Only write where the cell is currently solid (recolor).
    OnlySolid,
}

/// A voxel value: the string `"air"`, or an object describing a solid.
#[derive(Debug, Clone, PartialEq)]
pub enum VoxelSpec {
    Air,
    Solid(SolidVoxel),
}

/// The object form of [`VoxelSpec`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub struct SolidVoxel {
    pub rgb: [u8; 3],
    /// Accepted only as `255`. Every voxel in the world is opaque — the
    /// greedy mesher's "no visible face" sentinel and the flood fill
    /// both depend on it — so the field exists to give an agent that
    /// thinks in RGBA a real answer instead of an "unknown field".
    #[serde(default)]
    pub a: Option<u8>,
    /// Material id, default 1 (what the editor's brush uses). 0 is air.
    #[serde(default)]
    pub material: Option<u16>,
    #[serde(default)]
    pub emissive: bool,
    #[serde(default)]
    pub metallic: bool,
    /// Faction recolor zone, 0..=3 (none / primary / secondary /
    /// reserved). Exported per-vertex as `_TINTZONE`.
    #[serde(default)]
    pub tint_zone: u8,
}

impl<'de> Deserialize<'de> for VoxelSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Hand-written rather than `#[serde(untagged)]`: untagged's
        // "data did not match any variant" tells an agent nothing about
        // which half it got wrong.
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(s) if s == "air" => Ok(VoxelSpec::Air),
            serde_json::Value::String(s) => Err(D::Error::custom(format!(
                "unknown voxel {s:?}; the only string form is \"air\", \
                 and a solid voxel is an object like {{\"rgb\": [200, 100, 50]}}"
            ))),
            value @ serde_json::Value::Object(_) => SolidVoxel::deserialize(value)
                .map(VoxelSpec::Solid)
                .map_err(D::Error::custom),
            _ => Err(D::Error::custom(
                "voxel must be \"air\" or an object like {\"rgb\": [200, 100, 50]}",
            )),
        }
    }
}

/// Hand-written for the same reason [`Deserialize`] is: this type is
/// two shapes, and no derive infers a schema for a custom deserializer.
///
/// It is the only written-by-hand schema in the protocol, so it's the
/// only one that can drift from what the code accepts — hence the test
/// below, and hence the object half being `SolidVoxel`'s own generated
/// schema by reference rather than a transcription of it.
#[cfg(feature = "mcp")]
impl schemars::JsonSchema for VoxelSpec {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "VoxelSpec".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        concat!(module_path!(), "::VoxelSpec").into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let solid = generator.subschema_for::<SolidVoxel>();
        schemars::json_schema!({
            "description": "A voxel value: the string \"air\" to clear a cell, or an object describing a solid one.",
            "anyOf": [
                { "type": "string", "const": "air" },
                solid
            ]
        })
    }
}

impl VoxelSpec {
    /// Resolve to the value that will be written. Range checks live
    /// here rather than in `Deserialize` so their messages can name the
    /// invariant they protect.
    pub fn to_voxel(&self) -> Result<Voxel, OpsError> {
        let spec = match self {
            VoxelSpec::Air => return Ok(Voxel::AIR),
            VoxelSpec::Solid(spec) => spec,
        };
        if let Some(a) = spec.a {
            if a != 255 {
                return Err(OpsError::new(
                    ErrorCode::InvalidArgument,
                    format!("alpha {a} is not allowed: every voxel in the world is opaque (a = 255); use \"air\" to clear a cell"),
                ));
            }
        }
        let material = spec.material.unwrap_or(1);
        if material == 0 {
            return Err(OpsError::new(
                ErrorCode::InvalidArgument,
                "material 0 means air; write \"air\" instead, or use material 1",
            ));
        }
        if spec.tint_zone > 3 {
            return Err(OpsError::new(
                ErrorCode::InvalidArgument,
                format!("tint_zone {} is out of range 0..=3", spec.tint_zone),
            ));
        }
        let mut voxel = Voxel::new(material, spec.rgb[0], spec.rgb[1], spec.rgb[2]);
        voxel.set_emissive(spec.emissive);
        voxel.set_metallic(spec.metallic);
        voxel.set_tint_zone(spec.tint_zone);
        Ok(voxel)
    }
}

/// One entry of a `set_voxels` op: `[x, y, z, voxel]`.
///
/// A positional array, not `{"pos": …, "voxel": …}`, because this op
/// exists to carry thousands of them and every repeated key is tokens
/// the agent pays for twice (writing them, then re-reading them).
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub struct VoxelEntry(pub i32, pub i32, pub i32, pub VoxelSpec);

/// One operation. Tagged by `"op"`.
///
/// Every shape op takes a `voxel`, and `"air"` is a legal value — so
/// `box` with `"air"` is "erase this region" and there is no separate
/// erase op.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub enum Op {
    /// Axis-aligned box. `filled: false` leaves a 1-cell shell.
    Box {
        min: [i32; 3],
        max: [i32; 3],
        voxel: VoxelSpec,
        #[serde(default = "yes")]
        filled: bool,
        #[serde(default)]
        write_mode: WriteMode,
    },
    /// Sphere of `radius` around `center` (diameter `2·radius + 1`).
    Sphere {
        center: [i32; 3],
        radius: i32,
        voxel: VoxelSpec,
        #[serde(default = "yes")]
        filled: bool,
        #[serde(default)]
        write_mode: WriteMode,
    },
    /// Cylinder standing on `base` (its axis-perpendicular center),
    /// extending `height` cells along +`axis`.
    Cylinder {
        base: [i32; 3],
        radius: i32,
        height: i32,
        #[serde(default)]
        axis: AxisSpec,
        voxel: VoxelSpec,
        #[serde(default = "yes")]
        filled: bool,
        #[serde(default)]
        write_mode: WriteMode,
    },
    /// 1-cell-thick line, both endpoints included.
    Line {
        from: [i32; 3],
        to: [i32; 3],
        voxel: VoxelSpec,
        #[serde(default)]
        write_mode: WriteMode,
    },
    /// Clear every cell in the region that is fully enclosed by solid
    /// neighbors — the cheap way to turn a solid block into a shell
    /// before exporting it.
    Hollow { min: [i32; 3], max: [i32; 3] },
    /// Explicit per-voxel writes, for detail work.
    SetVoxels {
        voxels: Vec<VoxelEntry>,
        #[serde(default)]
        write_mode: WriteMode,
    },
    /// Run a registered generator and write its patch. `params` are
    /// merged over the generator's defaults, so only the ones you care
    /// about need naming.
    Generate {
        generator: String,
        #[serde(default)]
        params: serde_json::Value,
        /// Offset applied to the whole patch, since not every generator
        /// takes an origin parameter.
        #[serde(default)]
        translate: [i32; 3],
        #[serde(default)]
        write_mode: WriteMode,
    },
    /// Store a procedural pipeline graph on the document and write what
    /// it produces.
    ///
    /// A graph is `{"nodes": [ … ]}`, one flat object per node:
    /// `{"id": 0, "kind": "builtin.perlin_terrain", "width": 32}`,
    /// `{"id": 1, "kind": "filter", "input": 0, "predicate": {"y_above": 4}}`,
    /// `{"id": 2, "kind": "output", "input": 1}`. Source nodes are named
    /// by generator id and take that generator's parameters directly —
    /// only the ones you want to differ. Transform nodes are
    /// `translate` / `filter` / `mask` / `combine`, and exactly one
    /// `output` node marks what the pipeline emits. Call
    /// `list_generators` for a ready-made graph to copy.
    ///
    /// The graph is kept with the project, so a human can open it in the
    /// editor's Graph panel afterwards and keep tuning the parameters —
    /// which is the point of sending one instead of voxels. Set
    /// `apply: false` to store it without evaluating, for building a
    /// graph up over several batches.
    Graph {
        graph: serde_json::Value,
        #[serde(default = "yes")]
        apply: bool,
        /// Offset applied to the whole patch, like `generate`'s.
        #[serde(default)]
        translate: [i32; 3],
        #[serde(default)]
        write_mode: WriteMode,
    },
    /// Change the graph the document already has, instead of resending
    /// the whole thing.
    ///
    /// `describe` reports the current graph; these edits name nodes by
    /// the ids it shows. The edits run in order against a copy, so a
    /// batch that fails part-way leaves the graph exactly as it was —
    /// a wire that would close a cycle is refused and nothing else in
    /// the list has to be undone by hand. Same `apply` rule as `graph`:
    /// on by default, `false` to edit without evaluating.
    GraphEdit {
        edits: Vec<GraphEdit>,
        #[serde(default = "yes")]
        apply: bool,
        #[serde(default)]
        translate: [i32; 3],
        #[serde(default)]
        write_mode: WriteMode,
    },
    /// Set the session selection (later ops can then omit `region`).
    Select { min: [i32; 3], max: [i32; 3] },
    /// Clear the session selection.
    Deselect,
    /// Rotate a region's contents in place, in right-handed +90° steps.
    /// `min` stays put; the AABB's extents swap for odd quarter turns.
    Rotate {
        axis: AxisSpec,
        /// 1 = +90°, 2 = 180°, 3 = 270°.
        quarters: u8,
        #[serde(default)]
        region: Option<Aabb>,
    },
    /// Flip a region's contents across its own midplane.
    Mirror {
        axis: AxisSpec,
        #[serde(default)]
        region: Option<Aabb>,
    },
    /// Reflect a region's solid voxels to the other side of a plane,
    /// keeping the original. Build one half, mirror it, get a
    /// symmetric model.
    MirrorCopy {
        axis: AxisSpec,
        /// The mirror plane, given as the **seam between cell
        /// `plane - 1` and cell `plane`** — cell `p` lands at
        /// `2·plane − 1 − p`. Integer because a voxel is a cell
        /// `[p, p+1)`, not a point: the seams are exactly the integers,
        /// and a plane through a cell's *middle* would only map that
        /// cell onto itself. Defaults to the seam just past the
        /// region's `max` on `axis`, i.e. the copy sits flush against
        /// the original. `0` mirrors across the world origin, matching
        /// the editor's symmetry planes.
        #[serde(default)]
        plane: Option<i32>,
        #[serde(default)]
        region: Option<Aabb>,
        #[serde(default)]
        write_mode: WriteMode,
    },
}

/// One change to the document's pipeline graph.
///
/// Six verbs, each mapping to something `PipelineGraph` already does,
/// which is why this is a vocabulary rather than a second graph API:
/// the cycle check on `connect` is the editor's own, so an agent and a
/// human dragging a wire are refused by the same code.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "edit", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "mcp", derive(schemars::JsonSchema))]
pub enum GraphEdit {
    /// Add a node. `node` is one graph node — the same object a `graph`
    /// op carries, `{"id": 3, "kind": "translate", "input": 1, "dy": 8}`
    /// — and its id must not already be in the graph.
    AddNode { node: serde_json::Value },
    /// Remove a node. Wires pointing at it are cleared, so the rest of
    /// the graph stays consistent.
    RemoveNode { id: u32 },
    /// Change a node's parameters in place. Only the ones you name;
    /// the rest keep their values. `kind` can't be changed this way —
    /// remove the node and add the one you meant.
    SetParams { id: u32, params: serde_json::Value },
    /// Wire `source`'s output into `target`'s input `slot` (0 for
    /// single-input nodes; `mask` and `combine` take 0 and 1).
    Connect {
        target: u32,
        #[serde(default)]
        slot: usize,
        source: u32,
    },
    /// Clear one input slot.
    Disconnect {
        target: u32,
        #[serde(default)]
        slot: usize,
    },
    /// Throw the whole graph away and start over.
    Clear,
}

fn yes() -> bool {
    true
}

/// Quarter turns as an agent writes them (1/2/3) → the editor's enum.
pub(super) fn quarter_from(quarters: u8) -> Result<Quarter, OpsError> {
    match quarters {
        1 => Ok(Quarter::Cw),
        2 => Ok(Quarter::Half),
        3 => Ok(Quarter::Ccw),
        other => Err(OpsError::new(
            ErrorCode::InvalidArgument,
            format!("quarters must be 1 (+90°), 2 (180°) or 3 (270°), got {other}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_op(json: &str) -> Result<Op, serde_json::Error> {
        serde_json::from_str(json)
    }

    #[test]
    fn a_misspelled_field_is_named_rather_than_ignored() {
        // The reason every type here denies unknown fields: an agent
        // that writes `filed` believes it asked for a shell, and a
        // silently ignored key would hand it a solid block with a
        // report saying everything went fine.
        let error =
            parse_op(r#"{"op":"box","min":[0,0,0],"max":[1,1,1],"voxel":"air","filed":false}"#)
                .expect_err("unknown field must be refused");
        assert!(
            error.to_string().contains("filed"),
            "the message should name the bad key, got: {error}"
        );
    }

    #[test]
    fn an_unknown_op_is_refused() {
        assert!(parse_op(r#"{"op":"extrude","min":[0,0,0]}"#).is_err());
    }

    #[test]
    fn optional_fields_default_the_way_the_editor_behaves() {
        let op = parse_op(r#"{"op":"box","min":[0,0,0],"max":[1,1,1],"voxel":"air"}"#).unwrap();
        match op {
            Op::Box {
                filled, write_mode, ..
            } => {
                assert!(filled, "shapes are solid unless asked otherwise");
                assert_eq!(write_mode, WriteMode::Replace);
            }
            other => panic!("parsed as {other:?}"),
        }
        let op =
            parse_op(r#"{"op":"cylinder","base":[0,0,0],"radius":1,"height":2,"voxel":"air"}"#)
                .unwrap();
        match op {
            Op::Cylinder { axis, .. } => assert_eq!(axis, AxisSpec::Y, "cylinders stand up"),
            other => panic!("parsed as {other:?}"),
        }
    }

    #[test]
    fn air_is_the_only_string_a_voxel_can_be() {
        assert_eq!(
            serde_json::from_str::<VoxelSpec>(r#""air""#).unwrap(),
            VoxelSpec::Air
        );
        let error = serde_json::from_str::<VoxelSpec>(r#""empty""#)
            .expect_err("only \"air\" is a voxel string");
        assert!(
            error.to_string().contains("\"air\""),
            "the message should point at the spelling that works, got: {error}"
        );
    }

    #[test]
    fn a_solid_voxel_carries_color_flags_and_zone() {
        let spec: VoxelSpec = serde_json::from_str(
            r#"{"rgb":[10,20,30],"material":4,"emissive":true,"metallic":true,"tint_zone":3}"#,
        )
        .unwrap();
        let voxel = spec.to_voxel().unwrap();
        assert_eq!(voxel.color(), [10, 20, 30, 255]);
        assert_eq!(voxel.material, 4);
        assert!(voxel.is_emissive() && voxel.is_metallic());
        assert_eq!(voxel.tint_zone(), 3);
    }

    #[test]
    fn a_translucent_voxel_is_refused_not_quietly_forced_opaque() {
        // Every voxel in the world is opaque; silently rewriting 128 to
        // 255 would leave the agent believing it made glass.
        let spec: VoxelSpec = serde_json::from_str(r#"{"rgb":[1,2,3],"a":128}"#).unwrap();
        let error = spec.to_voxel().expect_err("alpha 128 must be refused");
        assert_eq!(error.code, ErrorCode::InvalidArgument);

        // The field itself is accepted, so an agent that thinks in RGBA
        // gets an answer instead of "unknown field `a`".
        let opaque: VoxelSpec = serde_json::from_str(r#"{"rgb":[1,2,3],"a":255}"#).unwrap();
        assert!(opaque.to_voxel().is_ok());
    }

    #[test]
    fn material_zero_and_a_bad_tint_zone_are_refused() {
        let air_material: VoxelSpec =
            serde_json::from_str(r#"{"rgb":[1,2,3],"material":0}"#).unwrap();
        assert_eq!(
            air_material.to_voxel().unwrap_err().code,
            ErrorCode::InvalidArgument
        );
        let bad_zone: VoxelSpec = serde_json::from_str(r#"{"rgb":[1,2,3],"tint_zone":4}"#).unwrap();
        assert_eq!(
            bad_zone.to_voxel().unwrap_err().code,
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn quarter_turns_map_to_the_editors_right_handed_steps() {
        assert_eq!(quarter_from(1).unwrap(), Quarter::Cw);
        assert_eq!(quarter_from(2).unwrap(), Quarter::Half);
        assert_eq!(quarter_from(3).unwrap(), Quarter::Ccw);
        for bad in [0, 4, 255] {
            assert_eq!(
                quarter_from(bad).unwrap_err().code,
                ErrorCode::InvalidArgument
            );
        }
    }

    #[test]
    fn corners_may_arrive_in_either_order() {
        let region = Aabb {
            min: [5, 5, 5],
            max: [0, 0, 0],
        }
        .to_selection();
        assert_eq!(region.min, (0, 0, 0));
        assert_eq!(region.max, (5, 5, 5));
    }

    #[test]
    fn a_stray_key_in_the_envelope_is_refused() {
        assert!(serde_json::from_str::<OpsBatch>(
            r#"{"version":1,"ops":[{"op":"deselect"}],"dryrun":true}"#
        )
        .is_err());
        assert!(serde_json::from_str::<OpsBatch>(
            r#"{"version":1,"ops":[{"op":"deselect"}],"options":{"dry":true}}"#
        )
        .is_err());
    }
}
