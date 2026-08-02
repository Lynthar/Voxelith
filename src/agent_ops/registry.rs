//! The generator table — what an agent can name in a `generate` op.
//!
//! Parameters are not described by a hand-written JSON Schema. Each
//! entry serializes its generator's `Default` and hands *that* over as
//! the template: it is the real parameter set by construction, it can't
//! drift from the struct, and an agent can copy it, change two fields,
//! and send it back. Partial params are merged over the defaults, so a
//! request only names what it wants to differ.

use std::sync::OnceLock;

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use crate::procgen::{GeneratorMeta, LSystemTree, PerlinTerrain, VoxelGenerator, WfcGenerator};

use super::{ErrorCode, OpsError};

/// Ceiling on `width × depth × height-span` for terrain. Not a taste
/// judgment — `PerlinTerrain` sizes its patch buffer from these three
/// numbers, so `width: 100000` is an out-of-memory abort before any
/// budget downstream gets a look.
const MAX_TERRAIN_CELLS: i64 = 4_194_304;

/// Ceiling on WFC grid area, in tiles (each tile is 4³ voxels). Same
/// reason: the solver allocates the whole grid up front.
const MAX_WFC_TILES: u64 = 4096;

pub(super) struct GeneratorEntry {
    pub meta: GeneratorMeta,
    pub default_params: fn() -> Value,
    pub build: fn(&Value) -> Result<Box<dyn VoxelGenerator>, OpsError>,
}

/// The registered generators.
///
/// Built once at first use rather than declared as a `static` table so
/// each entry's metadata comes from the generator's own `metadata()` —
/// one source of truth for ids, which is what `generate` dispatches on.
pub(super) fn registry() -> &'static [GeneratorEntry] {
    static REGISTRY: OnceLock<Vec<GeneratorEntry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        vec![
            GeneratorEntry {
                meta: PerlinTerrain::default().metadata(),
                default_params: defaults::<PerlinTerrain>,
                build: build_terrain,
            },
            GeneratorEntry {
                meta: LSystemTree::default().metadata(),
                default_params: defaults::<LSystemTree>,
                build: build_tree,
            },
            GeneratorEntry {
                meta: WfcGenerator::default().metadata(),
                default_params: defaults::<WfcGenerator>,
                build: build_wfc,
            },
        ]
    })
}

/// Instantiate a generator by id with partial params merged over its
/// defaults.
pub(super) fn build(id: &str, params: &Value) -> Result<Box<dyn VoxelGenerator>, OpsError> {
    let entry = registry()
        .iter()
        .find(|entry| entry.meta.id == id)
        .ok_or_else(|| {
            let known: Vec<&str> = registry().iter().map(|e| e.meta.id).collect();
            OpsError::new(
                ErrorCode::UnknownGenerator,
                format!("unknown generator {id:?}; registered: {}", known.join(", ")),
            )
        })?;
    // The builders don't know their own name; the dispatcher does, and
    // an agent staring at a params error wants it in the message.
    (entry.build)(params).map_err(|mut e| {
        e.message = format!("{id}: {}", e.message);
        e
    })
}

/// One generator as `list_generators` reports it.
#[derive(Debug, Clone, Serialize)]
pub struct GeneratorInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub category: String,
    /// Every parameter at its default value — copy, edit, send back.
    pub default_params: Value,
}

pub fn generator_infos() -> Vec<GeneratorInfo> {
    registry()
        .iter()
        .map(|entry| GeneratorInfo {
            id: entry.meta.id,
            name: entry.meta.name,
            description: entry.meta.description,
            category: format!("{:?}", entry.meta.category),
            default_params: (entry.default_params)(),
        })
        .collect()
}

fn defaults<T: Default + Serialize>() -> Value {
    serde_json::to_value(T::default()).expect("generator params must serialize to JSON")
}

/// Merge `params` over `T::default()` and deserialize.
///
/// Unknown keys are rejected *here* rather than with
/// `#[serde(deny_unknown_fields)]` on the generator structs: those same
/// structs are embedded in `prefs.ron` and `.vxlt`, whose forward
/// compatibility depends on unknown fields being ignored. The strict
/// check belongs to this protocol, not to the storage format.
///
/// Only top-level keys are checked, which covers every generator today
/// (their params are flat — nested values are arrays and enum strings,
/// not objects).
fn from_partial<T: Default + Serialize + DeserializeOwned>(params: &Value) -> Result<T, OpsError> {
    let mut merged = match serde_json::to_value(T::default()) {
        Ok(Value::Object(map)) => map,
        _ => panic!("generator params must serialize to a JSON object"),
    };
    match params {
        Value::Null => {}
        Value::Object(overrides) => {
            for (key, value) in overrides {
                if !merged.contains_key(key) {
                    let valid: Vec<&str> = merged.keys().map(String::as_str).collect();
                    return Err(OpsError::new(
                        ErrorCode::InvalidParams,
                        format!(
                            "unknown param {key:?}; valid params are {}",
                            valid.join(", ")
                        ),
                    ));
                }
                merged.insert(key.clone(), value.clone());
            }
        }
        _ => {
            return Err(OpsError::new(
                ErrorCode::InvalidParams,
                "params must be a JSON object",
            ))
        }
    }
    serde_json::from_value(Value::Object(merged))
        .map_err(|e| OpsError::new(ErrorCode::InvalidParams, e.to_string()))
}

fn build_terrain(params: &Value) -> Result<Box<dyn VoxelGenerator>, OpsError> {
    let terrain: PerlinTerrain = from_partial(params)?;
    let span = (terrain.max_height as i64 - terrain.min_height as i64 + 1).max(1);
    let cells = terrain.width as i64 * terrain.depth as i64 * span;
    if cells > MAX_TERRAIN_CELLS {
        return Err(OpsError::new(
            ErrorCode::InvalidParams,
            format!(
                "{}×{} over a height span of {} is {} cells; at most {} are allowed per generate",
                terrain.width, terrain.depth, span, cells, MAX_TERRAIN_CELLS
            ),
        ));
    }
    Ok(Box::new(terrain))
}

fn build_tree(params: &Value) -> Result<Box<dyn VoxelGenerator>, OpsError> {
    // No size cap: `LSystemTree` already refuses more than 7 rewrite
    // rounds, and 7 rounds is a few hundred thousand voxels — inside
    // the batch cell budget, which catches it downstream.
    let tree: LSystemTree = from_partial(params)?;
    Ok(Box::new(tree))
}

fn build_wfc(params: &Value) -> Result<Box<dyn VoxelGenerator>, OpsError> {
    let wfc: WfcGenerator = from_partial(params)?;
    let tiles = wfc.width as u64 * wfc.depth as u64;
    if tiles > MAX_WFC_TILES {
        return Err(OpsError::new(
            ErrorCode::InvalidParams,
            format!(
                "{}×{} is {} tiles; at most {} are allowed per generate",
                wfc.width, wfc.depth, tiles, MAX_WFC_TILES
            ),
        ));
    }
    Ok(Box::new(wfc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `Box<dyn VoxelGenerator>` isn't `Debug`, so drop the success
    /// value before asking for the error.
    fn build_err(id: &str, params: &Value) -> OpsError {
        build(id, params)
            .map(|_| ())
            .expect_err("this should have been refused")
    }

    #[test]
    fn the_advertised_defaults_are_accepted_verbatim() {
        // The contract of "defaults are the template": whatever
        // `list_generators` hands an agent must come straight back in a
        // `generate` op without editing.
        for info in generator_infos() {
            assert!(
                info.default_params.is_object(),
                "{} advertises {:?}, which an agent can't edit and resend",
                info.id,
                info.default_params
            );
            build(info.id, &info.default_params)
                .unwrap_or_else(|e| panic!("{} rejected its own defaults: {e}", info.id));
        }
    }

    #[test]
    fn every_generator_is_reachable_by_the_id_it_reports() {
        let ids: Vec<&str> = generator_infos().iter().map(|info| info.id).collect();
        assert_eq!(
            ids,
            vec!["builtin.perlin_terrain", "builtin.lsystem_tree", "builtin.wfc"]
        );
    }

    #[test]
    fn partial_params_land_on_top_of_the_defaults() {
        let terrain: PerlinTerrain = from_partial(&json!({"seed": 99})).unwrap();
        assert_eq!(terrain.seed, 99);
        assert_eq!(
            terrain.width,
            PerlinTerrain::default().width,
            "unnamed params must keep their default"
        );
    }

    #[test]
    fn omitted_params_are_the_same_as_an_empty_object() {
        let implicit: PerlinTerrain = from_partial(&Value::Null).unwrap();
        assert_eq!(implicit, PerlinTerrain::default());
    }

    #[test]
    fn a_misspelled_param_lists_the_real_ones() {
        let error = build_err("builtin.perlin_terrain", &json!({"octave": 3}));
        assert_eq!(error.code, ErrorCode::InvalidParams);
        assert!(
            error.message.contains("octaves") && error.message.contains("builtin.perlin_terrain"),
            "the message should name the generator and the valid keys, got: {}",
            error.message
        );
    }

    #[test]
    fn a_param_of_the_wrong_type_is_refused() {
        let error = build_err("builtin.lsystem_tree", &json!({"iterations": "four"}));
        assert_eq!(error.code, ErrorCode::InvalidParams);
    }

    #[test]
    fn params_that_are_not_an_object_are_refused() {
        assert_eq!(
            build_err("builtin.wfc", &json!([1, 2, 3])).code,
            ErrorCode::InvalidParams
        );
    }

    #[test]
    fn a_generator_big_enough_to_exhaust_memory_is_refused_before_it_runs() {
        // Both caps guard an eager allocation sized straight from the
        // params — there is no downstream budget that gets a look first.
        let terrain = build_err(
            "builtin.perlin_terrain",
            &json!({"width": 100000, "depth": 100000}),
        );
        assert_eq!(terrain.code, ErrorCode::InvalidParams);

        let wfc = build_err("builtin.wfc", &json!({"width": 4096, "depth": 4096}));
        assert_eq!(wfc.code, ErrorCode::InvalidParams);

        // …and the sizes a human would actually ask for still pass.
        assert!(build("builtin.perlin_terrain", &json!({"width": 256, "depth": 256})).is_ok());
        assert!(build("builtin.wfc", &json!({"width": 24, "depth": 24})).is_ok());
    }
}
