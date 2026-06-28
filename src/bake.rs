//! Headless batch export ("bake"): turn many `.vxlt` sources into
//! optimized, engine-ready `.glb` assets from one declarative spec, so
//! re-exporting a whole art set after a tweak is a single command instead
//! of N interactive dialog trips. See `docs/GAME_PIPELINE_ROADMAP.md` §3.5.
//!
//! The bake is CPU-only: it reuses the same mesh + [`crate::io::gltf`]
//! export path the interactive UI uses (it operates on `World` / mesh
//! data, never the wgpu render context), so it needs no GPU and no window.
//! `main.rs` routes `voxelith bake <spec.json>` here before the winit /
//! egui app is ever constructed.
//!
//! Pipeline per item:
//! 1. load `.vxlt` → `World` + `EditorState` ([`crate::io::load_world_with_state`]);
//! 2. export `.glb` (greedy, or Marching Cubes when `smoothing` is set)
//!    with a deterministic placement transform (pivot / up-axis /
//!    unit-scale, §3.5);
//! 3. optional geometry compression via `gltfpack` (meshopt, §3.4);
//! 4. write a per-item JSON report next to the output.
//!
//! ## Spec schema
//!
//! ```jsonc
//! {
//!   "defaults": { "mesher": "greedy", "smoothing": "none",
//!                 "up_axis": "y", "unit_scale": 1.0,
//!                 "pivot": "base-center", "optimize": "meshopt" },
//!   "items": [
//!     { "src": "buildings/farm.vxlt", "out": "buildings/farm.glb" },
//!     { "src": "chars/knight.vxlt",   "out": "chars/knight.glb", "pivot": "feet" },
//!     { "srcDir": "creatures/", "outDir": "creatures/" }   // one item per .vxlt
//!   ]
//! }
//! ```
//!
//! Per-item fields override the matching `defaults`; anything unset falls
//! back to the *tool* defaults (identity placement, greedy mesh, no
//! optimize), so a minimal `{ "items": [...] }` reproduces the interactive
//! export. Paths are resolved relative to the spec file's directory.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::editor::Socket;
use crate::io::{self, ExportTransform, Pivot, SocketNode, UpAxis};

/// A spec-level failure that aborts the whole bake before any item runs
/// (unreadable / invalid spec, bad `--shard`, unreadable `srcDir`).
/// Per-item failures do *not* surface here — they're captured in that
/// item's [`ItemReport`] so one bad model never sinks the batch.
#[derive(Debug)]
pub enum BakeError {
    Spec(String),
}

impl std::fmt::Display for BakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BakeError::Spec(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for BakeError {}

// ===========================================================================
// Spec (deserialized straight from the JSON file)
// ===========================================================================

/// The raw spec file: a `defaults` block plus a list of items.
#[derive(Debug, Clone, Deserialize)]
pub struct BakeSpec {
    #[serde(default)]
    pub defaults: Settings,
    #[serde(default)]
    pub items: Vec<RawItem>,
}

/// The tunable knobs, all optional so the `defaults` block and per-item
/// overrides share one shape and merge field-by-field. Unknown keys are
/// ignored (forward-compatible).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub mesher: Option<String>,
    pub smoothing: Option<String>,
    pub up_axis: Option<String>,
    pub unit_scale: Option<f32>,
    pub pivot: Option<String>,
    pub optimize: Option<String>,
    /// Escape hatch: explicit `gltfpack` args replacing the safe default
    /// set. Use to enable quantization when you don't rely on faction
    /// tint (quantizing the zone UV can corrupt it — see §3.4).
    pub optimize_args: Option<Vec<String>>,
}

/// One spec entry: either a single `src` → `out`, or a bulk `srcDir` →
/// `outDir` (expanded to one item per `.vxlt` in `srcDir`). Any [`Settings`]
/// field set here overrides the spec defaults for this entry.
#[derive(Debug, Clone, Deserialize)]
pub struct RawItem {
    pub src: Option<String>,
    pub out: Option<String>,
    #[serde(rename = "srcDir")]
    pub src_dir: Option<String>,
    #[serde(rename = "outDir")]
    pub out_dir: Option<String>,
    #[serde(flatten)]
    pub settings: Settings,
}

// ===========================================================================
// Resolved (validated, typed) items
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum Smoothing {
    None,
    Light,
    Heavy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Optimize {
    None,
    Meshopt,
}

/// A fully-resolved, validated bake job for a single output file.
#[derive(Debug, Clone)]
struct ResolvedItem {
    src: PathBuf,
    out: PathBuf,
    smoothing: Smoothing,
    transform: ExportTransform,
    optimize: Optimize,
    optimize_args: Option<Vec<String>>,
    pivot_label: String,
    up_label: String,
}

impl ResolvedItem {
    /// Report skeleton with the static metadata filled in (ok defaults to
    /// false; callers flip it and fill counts on success).
    fn base_report(&self) -> ItemReport {
        ItemReport {
            src: self.src.display().to_string(),
            out: self.out.display().to_string(),
            ok: false,
            error: None,
            format: "glTF Binary (.glb)".to_string(),
            mesh_source: mesh_source_label(self.smoothing).to_string(),
            pivot: self.pivot_label.clone(),
            up_axis: self.up_label.clone(),
            unit_scale: self.transform.unit_scale,
            triangles: 0,
            vertices: 0,
            chunks: 0,
            sockets: 0,
            color_model: "Per-vertex RGBA; AO baked into RGB; faction tint-zone \
                          in _TINTZONE + TEXCOORD_0.x"
                .to_string(),
            optimize: "none".to_string(),
            bytes_raw: 0,
            bytes_final: 0,
            notes: Vec::new(),
        }
    }

    fn failed(&self, msg: impl Into<String>) -> ItemReport {
        let mut r = self.base_report();
        r.error = Some(msg.into());
        r
    }
}

// ===========================================================================
// Reports
// ===========================================================================

/// Per-item outcome, written next to the output as `<out>.report.json`
/// (the headless analogue of the interactive post-export dialog) and
/// returned in the [`BakeOutcome`].
#[derive(Debug, Clone, Serialize)]
pub struct ItemReport {
    pub src: String,
    pub out: String,
    pub ok: bool,
    pub error: Option<String>,
    pub format: String,
    pub mesh_source: String,
    pub pivot: String,
    pub up_axis: String,
    pub unit_scale: f32,
    pub triangles: usize,
    pub vertices: usize,
    pub chunks: usize,
    pub sockets: usize,
    pub color_model: String,
    /// What happened in the optimize step: `none`, `meshopt (gltfpack)`,
    /// or a `skipped (...)` / `failed` note.
    pub optimize: String,
    /// File size right after export, before optimize.
    pub bytes_raw: u64,
    /// File size after optimize (== `bytes_raw` when not optimized).
    pub bytes_final: u64,
    pub notes: Vec<String>,
}

/// The result of a whole bake run.
#[derive(Debug)]
pub struct BakeOutcome {
    pub reports: Vec<ItemReport>,
}

impl BakeOutcome {
    pub fn ok_count(&self) -> usize {
        self.reports.iter().filter(|r| r.ok).count()
    }

    pub fn any_failed(&self) -> bool {
        self.reports.iter().any(|r| !r.ok)
    }

    /// Human-readable console summary (one line per item + notes). Lives
    /// here so the binary just prints it and stays presentation-free.
    pub fn summary_string(&self) -> String {
        use std::fmt::Write;
        let total = self.reports.len();
        let ok = self.ok_count();
        let failed = total - ok;
        let mut out = String::new();
        if failed == 0 {
            let _ = writeln!(out, "Baked {ok}/{total} item(s).");
        } else {
            let _ = writeln!(out, "Baked {ok}/{total} item(s) ({failed} failed).");
        }
        for r in &self.reports {
            if r.ok {
                let size = if r.bytes_final != r.bytes_raw {
                    format!(
                        "{} -> {}",
                        format_bytes(r.bytes_raw),
                        format_bytes(r.bytes_final)
                    )
                } else {
                    format_bytes(r.bytes_raw)
                };
                let _ = writeln!(
                    out,
                    "  ok    {}  ({} tris, {} verts, {})  [{}]",
                    r.out,
                    group_thousands(r.triangles),
                    group_thousands(r.vertices),
                    size,
                    r.optimize,
                );
                for n in &r.notes {
                    let _ = writeln!(out, "          - {n}");
                }
            } else {
                let _ = writeln!(
                    out,
                    "  FAIL  {}  {}",
                    r.out,
                    r.error.as_deref().unwrap_or("unknown error"),
                );
            }
        }
        out
    }
}

// ===========================================================================
// Public entry point
// ===========================================================================

/// Read and run a bake spec. `shard`, when `Some("i/n")`, processes only
/// every n-th item starting at i — for CI fan-out across processes.
///
/// Returns `Err` only for spec-level problems (the whole run can't start);
/// individual model failures are recorded in the returned reports with
/// `ok == false`, so the batch always completes what it can.
pub fn run_bake(spec_path: &Path, shard: Option<&str>) -> Result<BakeOutcome, BakeError> {
    let text = std::fs::read_to_string(spec_path).map_err(|e| {
        BakeError::Spec(format!("could not read spec {}: {e}", spec_path.display()))
    })?;
    let spec: BakeSpec = serde_json::from_str(&text)
        .map_err(|e| BakeError::Spec(format!("invalid spec {}: {e}", spec_path.display())))?;

    // Paths in the spec are relative to the spec file's directory.
    let base_dir = spec_path.parent().unwrap_or_else(|| Path::new("."));
    let mut items = expand_items(&spec, base_dir)?;

    if let Some(sh) = shard {
        let (i, n) = parse_shard(sh)?;
        items = items
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| *idx % n == i)
            .map(|(_, it)| it)
            .collect();
    }

    let reports = items.iter().map(bake_item).collect();
    Ok(BakeOutcome { reports })
}

// ===========================================================================
// Spec resolution
// ===========================================================================

fn expand_items(spec: &BakeSpec, base: &Path) -> Result<Vec<ResolvedItem>, BakeError> {
    let mut out = Vec::new();
    for (i, raw) in spec.items.iter().enumerate() {
        let merged = merge(&spec.defaults, &raw.settings);
        let parsed =
            parse_settings(&merged).map_err(|e| BakeError::Spec(format!("item {i}: {e}")))?;

        match (&raw.src, &raw.out, &raw.src_dir, &raw.out_dir) {
            (Some(src), Some(o), None, None) => {
                out.push(make_item(base.join(src), base.join(o), &parsed));
            }
            (None, None, Some(sd), Some(od)) => {
                let dir = base.join(sd);
                let entries = std::fs::read_dir(&dir).map_err(|e| {
                    BakeError::Spec(format!(
                        "item {i}: cannot read srcDir {}: {e}",
                        dir.display()
                    ))
                })?;
                let mut vxlt: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| {
                        p.is_file()
                            && p.extension()
                                .and_then(|x| x.to_str())
                                .map(|x| x.eq_ignore_ascii_case("vxlt"))
                                .unwrap_or(false)
                    })
                    .collect();
                vxlt.sort(); // deterministic order across runs / shards
                if vxlt.is_empty() {
                    return Err(BakeError::Spec(format!(
                        "item {i}: no .vxlt files in srcDir {}",
                        dir.display()
                    )));
                }
                let out_dir = base.join(od);
                for src in vxlt {
                    let stem = src
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("model");
                    let o = out_dir.join(format!("{stem}.glb"));
                    out.push(make_item(src, o, &parsed));
                }
            }
            _ => {
                return Err(BakeError::Spec(format!(
                    "item {i}: must have either both `src` and `out`, or both \
                     `srcDir` and `outDir`"
                )));
            }
        }
    }
    Ok(out)
}

/// Item settings win over defaults, field by field.
fn merge(defaults: &Settings, item: &Settings) -> Settings {
    Settings {
        mesher: item.mesher.clone().or_else(|| defaults.mesher.clone()),
        smoothing: item.smoothing.clone().or_else(|| defaults.smoothing.clone()),
        up_axis: item.up_axis.clone().or_else(|| defaults.up_axis.clone()),
        unit_scale: item.unit_scale.or(defaults.unit_scale),
        pivot: item.pivot.clone().or_else(|| defaults.pivot.clone()),
        optimize: item.optimize.clone().or_else(|| defaults.optimize.clone()),
        optimize_args: item
            .optimize_args
            .clone()
            .or_else(|| defaults.optimize_args.clone()),
    }
}

/// Parsed, validated settings (paths get added later by `make_item`).
struct ParsedSettings {
    smoothing: Smoothing,
    transform: ExportTransform,
    optimize: Optimize,
    optimize_args: Option<Vec<String>>,
    pivot_label: String,
    up_label: String,
}

fn parse_settings(s: &Settings) -> Result<ParsedSettings, String> {
    // Only the greedy mesher exists for export; Marching Cubes is selected
    // via `smoothing`, not `mesher`.
    let mesher = s.mesher.as_deref().unwrap_or("greedy");
    if mesher != "greedy" {
        return Err(format!(
            "unknown mesher '{mesher}' (only 'greedy' is supported; use \
             'smoothing' for Marching Cubes)"
        ));
    }

    let smoothing = match s.smoothing.as_deref().unwrap_or("none") {
        "none" => Smoothing::None,
        "light" => Smoothing::Light,
        "heavy" => Smoothing::Heavy,
        other => {
            return Err(format!(
                "unknown smoothing '{other}' (expected none|light|heavy)"
            ))
        }
    };

    let (pivot, pivot_label) = match s.pivot.as_deref().unwrap_or("origin") {
        "origin" => (Pivot::Origin, "origin"),
        "base-center" => (Pivot::BaseCenter, "base-center"),
        "feet" => (Pivot::BaseCenter, "feet"), // bottom-center, same point
        "center" => (Pivot::Center, "center"),
        other => {
            return Err(format!(
                "unknown pivot '{other}' (expected origin|base-center|feet|center)"
            ))
        }
    };

    let (up_axis, up_label) = match s.up_axis.as_deref().unwrap_or("y") {
        "y" | "Y" => (UpAxis::Y, "y"),
        "z" | "Z" => (UpAxis::Z, "z"),
        other => return Err(format!("unknown up_axis '{other}' (expected y|z)")),
    };

    let unit_scale = s.unit_scale.unwrap_or(1.0);
    if !(unit_scale.is_finite() && unit_scale > 0.0) {
        return Err(format!(
            "unit_scale must be a finite positive number, got {unit_scale}"
        ));
    }

    let optimize = match s.optimize.as_deref().unwrap_or("none") {
        "none" => Optimize::None,
        "meshopt" => Optimize::Meshopt,
        other => {
            return Err(format!(
                "unknown optimize '{other}' (expected none|meshopt)"
            ))
        }
    };

    Ok(ParsedSettings {
        smoothing,
        transform: ExportTransform {
            pivot,
            up_axis,
            unit_scale,
        },
        optimize,
        optimize_args: s.optimize_args.clone(),
        pivot_label: pivot_label.to_string(),
        up_label: up_label.to_string(),
    })
}

fn make_item(src: PathBuf, out: PathBuf, p: &ParsedSettings) -> ResolvedItem {
    ResolvedItem {
        src,
        out,
        smoothing: p.smoothing,
        transform: p.transform,
        optimize: p.optimize,
        optimize_args: p.optimize_args.clone(),
        pivot_label: p.pivot_label.clone(),
        up_label: p.up_label.clone(),
    }
}

fn parse_shard(s: &str) -> Result<(usize, usize), BakeError> {
    let bad = || BakeError::Spec(format!("invalid --shard '{s}' (expected i/n, e.g. 0/4)"));
    let (i, n) = s.split_once('/').ok_or_else(|| bad())?;
    let i: usize = i.trim().parse().map_err(|_| bad())?;
    let n: usize = n.trim().parse().map_err(|_| bad())?;
    if n == 0 || i >= n {
        return Err(BakeError::Spec(format!(
            "invalid --shard '{s}': need 0 <= i < n and n > 0"
        )));
    }
    Ok((i, n))
}

fn mesh_source_label(s: Smoothing) -> &'static str {
    match s {
        Smoothing::None => "Greedy mesh",
        Smoothing::Light => "Marching Cubes (light)",
        Smoothing::Heavy => "Marching Cubes (heavy)",
    }
}

// ===========================================================================
// Per-item bake
// ===========================================================================

fn bake_item(item: &ResolvedItem) -> ItemReport {
    let report = bake_item_inner(item);
    write_item_report(&item.out, &report);
    report
}

fn bake_item_inner(item: &ResolvedItem) -> ItemReport {
    let (world, state) = match io::load_world_with_state(&item.src) {
        Ok(v) => v,
        Err(e) => return item.failed(format!("load failed: {e}")),
    };

    // Sockets → glTF empty-node descriptors. The `+Y → normal` rotation
    // convention lives in `Socket::rotation` (same as interactive export).
    let sockets: Vec<SocketNode> = state
        .sockets
        .iter()
        .map(|sd| {
            let s = Socket::new(sd.name.clone(), sd.position, sd.normal);
            SocketNode {
                name: sd.name.clone(),
                translation: sd.position,
                rotation: s.rotation(),
            }
        })
        .collect();

    if let Some(parent) = item.out.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return item.failed(format!(
                "cannot create output dir {}: {e}",
                parent.display()
            ));
        }
    }

    let stats = match item.smoothing {
        Smoothing::None => {
            io::export_glb_with_transform(&world, &sockets, &item.out, item.transform)
        }
        Smoothing::Light => io::export_glb_smoothed_with_transform(
            &world,
            &sockets,
            &item.out,
            false,
            item.transform,
        ),
        Smoothing::Heavy => io::export_glb_smoothed_with_transform(
            &world,
            &sockets,
            &item.out,
            true,
            item.transform,
        ),
    };
    let stats = match stats {
        Ok(s) => s,
        Err(e) => return item.failed(format!("export failed: {e}")),
    };

    let mut report = item.base_report();
    report.ok = true;
    report.triangles = stats.triangle_count;
    report.vertices = stats.vertex_count;
    report.chunks = stats.chunk_count;
    report.sockets = sockets.len();
    report.bytes_raw = std::fs::metadata(&item.out)
        .map(|m| m.len())
        .unwrap_or(stats.byte_size as u64);
    report.bytes_final = report.bytes_raw;

    if stats.triangle_count == 0 {
        report
            .notes
            .push("no geometry — exported an empty / socket-only glb".to_string());
    }

    match item.optimize {
        Optimize::None => report.optimize = "none".to_string(),
        Optimize::Meshopt if stats.triangle_count == 0 => {
            report.optimize = "skipped (no geometry)".to_string();
        }
        Optimize::Meshopt => match run_gltfpack(&item.out, item.optimize_args.as_deref()) {
            Ok(()) => {
                report.bytes_final = std::fs::metadata(&item.out)
                    .map(|m| m.len())
                    .unwrap_or(report.bytes_raw);
                report.optimize = "meshopt (gltfpack)".to_string();
                report.notes.push(
                    "meshopt may drop the custom _TINTZONE attribute; the \
                     TEXCOORD_0.x zone mirror is preserved"
                        .to_string(),
                );
            }
            Err(OptimizeError::NotFound) => {
                report.optimize = "skipped (gltfpack not found)".to_string();
                report.notes.push(
                    "`gltfpack` not on PATH — kept the un-optimized glb. Install \
                     meshoptimizer's gltfpack to enable compression."
                        .to_string(),
                );
            }
            Err(e) => {
                report.optimize = "failed".to_string();
                report
                    .notes
                    .push(format!("optimize failed: {e} — kept the un-optimized glb"));
            }
        },
    }

    report
}

/// Write `<out>.report.json` next to the output (best-effort; a failure
/// here is logged, not fatal — the .glb is what matters).
fn write_item_report(out: &Path, report: &ItemReport) {
    let report_path = out.with_extension("report.json");
    if let Some(parent) = report_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(report) {
        Ok(s) => {
            if let Err(e) = std::fs::write(&report_path, s) {
                log::warn!("could not write report {}: {e}", report_path.display());
            }
        }
        Err(e) => log::warn!("could not serialize report for {}: {e}", out.display()),
    }
}

// ===========================================================================
// Optimize (gltfpack / meshopt, §3.4)
// ===========================================================================

#[derive(Debug)]
enum OptimizeError {
    /// `gltfpack` isn't installed / not on PATH.
    NotFound,
    /// `gltfpack` ran but exited non-zero.
    Failed(String),
    /// Spawning / file I/O around the call failed.
    Io(String),
}

impl std::fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizeError::NotFound => write!(f, "gltfpack not found"),
            OptimizeError::Failed(s) => write!(f, "gltfpack: {s}"),
            OptimizeError::Io(s) => write!(f, "io: {s}"),
        }
    }
}

/// Compress `glb` in place via meshoptimizer's `gltfpack`.
///
/// Default args are correctness-first for the Voxelith format (§3.4):
/// - `-cc` : `EXT_meshopt_compression` — the main win (vertex-cache
///   reorder + vertex/index buffer compression).
/// - `-noq`: no quantization — protects the **integer tint-zone** carried
///   in `TEXCOORD_0.x` (quantization can shift it and corrupt faction
///   recolor) and keeps `COLOR_0` AO exact. Override via `optimize_args`
///   to trade this for smaller files when you don't use faction tint.
/// - `-kn` : keep named nodes — preserves the named socket empty-nodes.
/// - `-km` : keep named materials — preserves the plain/emissive/metallic split.
///
/// Never decimates (no `-si`) — simplification would destroy the hard
/// voxel edges and per-vertex AO the export works to preserve.
fn run_gltfpack(glb: &Path, args_override: Option<&[String]>) -> Result<(), OptimizeError> {
    // gltfpack writes a fresh file; go via a temp then rename over the
    // input so a mid-run failure can't leave a half-written .glb.
    let tmp = glb.with_extension("opt.tmp.glb");
    let mut cmd = std::process::Command::new("gltfpack");
    cmd.arg("-i").arg(glb).arg("-o").arg(&tmp);
    match args_override {
        Some(args) => {
            cmd.args(args);
        }
        None => {
            cmd.args(["-cc", "-noq", "-kn", "-km"]);
        }
    }

    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(OptimizeError::NotFound)
        }
        Err(e) => return Err(OptimizeError::Io(e.to_string())),
    };

    if !output.status.success() {
        let _ = std::fs::remove_file(&tmp);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(3).collect();
        let msg = if tail.is_empty() {
            format!("exited with {}", output.status)
        } else {
            tail.into_iter().rev().collect::<Vec<_>>().join("; ")
        };
        return Err(OptimizeError::Failed(msg));
    }

    std::fs::rename(&tmp, glb).map_err(|e| OptimizeError::Io(e.to_string()))?;
    Ok(())
}

// ===========================================================================
// Formatting helpers
// ===========================================================================

fn format_bytes(n: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let f = n as f64;
    if f >= MIB {
        format!("{:.1} MiB", f / MIB)
    } else if f >= KIB {
        format!("{:.1} KiB", f / KIB)
    } else {
        format!("{n} B")
    }
}

fn group_thousands(n: usize) -> String {
    let s = n.to_string();
    let len = s.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_item_overrides_defaults() {
        let defaults = Settings {
            pivot: Some("origin".into()),
            optimize: Some("meshopt".into()),
            unit_scale: Some(1.0),
            ..Default::default()
        };
        let item = Settings {
            pivot: Some("feet".into()),
            ..Default::default()
        };
        let m = merge(&defaults, &item);
        assert_eq!(m.pivot.as_deref(), Some("feet")); // item wins
        assert_eq!(m.optimize.as_deref(), Some("meshopt")); // default kept
        assert_eq!(m.unit_scale, Some(1.0));
    }

    #[test]
    fn parse_settings_defaults_are_identity() {
        let p = parse_settings(&Settings::default()).unwrap();
        assert!(p.transform.is_identity());
        assert_eq!(p.smoothing, Smoothing::None);
        assert_eq!(p.optimize, Optimize::None);
    }

    #[test]
    fn parse_settings_maps_pivot_and_smoothing() {
        let p = parse_settings(&Settings {
            pivot: Some("base-center".into()),
            smoothing: Some("heavy".into()),
            up_axis: Some("z".into()),
            unit_scale: Some(0.5),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(p.transform.pivot, Pivot::BaseCenter);
        assert_eq!(p.transform.up_axis, UpAxis::Z);
        assert_eq!(p.transform.unit_scale, 0.5);
        assert_eq!(p.smoothing, Smoothing::Heavy);

        // "feet" is an alias for base-center but keeps its own label.
        let feet = parse_settings(&Settings {
            pivot: Some("feet".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(feet.transform.pivot, Pivot::BaseCenter);
        assert_eq!(feet.pivot_label, "feet");
    }

    #[test]
    fn parse_settings_rejects_bad_values() {
        assert!(parse_settings(&Settings {
            pivot: Some("middle".into()),
            ..Default::default()
        })
        .is_err());
        assert!(parse_settings(&Settings {
            smoothing: Some("ultra".into()),
            ..Default::default()
        })
        .is_err());
        assert!(parse_settings(&Settings {
            unit_scale: Some(0.0),
            ..Default::default()
        })
        .is_err());
        assert!(parse_settings(&Settings {
            mesher: Some("naive".into()),
            ..Default::default()
        })
        .is_err());
    }

    #[test]
    fn parse_shard_valid_and_invalid() {
        assert_eq!(parse_shard("0/4").unwrap(), (0, 4));
        assert_eq!(parse_shard("3/4").unwrap(), (3, 4));
        assert!(parse_shard("4/4").is_err()); // i must be < n
        assert!(parse_shard("0/0").is_err()); // n > 0
        assert!(parse_shard("abc").is_err());
    }

    #[test]
    fn spec_deserializes_defaults_and_items() {
        let json = r#"{
            "defaults": { "pivot": "base-center", "optimize": "meshopt" },
            "items": [
                { "src": "a.vxlt", "out": "a.glb" },
                { "src": "b.vxlt", "out": "b.glb", "pivot": "feet" },
                { "srcDir": "c/", "outDir": "c/" }
            ]
        }"#;
        let spec: BakeSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.defaults.pivot.as_deref(), Some("base-center"));
        assert_eq!(spec.items.len(), 3);
        assert_eq!(spec.items[1].settings.pivot.as_deref(), Some("feet"));
        assert_eq!(spec.items[2].src_dir.as_deref(), Some("c/"));
    }

    #[test]
    fn group_thousands_formats() {
        assert_eq!(group_thousands(0), "0");
        assert_eq!(group_thousands(42), "42");
        assert_eq!(group_thousands(1234), "1,234");
        assert_eq!(group_thousands(1234567), "1,234,567");
    }

    #[test]
    fn bake_single_item_produces_glb_and_report() {
        use crate::core::{Voxel, World};

        let dir = std::env::temp_dir().join("voxelith_bake_it");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("cube.vxlt");
        let out = dir.join("cube.glb");
        let spec_path = dir.join("spec.json");

        let mut world = World::new();
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    world.set_voxel(x, y, z, Voxel::from_rgb(200, 100, 50));
                }
            }
        }
        io::save_world_with_state(&world, io::EditorState::default(), &src).unwrap();

        // Absolute paths; `base.join(absolute)` keeps the absolute path.
        fn esc(p: &Path) -> String {
            p.display().to_string().replace('\\', "\\\\")
        }
        let spec = format!(
            r#"{{ "defaults": {{ "pivot": "base-center", "optimize": "none" }},
                  "items": [ {{ "src": "{}", "out": "{}" }} ] }}"#,
            esc(src.as_path()),
            esc(out.as_path()),
        );
        std::fs::write(&spec_path, spec).unwrap();

        let outcome = run_bake(&spec_path, None).unwrap();
        assert_eq!(outcome.reports.len(), 1);
        assert!(
            outcome.reports[0].ok,
            "bake failed: {:?}",
            outcome.reports[0].error
        );
        assert!(out.exists(), "glb not written");
        assert!(
            out.with_extension("report.json").exists(),
            "report not written"
        );
        assert!(outcome.reports[0].triangles > 0);
        assert!(!outcome.any_failed());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
