//! The editor's saved workspace — window geometry, panel visibility,
//! viewport and procgen settings, brush state, recent files — as `ron`
//! in the platform config dir. Every nested struct defaults.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ui::{ProcgenSettings, ViewportSettings};

/// Maximum entries kept in the recent-files MRU.
pub const MAX_RECENT_FILES: usize = 10;

/// True if `path` names a Voxelith project (`.vxlt`, case-insensitive
/// so a `.VXLT` typed into a save dialog still counts).
fn is_project_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("vxlt"))
}

/// Top-level preferences container.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    pub window: WindowPrefs,
    pub panels: PanelVisibility,
    pub viewport: ViewportSettings,
    pub procgen: ProcgenSettings,
    pub editor: EditorPrefs,
    pub recent_files: Vec<PathBuf>,
    /// Directory of the last successful export, seeding the next export
    /// dialog. Exports deliberately stay out of `recent_files`, so this
    /// is where that information lives instead.
    pub last_export_dir: Option<PathBuf>,
    /// Directory of the last successful `.vox` import, same rationale.
    pub last_import_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowPrefs {
    pub width: u32,
    pub height: u32,
}

impl Default for WindowPrefs {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
        }
    }
}

/// Which workspace panels are open. `UiState` holds one of these
/// directly, so load and save are whole-struct assignments and a new
/// panel can't be persisted at one end only. Transient windows stay off.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PanelVisibility {
    pub show_stats: bool,
    /// The context Inspector (the Tools float's successor; renamed
    /// from `show_tools`, so an older prefs file's value for that key
    /// is ignored and this starts from its default once).
    pub show_inspector: bool,
    pub show_palette: bool,
    pub show_viewport_settings: bool,
    pub show_graph: bool,
    pub show_agent: bool,
}

impl Default for PanelVisibility {
    /// The lean default workspace: toolbar, status bar, palette and the
    /// Inspector; everything else is opt-in via the View menu. Only a
    /// fresh install sees these — an existing file's own set wins.
    fn default() -> Self {
        Self {
            show_stats: false,
            show_inspector: true,
            show_palette: true,
            show_viewport_settings: false,
            show_graph: false,
            show_agent: false,
        }
    }
}

/// Editor brush state worth restoring across sessions. `selected_tool`
/// uses the same numeric encoding as `io::EditorState`, with
/// `app::tool_from_index` as the authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EditorPrefs {
    pub brush_color: [u8; 4],
    pub brush_size: u8,
    pub selected_tool: u8,
    /// Custom palette. Empty means "use Editor's built-in defaults".
    pub palette: Vec<[u8; 4]>,
    /// Symmetry axes (`[x, y, z]`). Stored as a plain array rather than
    /// a struct so the on-disk shape stays trivial.
    pub symmetry: [bool; 3],
    /// Brush material flags (`Voxel::flags`: bit0 emissive / bit1
    /// metallic) so the emissive / metallic toggles survive a restart.
    pub brush_flags: u8,
    /// Brush tint zone (`Voxel::tint_zone`: 0 none / 1 primary /
    /// 2 secondary / 3 reserved) so the zone picker survives a restart.
    pub brush_tint_zone: u8,
}

impl Default for EditorPrefs {
    fn default() -> Self {
        Self {
            brush_color: [200, 100, 50, 255],
            brush_size: 1,
            selected_tool: 0,
            palette: Vec::new(),
            symmetry: [false; 3],
            brush_flags: 0,
            brush_tint_zone: 0,
        }
    }
}

impl Prefs {
    /// Path of the prefs file on this platform, or `None` if the OS
    /// doesn't expose a config dir (extremely rare; non-fatal).
    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("voxelith").join("prefs.ron"))
    }

    /// Load prefs from the standard location. Any failure (missing
    /// file, parse error, missing config dir) returns `Default`.
    pub fn load() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        let data = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Self::default();
            }
            Err(e) => {
                log::warn!("Failed to read prefs from {}: {}", path.display(), e);
                return Self::default();
            }
        };
        match ron::from_str::<Prefs>(&data) {
            Ok(p) => p,
            Err(e) => {
                log::warn!(
                    "Failed to parse prefs at {}: {}; using defaults",
                    path.display(),
                    e
                );
                // Set the broken file aside rather than let it be
                // overwritten on exit: a `.corrupt` file the user can
                // inspect beats silently destroying the evidence.
                let quarantine = path.with_extension("ron.corrupt");
                if let Err(rename_err) = std::fs::rename(&path, &quarantine) {
                    log::warn!(
                        "Could not set the corrupt prefs aside at {}: {}",
                        quarantine.display(),
                        rename_err
                    );
                } else {
                    log::warn!("Corrupt prefs kept at {}", quarantine.display());
                }
                Self::default()
            }
        }
    }

    /// Persist prefs to the standard location, creating the parent
    /// directory if needed.
    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = Self::config_path() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no config directory available on this platform",
            ));
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(std::io::Error::other)?;
        std::fs::write(path, data)
    }

    /// Insert `path` at the head of `recent_files`, deduped and capped.
    /// Only `.vxlt` projects: the Open Recent menu hands every entry to
    /// the project loader, so anything else can only fail there.
    pub fn touch_recent(&mut self, path: &Path) {
        if !is_project_file(path) {
            return;
        }
        let path = path.to_path_buf();
        self.recent_files.retain(|p| p != &path);
        self.recent_files.insert(0, path);
        self.recent_files.truncate(MAX_RECENT_FILES);
    }

    /// Remember where the user last wrote an exported asset.
    pub fn remember_export_dir(&mut self, path: &Path) {
        if let Some(parent) = path.parent() {
            self.last_export_dir = Some(parent.to_path_buf());
        }
    }

    /// Remember where the user last read a `.vox` from.
    pub fn remember_import_dir(&mut self, path: &Path) {
        if let Some(parent) = path.parent() {
            self.last_import_dir = Some(parent.to_path_buf());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A graph written by a build that kept graphs in prefs must survive
    /// the move into project files — the user's work, not a cache.
    #[test]
    fn a_prefs_file_with_the_retired_graph_field_still_reads() {
        // A prefs file that still carries the old `graph` /
        // `graph_migrated` fields must have them ignored, not fail the
        // whole parse.
        let ron = r#"(
            graph: (
                nodes: [
                    (id: 0, kind: Terrain((seed: 7, width: 16, depth: 16,
                        min_height: 0, max_height: 4, frequency: 0.03, octaves: 3)),
                        position: (60.0, 40.0)),
                ],
                next_id: 1,
                output_node: None,
            ),
            graph_migrated: true,
            recent_files: ["/tmp/a.vxlt"],
        )"#;
        let prefs: Prefs = ron::from_str(ron).expect("an older prefs file must still parse");
        assert_eq!(prefs.recent_files.len(), 1, "the rest of the file survives");
    }

    #[test]
    fn test_default_roundtrip() {
        let p = Prefs::default();
        let s = ron::ser::to_string_pretty(&p, ron::ser::PrettyConfig::default()).unwrap();
        let back: Prefs = ron::from_str(&s).unwrap();
        assert_eq!(back.window.width, p.window.width);
        assert_eq!(back.panels.show_stats, p.panels.show_stats);
        assert_eq!(back.recent_files, p.recent_files);
    }

    #[test]
    fn test_partial_ron_falls_back_to_defaults() {
        // Only provide window — every other section must default.
        let s = "( window: ( width: 1024, height: 768 ) )";
        let p: Prefs = ron::from_str(s).unwrap();
        assert_eq!(p.window.width, 1024);
        assert_eq!(p.window.height, 768);
        // The lean default workspace: palette on, the rest opt-in.
        assert!(p.panels.show_palette);
        assert!(!p.panels.show_stats);
        assert!(p.recent_files.is_empty());
    }

    #[test]
    fn test_unknown_field_is_tolerated() {
        // serde with default attribute ignores extra fields by
        // default; this just confirms forward compatibility.
        let s = "( window: ( width: 800, height: 600 ), panels: ( show_stats: true, from_the_future: 42 ) )";
        let p: Prefs = ron::from_str(s).unwrap();
        assert!(p.panels.show_stats);
        // A field the file omits takes the struct default — assert on
        // one whose default is `true`, so this can't pass by zero-init.
        assert!(p.panels.show_palette);
    }

    #[test]
    fn test_touch_recent_dedup_and_cap() {
        let mut p = Prefs::default();
        for i in 0..15 {
            p.touch_recent(Path::new(&format!("/tmp/file{}.vxlt", i)));
        }
        assert_eq!(p.recent_files.len(), MAX_RECENT_FILES);
        // Most recent is at the head.
        assert_eq!(p.recent_files[0], PathBuf::from("/tmp/file14.vxlt"));

        // Re-touching an existing path moves it to the front, doesn't
        // duplicate.
        p.touch_recent(Path::new("/tmp/file10.vxlt"));
        assert_eq!(p.recent_files.len(), MAX_RECENT_FILES);
        assert_eq!(p.recent_files[0], PathBuf::from("/tmp/file10.vxlt"));
    }

    #[test]
    fn test_touch_recent_only_accepts_projects() {
        // Open Recent feeds every entry to the project loader, so
        // anything that isn't a .vxlt would be an entry that can only
        // error out — and would evict a real project on the way in.
        let mut p = Prefs::default();
        p.touch_recent(Path::new("/tmp/keeper.vxlt"));
        for junk in [
            "/tmp/model.glb",
            "/tmp/model.obj",
            "/tmp/model.vox",
            "/tmp/no-extension",
        ] {
            p.touch_recent(Path::new(junk));
        }
        assert_eq!(p.recent_files, vec![PathBuf::from("/tmp/keeper.vxlt")]);

        // Extension matching is case-insensitive.
        p.touch_recent(Path::new("/tmp/Shouty.VXLT"));
        assert_eq!(p.recent_files[0], PathBuf::from("/tmp/Shouty.VXLT"));
    }

    #[test]
    fn test_remember_dirs_store_parent() {
        let mut p = Prefs::default();
        p.remember_export_dir(Path::new("/tmp/assets/model.glb"));
        p.remember_import_dir(Path::new("/tmp/source/model.vox"));
        assert_eq!(p.last_export_dir, Some(PathBuf::from("/tmp/assets")));
        assert_eq!(p.last_import_dir, Some(PathBuf::from("/tmp/source")));
    }
}
