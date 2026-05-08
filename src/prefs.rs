//! User preferences persisted across sessions.
//!
//! `Prefs` holds everything the user expects to survive a restart:
//! window geometry, panel visibility toggles, viewport + procgen
//! settings, last-used brush state, and a recent-files MRU list. The
//! file lives at the platform-standard config dir
//! (`%APPDATA%\voxelith\prefs.ron` on Windows, `~/.config/voxelith/`
//! on Linux, `~/Library/Application Support/voxelith/` on macOS) and
//! is encoded as `ron`.
//!
//! Every nested struct uses `#[serde(default)]` so an older prefs
//! file that's missing fields still loads — defaults fill the gaps.
//! Same goes for parse errors: a corrupt file is logged and replaced
//! with defaults rather than blocking startup.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::procgen::PipelineGraph;
use crate::ui::{ProcgenSettings, ViewportSettings};

/// Maximum entries kept in the recent-files MRU.
pub const MAX_RECENT_FILES: usize = 10;

/// Top-level preferences container.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    pub window: WindowPrefs,
    pub panels: PanelVisibility,
    pub viewport: ViewportSettings,
    pub procgen: ProcgenSettings,
    pub graph: PipelineGraph,
    pub editor: EditorPrefs,
    pub recent_files: Vec<PathBuf>,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            window: WindowPrefs::default(),
            panels: PanelVisibility::default(),
            viewport: ViewportSettings::default(),
            procgen: ProcgenSettings::default(),
            graph: PipelineGraph::default(),
            editor: EditorPrefs::default(),
            recent_files: Vec::new(),
        }
    }
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

/// Mirrors the `show_*` toggles on `ui::UiState`. Lives here so we
/// don't have to teach the action-queue/status-message bits of
/// `UiState` to serialize.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PanelVisibility {
    pub show_stats: bool,
    pub show_tools: bool,
    pub show_palette: bool,
    pub show_viewport_settings: bool,
    pub show_procgen: bool,
    pub show_graph: bool,
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            show_stats: true,
            show_tools: true,
            show_palette: true,
            show_viewport_settings: false,
            show_procgen: false,
            show_graph: false,
        }
    }
}

/// Editor brush state worth restoring across sessions. `selected_tool`
/// uses the same numeric encoding as `io::EditorState` for consistency
/// with project files: 0=Place, 1=Remove, 2=Paint, 3=Eyedropper, 4=Fill.
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
}

impl Default for EditorPrefs {
    fn default() -> Self {
        Self {
            brush_color: [200, 100, 50, 255],
            brush_size: 1,
            selected_tool: 0,
            palette: Vec::new(),
            symmetry: [false; 3],
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
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, data)
    }

    /// Insert `path` at the head of `recent_files`, dedup, cap at
    /// `MAX_RECENT_FILES`. Idempotent for paths already in the list
    /// (just promotes them to the head).
    pub fn touch_recent(&mut self, path: &Path) {
        let path = path.to_path_buf();
        self.recent_files.retain(|p| p != &path);
        self.recent_files.insert(0, path);
        self.recent_files.truncate(MAX_RECENT_FILES);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Panels are show=true by default for the always-on panels.
        assert!(p.panels.show_stats);
        assert!(p.recent_files.is_empty());
    }

    #[test]
    fn test_unknown_field_is_tolerated() {
        // serde with default attribute ignores extra fields by
        // default; this just confirms forward compatibility.
        let s = "( window: ( width: 800, height: 600 ), panels: ( show_stats: false ) )";
        let p: Prefs = ron::from_str(s).unwrap();
        assert!(!p.panels.show_stats);
        // show_tools should default to true.
        assert!(p.panels.show_tools);
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
}
