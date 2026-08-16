//! The single source for tool and chord descriptors: each row feeds the
//! toolbar, the Inspector, the help window and the dispatch. Stateful
//! bindings stay out — a row can't say "unless a gesture is in flight".

use winit::keyboard::KeyCode;

use crate::editor::Tool;

use super::panels::UiAction;

/// One toolbar, Inspector and help entry for a tool. Name and shortcut
/// come from [`Tool`] and the icon from `icons`, so this row adds only
/// the usage note and the toolbar grouping.
pub struct ToolSpec {
    pub tool: Tool,
    /// How the tool is used: the toolbar shows it on hover and the
    /// Inspector as its hint line, one string for both. Empty for tools
    /// whose name says it all.
    pub note: &'static str,
    /// Draw a group separator above this button (brush / shape /
    /// select / socket sections).
    pub separator_before: bool,
}

/// Every tool, in toolbar order.
#[rustfmt::skip] // one row per tool — the table reads as a table
pub const TOOL_SPECS: &[ToolSpec] = &[
    ToolSpec { tool: Tool::Place, note: "", separator_before: false },
    ToolSpec { tool: Tool::Remove, note: "", separator_before: false },
    ToolSpec { tool: Tool::Paint, note: "", separator_before: false },
    ToolSpec {
        tool: Tool::Eyedropper,
        note: "Click a voxel to pick its color and material into the brush.",
        separator_before: false,
    },
    ToolSpec {
        tool: Tool::Fill,
        note: "Click a solid voxel to recolor its contiguous same-color region.",
        separator_before: false,
    },
    ToolSpec {
        tool: Tool::Line,
        note: "Drag from anchor to end (3D Bresenham line).",
        separator_before: true,
    },
    ToolSpec {
        tool: Tool::Box,
        note: "Drag corner to corner (filled AABB).",
        separator_before: false,
    },
    ToolSpec {
        tool: Tool::Sphere,
        note: "Drag a bounding box; the ellipsoid fits inside it.",
        separator_before: false,
    },
    ToolSpec {
        tool: Tool::Cylinder,
        note: "Drag a footprint, then pull up — the cylinder stands \
               along the locked face's normal.",
        separator_before: false,
    },
    ToolSpec {
        tool: Tool::Select,
        note: "Drag to mark an AABB. Esc or Ctrl+D deselects.",
        separator_before: true,
    },
    ToolSpec {
        tool: Tool::Socket,
        note: "Click a voxel face (or the ground) to drop a named \
               attachment point. Exports to glTF as an empty node.",
        separator_before: true,
    },
];

/// The descriptor row for `tool` — the Inspector's way in. The
/// toolbar and help window iterate instead.
pub fn spec_of(tool: Tool) -> &'static ToolSpec {
    TOOL_SPECS
        .iter()
        .find(|s| s.tool == tool)
        .expect("every tool has a spec row; a test pins this")
}

/// Which help-window section a chord row renders under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpSection {
    Edit,
    Selection,
    File,
}

/// One command chord: the platform modifier plus `key`, dispatched
/// through the same `UiAction` queue the menus use. Labels say "Ctrl" —
/// the help window's macOS banner translates it once for every row.
pub struct ChordSpec {
    pub key: KeyCode,
    /// `Some(true)`: fires only with Shift held; `Some(false)`: only
    /// without; `None`: either way.
    pub shift: Option<bool>,
    pub make: fn() -> UiAction,
    pub section: HelpSection,
    pub chord_label: &'static str,
    pub help: &'static str,
}

/// Every pure "primary chord → action" binding, in help-window order.
pub const CHORDS: &[ChordSpec] = &[
    // -- Edit --
    ChordSpec {
        key: KeyCode::KeyZ,
        shift: Some(false),
        make: || UiAction::Undo,
        section: HelpSection::Edit,
        chord_label: "Ctrl+Z",
        help: "Undo",
    },
    ChordSpec {
        key: KeyCode::KeyZ,
        shift: Some(true),
        make: || UiAction::Redo,
        section: HelpSection::Edit,
        chord_label: "Ctrl+Shift+Z",
        help: "Redo",
    },
    ChordSpec {
        key: KeyCode::KeyY,
        shift: None,
        make: || UiAction::Redo,
        section: HelpSection::Edit,
        chord_label: "Ctrl+Y",
        help: "Redo",
    },
    // -- Selection --
    ChordSpec {
        key: KeyCode::KeyC,
        shift: None,
        make: || UiAction::CopySelection,
        section: HelpSection::Selection,
        chord_label: "Ctrl+C",
        help: "Copy non-air voxels",
    },
    ChordSpec {
        key: KeyCode::KeyX,
        shift: None,
        make: || UiAction::CutSelection,
        section: HelpSection::Selection,
        chord_label: "Ctrl+X",
        help: "Cut non-air voxels",
    },
    ChordSpec {
        key: KeyCode::KeyV,
        shift: Some(false),
        make: || UiAction::PasteClipboard { at_cursor: false },
        section: HelpSection::Selection,
        chord_label: "Ctrl+V",
        help: "Paste at selection origin (or cursor)",
    },
    ChordSpec {
        key: KeyCode::KeyV,
        shift: Some(true),
        make: || UiAction::PasteClipboard { at_cursor: true },
        section: HelpSection::Selection,
        chord_label: "Ctrl+Shift+V",
        help: "Paste at cursor cell",
    },
    ChordSpec {
        key: KeyCode::KeyA,
        shift: None,
        make: || UiAction::SelectAllSolid,
        section: HelpSection::Selection,
        chord_label: "Ctrl+A",
        help: "Select all (AABB of all solid voxels)",
    },
    ChordSpec {
        key: KeyCode::KeyD,
        shift: None,
        make: || UiAction::Deselect,
        section: HelpSection::Selection,
        chord_label: "Ctrl+D",
        help: "Deselect (Esc does too, outside a gesture)",
    },
    // -- File --
    ChordSpec {
        key: KeyCode::KeyN,
        shift: None,
        make: || UiAction::NewProject,
        section: HelpSection::File,
        chord_label: "Ctrl+N",
        help: "New project",
    },
    ChordSpec {
        key: KeyCode::KeyO,
        shift: None,
        make: || UiAction::OpenProject,
        section: HelpSection::File,
        chord_label: "Ctrl+O",
        help: "Open project",
    },
    ChordSpec {
        key: KeyCode::KeyS,
        shift: Some(false),
        make: || UiAction::SaveProject,
        section: HelpSection::File,
        chord_label: "Ctrl+S",
        help: "Save project",
    },
    ChordSpec {
        key: KeyCode::KeyS,
        shift: Some(true),
        make: || UiAction::SaveAs,
        section: HelpSection::File,
        chord_label: "Ctrl+Shift+S",
        help: "Save as…",
    },
];

/// The chord bound to `key` with the given shift state, if any.
pub fn find_chord(key: KeyCode, shift: bool) -> Option<&'static ChordSpec> {
    CHORDS
        .iter()
        .find(|c| c.key == key && c.shift.is_none_or(|s| s == shift))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_chord_is_reachable() {
        // A row shadowed by an earlier row (same key, overlapping
        // shift requirement) would render in the help window yet never
        // fire — the exact drift this table exists to prevent.
        for (i, chord) in CHORDS.iter().enumerate() {
            for shift in [false, true] {
                if chord.shift.is_none_or(|s| s == shift) {
                    let found = find_chord(chord.key, shift).unwrap();
                    let is_self = std::ptr::eq(found, chord);
                    let earlier = CHORDS[..i]
                        .iter()
                        .any(|c| c.key == chord.key && c.shift.is_none_or(|s| s == shift));
                    assert!(
                        is_self || earlier,
                        "chord {} is shadowed and can never fire",
                        chord.chord_label
                    );
                }
            }
        }
    }

    #[test]
    fn every_tool_has_exactly_one_spec() {
        use crate::editor::Tool::*;
        let all = [
            Place, Remove, Paint, Eyedropper, Fill, Line, Box, Sphere, Cylinder, Select, Socket,
        ];
        for tool in all {
            let count = TOOL_SPECS.iter().filter(|s| s.tool == tool).count();
            assert_eq!(count, 1, "{tool:?} must appear exactly once");
        }
        assert_eq!(TOOL_SPECS.len(), all.len());
    }
}
