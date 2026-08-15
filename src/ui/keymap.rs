//! The single source for tool and chord descriptors.
//!
//! Four consumers used to carry their own copy of this information —
//! the toolbar's buttons, the Tools panel's labels, the help window's
//! table, and the keyboard dispatch — and they drifted: an audit found
//! four live mismatches between what a surface said and what the code
//! did. Each row here feeds all of its consumers, so a new tool or
//! chord is one entry, and the help window can no longer promise a
//! binding the dispatcher doesn't have.
//!
//! Deliberately NOT in these tables: the number keys (they mirror
//! `Tool`'s declaration order and are printed from `Tool::shortcut`),
//! and the stateful bindings — R / M / F / arrows / Esc / Delete —
//! whose behavior depends on what is selected or in flight. A table
//! row can say "key → action"; it can't say "unless a shape gesture is
//! mid-Height", and pretending otherwise would move the truth back out
//! of the table. Camera bindings live in `CameraController`. The
//! README's key list remains hand-maintained prose.

use winit::keyboard::KeyCode;

use crate::editor::Tool;

use super::panels::UiAction;

/// One toolbar / panel / help entry for a tool. Name and shortcut come
/// from [`Tool`] itself — this row adds only what the type doesn't
/// carry: the button glyph and the long-form hover note.
pub struct ToolSpec {
    pub tool: Tool,
    pub icon: &'static str,
    /// Extra hover detail; empty for tools whose name says it all.
    pub note: &'static str,
    /// Draw a group separator above this button (brush / shape /
    /// select / socket sections).
    pub separator_before: bool,
}

/// Every tool, in toolbar order.
#[rustfmt::skip] // one row per tool — the table reads as a table
pub const TOOL_SPECS: &[ToolSpec] = &[
    ToolSpec { tool: Tool::Place, icon: "+", note: "", separator_before: false },
    ToolSpec { tool: Tool::Remove, icon: "-", note: "", separator_before: false },
    ToolSpec { tool: Tool::Paint, icon: "P", note: "", separator_before: false },
    ToolSpec { tool: Tool::Eyedropper, icon: "E", note: "", separator_before: false },
    ToolSpec { tool: Tool::Fill, icon: "F", note: "", separator_before: false },
    ToolSpec { tool: Tool::Line, icon: "L", note: "", separator_before: true },
    ToolSpec { tool: Tool::Box, icon: "▢", note: "", separator_before: false },
    ToolSpec { tool: Tool::Sphere, icon: "○", note: "", separator_before: false },
    ToolSpec { tool: Tool::Cylinder, icon: "⌭", note: "", separator_before: false },
    ToolSpec {
        tool: Tool::Select,
        icon: "▭",
        note: "Drag to mark an AABB. Esc or Ctrl+D deselects.",
        separator_before: true,
    },
    ToolSpec {
        tool: Tool::Socket,
        icon: "⚓",
        note: "Click a voxel face (or the ground) to drop a named \
               attachment point. Exports to glTF as an empty node.",
        separator_before: true,
    },
];

/// Which help-window section a chord row renders under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpSection {
    Edit,
    Selection,
    File,
}

/// One command chord: platform primary modifier + `key` (+ shift per
/// `shift`), dispatching `make()` through the same `UiAction` queue
/// the menus use. `chord_label` is written with "Ctrl" — the help
/// window's macOS banner says "use ⌘ wherever Ctrl is shown" once,
/// instead of every row saying it twice.
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
