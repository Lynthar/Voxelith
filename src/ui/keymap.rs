//! The single source for tool and chord descriptors: each row feeds the
//! toolbar, the Inspector, the help window and the dispatch. Stateful
//! bindings stay out — a row can't say "unless a gesture is in flight".

use winit::keyboard::KeyCode;

use crate::editor::Tool;

use super::panels::UiAction;

/// One toolbar, Inspector, help and dispatch entry for a tool. The name
/// comes from [`Tool`] and the icon from `icons`; everything a surface
/// prints or a key press fires is in this row.
pub struct ToolSpec {
    pub tool: Tool,
    /// The bare key that selects the tool; `None` for toolbar-only
    /// tools. A test pins `shortcut` to this key's digit.
    pub key: Option<KeyCode>,
    /// The shortcut as the toolbar tooltip, Inspector and help window
    /// print it. Empty when there is no key.
    pub shortcut: &'static str,
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
pub static TOOL_SPECS: &[ToolSpec] = &[
    ToolSpec { tool: Tool::Place, key: Some(KeyCode::Digit1), shortcut: "1", note: "", separator_before: false },
    ToolSpec { tool: Tool::Remove, key: Some(KeyCode::Digit2), shortcut: "2", note: "", separator_before: false },
    ToolSpec { tool: Tool::Paint, key: Some(KeyCode::Digit3), shortcut: "3", note: "", separator_before: false },
    ToolSpec {
        tool: Tool::Eyedropper,
        key: Some(KeyCode::Digit4),
        shortcut: "4 / Alt",
        note: "Click a voxel to pick its color and material into the brush.",
        separator_before: false,
    },
    ToolSpec {
        tool: Tool::Fill,
        key: Some(KeyCode::Digit5),
        shortcut: "5",
        note: "Click a solid voxel to recolor its contiguous same-color region.",
        separator_before: false,
    },
    ToolSpec {
        tool: Tool::Line,
        key: Some(KeyCode::Digit6),
        shortcut: "6",
        note: "Drag from anchor to end (3D Bresenham line).",
        separator_before: true,
    },
    ToolSpec {
        tool: Tool::Box,
        key: Some(KeyCode::Digit7),
        shortcut: "7",
        note: "Drag corner to corner (filled AABB).",
        separator_before: false,
    },
    ToolSpec {
        tool: Tool::Sphere,
        key: Some(KeyCode::Digit8),
        shortcut: "8",
        note: "Drag a bounding box; the ellipsoid fits inside it.",
        separator_before: false,
    },
    ToolSpec {
        tool: Tool::Cylinder,
        key: Some(KeyCode::Digit9),
        shortcut: "9",
        note: "Drag a footprint, then pull up — the cylinder stands \
               along the locked face's normal.",
        separator_before: false,
    },
    ToolSpec {
        tool: Tool::Select,
        key: Some(KeyCode::Digit0),
        shortcut: "0",
        note: "Drag to mark an AABB. Esc or Ctrl+D deselects.",
        separator_before: true,
    },
    // No digit free; picked from the toolbar.
    ToolSpec {
        tool: Tool::Socket,
        key: None,
        shortcut: "",
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

/// The tool a bare `key` selects, if any — the digit-row dispatch.
pub fn tool_for_key(key: KeyCode) -> Option<Tool> {
    TOOL_SPECS
        .iter()
        .find(|s| s.key == Some(key))
        .map(|s| s.tool)
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
pub static CHORDS: &[ChordSpec] = &[
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
        for chord in CHORDS {
            for shift in [false, true] {
                if chord.shift.is_none_or(|s| s == shift) {
                    let found = find_chord(chord.key, shift).unwrap();
                    assert!(
                        std::ptr::eq(found, chord),
                        "chord {} is shadowed by {} and can never fire",
                        chord.chord_label,
                        found.chord_label
                    );
                }
            }
        }
    }

    #[test]
    fn every_tool_has_exactly_one_spec() {
        for tool in Tool::ALL {
            let count = TOOL_SPECS.iter().filter(|s| s.tool == tool).count();
            assert_eq!(count, 1, "{tool:?} must appear exactly once");
        }
        assert_eq!(TOOL_SPECS.len(), Tool::ALL.len());
    }

    #[test]
    fn every_key_selects_the_tool_its_label_promises() {
        // The label is what the toolbar and help window print, the key
        // is what fires; a digit bound twice would print for both rows
        // and select only the first.
        for spec in TOOL_SPECS {
            let Some(key) = spec.key else {
                assert!(
                    spec.shortcut.is_empty(),
                    "{:?} labels a key it lacks",
                    spec.tool
                );
                continue;
            };
            assert_eq!(tool_for_key(key), Some(spec.tool), "{key:?} is bound twice");
            let name = format!("{key:?}");
            let digit = name
                .strip_prefix("Digit")
                .expect("tool keys are the digit row");
            assert!(
                spec.shortcut.starts_with(digit),
                "{:?}: label {:?} does not name {key:?}",
                spec.tool,
                spec.shortcut
            );
        }
    }
}
