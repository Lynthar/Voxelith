//! The document: everything belonging to the *project* rather than the
//! session — world, sockets, pipeline graph, metadata — under one owner
//! and one revision counter.

use voxelith::core::World;
use voxelith::editor::Socket;
use voxelith::io::ProjectMetadata;
use voxelith::procgen::PipelineGraph;

/// The open project: world, sockets, pipeline graph and metadata.
/// Dirty state is a private [`DocumentMeta`], so "autosave must never
/// clear the unsaved flag" holds by construction.
pub(super) struct Document {
    pub world: World,
    pub sockets: Vec<Socket>,
    pub graph: PipelineGraph,
    /// Identity of the project. Held by the session and written out
    /// explicitly, so Save As onto an unrelated file doesn't inherit
    /// that file's name and author.
    pub metadata: ProjectMetadata,
    meta: DocumentMeta,
}

/// Three high-water marks over one monotonic revision counter;
/// `unsaved` and `autosave_due` are derived, never stored. Not the same
/// counter as `CommandHistory::generation()` — that one is undo-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DocumentMeta {
    /// Bumped on every document change.
    revision: u64,
    /// `revision` as of the last save / open / new / reload — the
    /// moments the document and the user's own file agree.
    saved: u64,
    /// `revision` as of the last successful autosave write.
    autosaved: u64,
}

impl Document {
    pub fn new() -> Self {
        Self {
            world: World::new(),
            sockets: Vec::new(),
            graph: PipelineGraph::default(),
            metadata: ProjectMetadata::default(),
            meta: DocumentMeta {
                revision: 0,
                saved: 0,
                autosaved: 0,
            },
        }
    }

    /// The document changed — a voxel write (via the mesh-rebuild
    /// chokepoint), a socket or graph edit, an undo/redo step, a
    /// recovery. The one producer verb.
    pub fn bump(&mut self) {
        self.meta.revision += 1;
    }

    /// The document and the user's own file now agree: save, open, new,
    /// import, reload. Retires the autosave debt too, since the file
    /// already holds this exact state.
    pub fn mark_saved(&mut self) {
        self.meta.saved = self.meta.revision;
        self.meta.autosaved = self.meta.revision;
    }

    /// A successful autosave write. Touches only the autosave mark, so
    /// `unsaved()` keeps answering true — a clean exit deletes the
    /// autosave, so "edit → autosave → close" must still prompt.
    pub fn mark_autosaved(&mut self) {
        self.meta.autosaved = self.meta.revision;
    }

    /// Recovered work has no file of its own, so the revision advances
    /// past every mark and `unsaved()` holds until a manual save. The
    /// autosave debt is retired — the autosave *is* this state.
    pub fn mark_recovered(&mut self) {
        self.meta.revision += 1;
        self.meta.autosaved = self.meta.revision;
    }

    /// Dirty relative to the file the user owns. Gates the
    /// unsaved-changes guard on every path that would throw the scene
    /// away, and the disk-conflict classification.
    pub fn unsaved(&self) -> bool {
        self.meta.revision != self.meta.saved
    }

    /// The autosave timer owes this document a write.
    pub fn autosave_due(&self) -> bool {
        self.meta.revision != self.meta.autosaved
    }
}

#[cfg(test)]
mod tests {
    //! The dirty-state invariants, written against the meta *before*
    //! the boolean pair was replaced — each test names the incident
    //! its invariant comes from.

    use super::Document;

    #[test]
    fn a_fresh_document_is_clean() {
        let doc = Document::new();
        assert!(!doc.unsaved());
        assert!(!doc.autosave_due());
    }

    #[test]
    fn edit_then_autosave_then_close_still_prompts() {
        // Autosave is a crash net a clean exit deletes. If its write
        // cleared `unsaved`, "edit → autosave → close" would skip the
        // prompt and destroy the only copy on the way out.
        let mut doc = Document::new();
        doc.bump();
        doc.mark_autosaved();
        assert!(doc.unsaved(), "autosave must never clear unsaved");
        assert!(!doc.autosave_due(), "but it does retire the timer's debt");
    }

    #[test]
    fn a_save_settles_both_marks() {
        let mut doc = Document::new();
        doc.bump();
        doc.bump();
        doc.mark_saved();
        assert!(!doc.unsaved());
        assert!(!doc.autosave_due(), "the file holds this state; no debt");
    }

    #[test]
    fn a_reload_is_a_save_shaped_event() {
        // Reload replaces the scene with the file's contents — the
        // mesh rebuild it triggers bumps like any edit, and the caller
        // then marks saved because the world came OUT of the file.
        let mut doc = Document::new();
        doc.bump(); // the rebuild after loading
        doc.mark_saved();
        assert!(!doc.unsaved());
    }

    #[test]
    fn recovered_work_is_unsaved_until_a_manual_save() {
        // The recovery copy isn't the user's real file: project_path
        // stays None and the next clean exit deletes the autosave, so
        // the guard must hold from the moment of recovery.
        let mut doc = Document::new();
        doc.mark_recovered();
        assert!(doc.unsaved());
        assert!(!doc.autosave_due(), "the autosave already holds this state");
        // …and a later save settles it like any other document.
        doc.mark_saved();
        assert!(!doc.unsaved());
    }

    #[test]
    fn undo_back_to_the_save_point_still_reads_unsaved() {
        // Revisions are monotonic, so undoing back to a
        // visually-identical state does not clear the flag. A content
        // hash could; this is the deliberate limitation.
        let mut doc = Document::new();
        doc.mark_saved();
        doc.bump(); // an edit
        doc.bump(); // its undo — steps bump like edits do
        assert!(doc.unsaved());
    }

    #[test]
    fn autosave_failure_keeps_the_debt() {
        // A failed write calls nothing; the next interval retries
        // because the due-check still answers true.
        let mut doc = Document::new();
        doc.bump();
        assert!(doc.autosave_due());
        // (no mark) — still due:
        assert!(doc.autosave_due());
        doc.mark_autosaved();
        assert!(!doc.autosave_due());
        // A new edit re-arms the debt.
        doc.bump();
        assert!(doc.autosave_due());
    }
}
