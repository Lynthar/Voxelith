//! The document: everything that belongs to the *project* rather than
//! to the session, in one place.
//!
//! Before this, the document's pieces lived on three hosts — the world
//! on `App`, the sockets on `Editor`, the pipeline graph on `Ui` (the
//! heaviest ownership misplacement of the three) — and "the document
//! changed" was two booleans maintained by hand at every producer.
//! R3 caught graph edits not raising them, R4 caught sockets, Clear
//! All and graph-only undo steps; each patch was correct and the
//! disease was structural. One owner, one counter.

use voxelith::core::World;
use voxelith::editor::Socket;
use voxelith::io::ProjectMetadata;
use voxelith::procgen::PipelineGraph;

/// The open project: world + sockets + pipeline graph + metadata.
///
/// Dirty state is a private [`DocumentMeta`] — revisions only move
/// through the methods below, so the invariant the two booleans used
/// to carry by convention ("autosave must never clear the
/// unsaved-changes flag") holds by construction: `mark_autosaved`
/// cannot touch the saved mark.
pub(super) struct Document {
    pub world: World,
    pub sockets: Vec<Socket>,
    pub graph: PipelineGraph,
    /// Identity of the project (`name` / `author` / `created_at` /
    /// `modified_at`). Held by the session and written out explicitly,
    /// so Save As onto an unrelated file no longer inherits *that*
    /// file's identity (the old read-back-at-save scheme did).
    pub metadata: ProjectMetadata,
    meta: DocumentMeta,
}

/// Three high-water marks over one monotonic revision counter.
///
/// `unsaved` and `autosave_due` are *derived*, never stored: a state
/// where autosave has run but the user's own file is stale is simply
/// `saved < autosaved == revision` — representable, correct, and not a
/// pair of booleans someone must remember to keep from clobbering each
/// other.
///
/// Deliberately NOT the same counter as `CommandHistory::generation()`:
/// that one only moves for undoable edits and exists to invalidate a
/// parked agent batch; this one moves for *every* document change,
/// sockets and graph edits included, which are managed outside the
/// undo history on purpose.
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

    /// The document and the user's own file now agree: save, open,
    /// new, import, reload. Retires the autosave debt too — the file
    /// holds this exact state, so rewriting the crash net would be
    /// pointless.
    pub fn mark_saved(&mut self) {
        self.meta.saved = self.meta.revision;
        self.meta.autosaved = self.meta.revision;
    }

    /// A successful autosave write. Touches ONLY the autosave mark —
    /// `unsaved()` keeps answering true, which is the invariant the
    /// old boolean pair carried by convention: a clean exit deletes
    /// the autosave, so "edit → autosave → close" must still prompt.
    pub fn mark_autosaved(&mut self) {
        self.meta.autosaved = self.meta.revision;
    }

    /// Recovered work is a document with no file of its own: advance
    /// the revision past every mark so `unsaved()` holds until a
    /// manual save — otherwise the guard would wave a clean exit
    /// through, and the exit would delete the autosave that held the
    /// only copy. The autosave debt is retired (the autosave IS this
    /// state); the timer restart is the caller's clock to manage.
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
        // THE invariant: autosave is a crash net that a clean exit
        // deletes. If its write cleared `unsaved`, the sequence
        // "edit → autosave fires → close" skipped the prompt and then
        // destroyed the only copy on the way out.
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
        // Matches the boolean model this replaces: revisions are
        // monotonic, so undoing back to visually-identical state does
        // not clear the flag. (A content hash could; recorded as the
        // deliberate, pre-existing limitation it is.)
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
