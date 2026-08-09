//! Where the server is allowed to read and write.
//!
//! Every path a tool accepts has to resolve inside one root directory.
//! Over stdio that buys little on its own — the client launched this
//! process and could have read the disk directly — but the same tool
//! bodies serve the HTTP transport, where "open this project" arrives
//! from whoever can reach the port. One rule for both beats a rule that
//! changes with the transport: an agent's working recipe shouldn't stop
//! working because it moved from a child process to a URL.

use std::fmt;
use std::path::{Path, PathBuf};

/// The directory every project path must land inside.
#[derive(Debug, Clone)]
pub struct Root {
    dir: PathBuf,
}

/// Why a path was refused.
///
/// All of these are answerable by sending a different path, which is
/// why they're separate from "the file didn't load" — that one an agent
/// can't fix by rewriting the argument.
#[derive(Debug)]
pub enum PathError {
    Empty,
    /// Nothing to write *to* — a bare `..`, or a root directory.
    Unanchored(PathBuf),
    /// The containing directory doesn't exist or can't be read. Also
    /// what an unreadable path component anywhere up the chain lands on.
    NoDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    Outside {
        path: PathBuf,
        root: PathBuf,
    },
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathError::Empty => write!(f, "a path is required, and this one is empty"),
            // Through this module's own `display`, not `Path::display`:
            // these paths have been through `canonicalize`, which on
            // Windows hands back the `\\?\` verbatim form. That prefix
            // is correct, unreadable, and — pasted back by an agent
            // taking the message at its word — not what it typed.
            PathError::Unanchored(path) => write!(
                f,
                "{} doesn't name a file — give a path ending in a file name",
                display(path)
            ),
            PathError::NoDirectory { path, source } => write!(
                f,
                "the directory holding {} can't be read ({source}); create it first",
                display(path)
            ),
            PathError::Outside { path, root } => write!(
                f,
                "{} is outside this server's root ({}); pass a path inside the root, \
                 or restart the server with --root pointing somewhere that contains it",
                display(path),
                display(root)
            ),
        }
    }
}

impl std::error::Error for PathError {}

/// A canonical path as a person — or a model — should see it.
///
/// `canonicalize` hands back Windows verbatim paths (`\\?\C:\…`). They
/// are correct, and they're what we keep for actual file operations
/// because that form is what lifts the `MAX_PATH` limit, but echoing one
/// back in a tool answer reads as line noise and invites an agent to
/// copy the prefix into places it doesn't belong.
pub fn display(path: &Path) -> String {
    let text = path.display().to_string();
    if let Some(unc) = text.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{unc}");
    }
    text.strip_prefix(r"\\?\").unwrap_or(&text).to_string()
}

impl Root {
    /// Anchor at `dir`, which must exist. Stored canonicalized so the
    /// containment test compares like with like — on Windows that means
    /// both sides are `\\?\`-prefixed, and everywhere it means symlinks
    /// in the root's *own* path don't cause false rejections.
    pub fn new(dir: &Path) -> std::io::Result<Self> {
        Ok(Self {
            dir: dir.canonicalize()?,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Resolve a requested path against the root, or refuse it.
    ///
    /// Relative paths hang off the root; absolute ones are taken as
    /// given and then checked. Either way the answer is canonical, so
    /// `..` segments and symlinks are resolved *before* the containment
    /// test rather than pattern-matched away — a check on the literal
    /// string is a check on spelling, not on where the write lands.
    pub fn resolve(&self, requested: &str) -> Result<PathBuf, PathError> {
        if requested.trim().is_empty() {
            return Err(PathError::Empty);
        }
        let requested = Path::new(requested);
        let joined = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            self.dir.join(requested)
        };

        // A save target usually doesn't exist yet, and you can't
        // canonicalize what isn't there — so fall back to canonicalizing
        // the directory (which must exist) and putting the file name
        // back. The directory is the part that could contain a symlink
        // or a `..` worth resolving; the final component can't.
        let resolved = match joined.canonicalize() {
            Ok(path) => path,
            Err(_) => {
                let parent = joined
                    .parent()
                    .ok_or_else(|| PathError::Unanchored(joined.clone()))?;
                let name = joined
                    .file_name()
                    .ok_or_else(|| PathError::Unanchored(joined.clone()))?;
                parent
                    .canonicalize()
                    .map_err(|source| PathError::NoDirectory {
                        path: joined.clone(),
                        source,
                    })?
                    .join(name)
            }
        };

        if !resolved.starts_with(&self.dir) {
            return Err(PathError::Outside {
                path: resolved,
                root: self.dir.clone(),
            });
        }
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each test gets its own tree — the suite runs in parallel.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("voxelith_root_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("nested")).unwrap();
        std::fs::write(dir.join("there.vxlt"), b"x").unwrap();
        dir
    }

    fn root(dir: &Path) -> Root {
        Root::new(dir).unwrap()
    }

    #[test]
    fn a_relative_path_hangs_off_the_root() {
        let dir = scratch("relative");
        let root = root(&dir);
        assert_eq!(
            root.resolve("there.vxlt").unwrap(),
            dir.canonicalize().unwrap().join("there.vxlt")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_that_does_not_exist_yet_is_fine_if_its_directory_does() {
        // The save case. Refusing it would mean an agent could only ever
        // overwrite files someone else had already made.
        let dir = scratch("save_target");
        let root = root(&dir);
        assert!(root.resolve("nested/new.vxlt").is_ok());
        assert!(matches!(
            root.resolve("no/such/dir/new.vxlt"),
            Err(PathError::NoDirectory { .. })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn climbing_out_with_dot_dot_is_refused() {
        // The whole point: `..` is resolved and *then* tested, so it
        // can't be spelled around — `nested/../../x` leaves the root
        // even though no single component looks suspicious.
        let dir = scratch("escape");
        let root = root(&dir);
        for attempt in ["../outside.vxlt", "nested/../../outside.vxlt", ".."] {
            assert!(
                root.resolve(attempt).is_err(),
                "{attempt} should not have resolved"
            );
        }
        // …while a `..` that stays inside is not the enemy.
        assert!(root.resolve("nested/../there.vxlt").is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_absolute_path_is_judged_by_where_it_lands() {
        let dir = scratch("absolute");
        let root = root(&dir);
        let inside = dir.join("there.vxlt");
        assert!(root.resolve(inside.to_str().unwrap()).is_ok());

        let outside = std::env::temp_dir().join("voxelith_root_absolute_elsewhere.vxlt");
        assert!(matches!(
            root.resolve(outside.to_str().unwrap()),
            Err(PathError::Outside { .. })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_path_says_so_rather_than_resolving_to_the_root() {
        let dir = scratch("empty");
        let root = root(&dir);
        assert!(matches!(root.resolve(""), Err(PathError::Empty)));
        assert!(matches!(root.resolve("   "), Err(PathError::Empty)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_refusal_names_the_root_so_it_can_be_acted_on() {
        let dir = scratch("message");
        let root = root(&dir);
        let message = root.resolve("../outside.vxlt").unwrap_err().to_string();
        assert!(message.contains("--root"), "got: {message}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_shown_path_carries_no_verbatim_prefix() {
        // What an agent reads back and may well paste into the next
        // call. `\\?\C:\…` is correct and unreadable.
        let dir = scratch("display");
        let shown = display(&root(&dir).resolve("there.vxlt").unwrap());
        assert!(!shown.contains(r"\\?\"), "got: {shown}");
        assert!(shown.ends_with("there.vxlt"), "got: {shown}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
