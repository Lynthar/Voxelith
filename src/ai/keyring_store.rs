//! Persistence of AI provider API keys in the OS keychain.
//!
//! Uses the `keyring` crate, which targets:
//! - Windows: Credential Manager (per-user)
//! - macOS: Keychain (per-user, login chain)
//! - Linux: Secret Service via D-Bus (typically GNOME Keyring / KWallet)
//!
//! Per-app service identifier is `voxelith`; the credential's
//! `username` field encodes which provider the key belongs to (e.g.
//! `fal_ai`). Future providers add new usernames without touching
//! existing entries.
//!
//! Keys are deliberately **not** persisted in `prefs.ron` — that file
//! sits next to user content and would be exposed by sharing a
//! project bundle.

use thiserror::Error;

const SERVICE: &str = "voxelith";

/// Wrapper around `keyring::Error` that's `Send + Sync + 'static` for
/// passing through the AI worker channel. The underlying error is
/// stringified rather than re-exported so the rest of the crate
/// doesn't need to depend on `keyring` types in its signatures.
#[derive(Debug, Error)]
pub enum KeyringError {
    #[error("Keyring backend: {0}")]
    Backend(String),
    #[error("No API key set")]
    NotFound,
}

impl From<keyring::Error> for KeyringError {
    fn from(e: keyring::Error) -> Self {
        match e {
            keyring::Error::NoEntry => KeyringError::NotFound,
            other => KeyringError::Backend(other.to_string()),
        }
    }
}

/// Read the API key for `provider` (e.g. `"fal_ai"`). Returns
/// `Err(NotFound)` when the user hasn't set one yet — the UI should
/// treat that as "show the API key entry box" rather than a crash.
pub fn load_api_key(provider: &str) -> Result<String, KeyringError> {
    let entry = keyring::Entry::new(SERVICE, provider)?;
    Ok(entry.get_password()?)
}

/// Write the API key for `provider`. Overwrites any existing entry.
pub fn save_api_key(provider: &str, key: &str) -> Result<(), KeyringError> {
    let entry = keyring::Entry::new(SERVICE, provider)?;
    entry.set_password(key)?;
    Ok(())
}

/// Remove the stored key for `provider`. No-op if not present.
pub fn clear_api_key(provider: &str) -> Result<(), KeyringError> {
    let entry = keyring::Entry::new(SERVICE, provider)?;
    match entry.delete_credential() {
        Ok(()) => Ok(()),
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Quick non-fatal check: does a key exist for `provider`? Used by
/// the AI panel to decide whether to enable the Generate button vs
/// show the "Set API key…" prompt.
pub fn has_api_key(provider: &str) -> bool {
    matches!(load_api_key(provider), Ok(s) if !s.is_empty())
}
