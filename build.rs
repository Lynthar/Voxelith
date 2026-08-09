//! Embeds the application icon into the Windows executable resource
//! table — what Explorer, the taskbar and shortcuts display. This is
//! distinct from the *window* icon, which winit sets at runtime from
//! the same artwork (`window_icon()` in `src/app/handler.rs`).
//!
//! The icon applies to every feature configuration (a headless
//! `voxelith.exe` still shows up in Explorer), so this is keyed on the
//! OS rather than on `gui`. A build on a non-Windows host compiles the
//! empty stub below and carries no `winresource` at all — Cargo.toml
//! provides that dependency only for Windows. (Known wart, shared with
//! the rest of the ecosystem: `[target.…]` build-dependencies and this
//! `#[cfg]` disagree about host vs. target when *cross*-compiling
//! between Windows and non-Windows. Same-platform builds — the dev
//! machine, CI, a Linux container — are all consistent.)

fn main() {
    println!("cargo:rerun-if-changed=assets/branding/voxelith.ico");
    embed_icon();
}

#[cfg(target_os = "windows")]
fn embed_icon() {
    winresource::WindowsResource::new()
        .set_icon("assets/branding/voxelith.ico")
        .compile()
        .expect("embed assets/branding/voxelith.ico into the executable");
}

#[cfg(not(target_os = "windows"))]
fn embed_icon() {}
