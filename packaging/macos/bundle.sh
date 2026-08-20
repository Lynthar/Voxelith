#!/usr/bin/env bash
# Assemble Voxelith.app — the only form in which macOS shows the icon.
#
# winit documents `set_window_icon` as Unsupported on macOS, so the
# artwork the editor embeds for Windows can never reach the Dock: there
# the icon is read from this bundle's Resources/voxelith.icns, named by
# CFBundleIconFile. A bare `cargo run` binary therefore shows the generic
# executable icon no matter what the code does — that is expected, not a
# regression, and building this bundle is the fix.
#
#   packaging/macos/bundle.sh [--no-build]
#
# Writes target/release/bundle/macos/Voxelith.app.

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
here="$root/packaging/macos"
icns="$root/assets/branding/voxelith.icns"
app="$root/target/release/bundle/macos/Voxelith.app"

[ "$(uname -s)" = "Darwin" ] || {
	echo "bundle.sh: macOS only — it packages a Mach-O binary into a .app" >&2
	exit 1
}
[ -f "$icns" ] || {
	echo "bundle.sh: missing $icns (run: python assets/branding/generate.py)" >&2
	exit 1
}

# One version, taken from the [package] section rather than written here
# a second time. Stop at the next section header so a dependency's own
# `version =` can't be picked up.
version=$(sed -n '/^\[package\]/,/^\[[^p]/ s/^version *= *"\([^"]*\)".*/\1/p' "$root/Cargo.toml" | head -1)
[ -n "$version" ] || { echo "bundle.sh: no version in Cargo.toml" >&2; exit 1; }

if [ "${1:-}" != "--no-build" ]; then
	cargo build --release --manifest-path "$root/Cargo.toml"
fi

bin="$root/target/release/voxelith"
[ -x "$bin" ] || { echo "bundle.sh: $bin not built" >&2; exit 1; }

rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$bin" "$app/Contents/MacOS/voxelith"
cp "$icns" "$app/Contents/Resources/voxelith.icns"
sed "s/__VERSION__/$version/g" "$here/Info.plist" > "$app/Contents/Info.plist"
printf 'APPL????' > "$app/Contents/PkgInfo"

# Ad-hoc signature. Gatekeeper still won't accept a downloaded copy
# (that needs a Developer ID and notarization), but a locally built app
# launches without the "damaged" dialog Apple Silicon shows for unsigned
# bundles.
if command -v codesign >/dev/null 2>&1; then
	codesign --force --sign - --timestamp=none "$app" >/dev/null 2>&1 &&
		echo "signed ad-hoc" || echo "warning: ad-hoc codesign failed; app may not launch" >&2
fi

plutil -lint "$app/Contents/Info.plist" >/dev/null || {
	echo "bundle.sh: generated Info.plist is malformed" >&2
	exit 1
}

echo "built $app (version $version)"
echo
echo "Run it:        open '$app'"
echo "Install it:    cp -R '$app' /Applications/"
echo
echo "The Dock caches icons per bundle id. If a rebuilt icon does not"
echo "show up, run:  touch '$app' && killall Dock"
