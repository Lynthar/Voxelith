#!/usr/bin/env python3
"""Run the glTFast gate: bake a fixture, import it with real glTFast, check UV0.

The gate answers the one question a unit test cannot: does the tint zone that
`src/io/gltf.rs` mirrors into TEXCOORD_0.x still reach a Unity mesh after
glTFast has had its way with the file? Exits non-zero when it does not.

Needs a Unity 6 editor (`--unity`, `$VOXELITH_UNITY`, or the Hub's default
location) and a built `voxelith` binary (`--voxelith`, or `cargo build`).
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "UnityProject"
REPO = HERE.parent.parent


def editor_version() -> str:
    # First line only: the file also carries m_EditorVersionWithRevision.
    text = (PROJECT / "ProjectSettings" / "ProjectVersion.txt").read_text(encoding="utf-8")
    return text.splitlines()[0].split(":", 1)[1].strip()


def find_unity(explicit: str | None) -> Path:
    """Locate a Unity editor.

    Raises:
        SystemExit: when no editor is found, listing where it looked.
    """
    if explicit or os.environ.get("VOXELITH_UNITY"):
        return Path(explicit or os.environ["VOXELITH_UNITY"])
    want = editor_version()
    system = platform.system()
    if system == "Windows":
        roots = [Path("C:/Program Files/Unity/Hub/Editor")]
        tail = Path("Editor/Unity.exe")
    elif system == "Darwin":
        roots = [Path("/Applications/Unity/Hub/Editor")]
        tail = Path("Unity.app/Contents/MacOS/Unity")
    else:
        roots = [Path.home() / "Unity/Hub/Editor", Path("/opt/unity/editors")]
        tail = Path("Editor/Unity")
    for root in roots:
        exact = root / want / tail
        if exact.exists():
            return exact
        for candidate in sorted(root.glob("6000.*"), reverse=True) if root.exists() else []:
            if (candidate / tail).exists():
                return candidate / tail
    sys.exit(f"no Unity editor found (wanted {want}); pass --unity or set VOXELITH_UNITY")


def find_voxelith(explicit: str | None) -> Path:
    """Build the exporter under test, or take the caller's binary.

    Always builds: a stale binary left in target/ would have the gate vouch
    for an exporter nobody is running any more.

    Raises:
        SystemExit: when cargo is missing or the build fails.
    """
    if explicit:
        return Path(explicit)
    exe = "voxelith.exe" if platform.system() == "Windows" else "voxelith"
    target = Path(os.environ.get("CARGO_TARGET_DIR", REPO / "target"))
    build = subprocess.run(["cargo", "build", "--quiet"], cwd=REPO)
    if build.returncode != 0:
        sys.exit("cargo build failed; fix it or pass --voxelith")
    binary = target / "debug" / exe
    if not binary.exists():
        sys.exit(f"cargo build produced no {binary}; pass --voxelith")
    return binary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unity")
    ap.add_argument("--voxelith")
    ap.add_argument("--zones", default="1,2,3", help="zones the gate demands in UV0.x")
    ap.add_argument("--keep", action="store_true", help="keep the baked .glb")
    args = ap.parse_args()

    # The editor log carries bytes that a Windows GBK console cannot encode.
    sys.stdout.reconfigure(errors="replace")
    unity, voxelith = find_unity(args.unity), find_voxelith(args.voxelith)
    work = Path(tempfile.mkdtemp(prefix="gltfast-gate-"))
    glb = work / "gate.glb"
    # No gltfpack: compression bugs would be indistinguishable from import bugs.
    subprocess.run([str(voxelith), "exec", str(HERE / "fixture.ops.json"), "--export", str(glb)],
                   check=True, stdout=subprocess.DEVNULL)
    print(f"baked {glb} ({glb.stat().st_size} bytes)")

    cmd = [str(unity), "-batchmode", "-nographics", "-projectPath", str(PROJECT),
           "-executeMethod", "GltfastGate.Run", "-glb", str(glb), "-zones", args.zones,
           "-logFile", "-"]
    print(f"$ {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    keep = [ln for ln in proc.stdout.splitlines()
            if ln.startswith(("PASS ", "FAIL ", "GATE ", "glTFast "))
            or "GATE FAILURES" in ln or "error CS" in ln]
    print("\n".join(keep) or proc.stdout[-4000:])
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
