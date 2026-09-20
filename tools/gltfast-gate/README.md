# glTFast gate

Voxelith mirrors each voxel's faction tint zone into `TEXCOORD_0.x` because
glTFast drops custom attributes like `_TINTZONE`. Unit tests can prove the
exporter *writes* that mirror; only a real glTFast import can prove it
*survives*. That is what this gate does, and it is the reason the reference
shader calls this the one unverified link in the consumption contract.

## Running it

```
python tools/gltfast-gate/run.py
```

Needs a Unity 6 editor with a valid licence. The runner finds one from
`ProjectSettings/ProjectVersion.txt`, or takes `--unity` / `$VOXELITH_UNITY`.
It always runs `cargo build` first and uses that binary — a stale one left in
`target/` would have the gate vouch for an exporter nobody runs any more.

## What it asserts

`fixture.ops.json` places voxels in tint zones 0, 1, 2 and 3. The runner
exports it straight to `.glb` — no `gltfpack`, so a compression bug can never
be mistaken for an import bug — and hands the file to glTFast. The editor
script then requires, of every mesh glTFast produced:

- UV0 exists and covers every vertex (the channel was not pruned);
- UV0.x holds whole numbers (the zone rode through unscaled);
- zones 1, 2 and 3 all arrive (`--zones` changes the set).

Only `.x` is asserted: glTFast flips V for Unity's texture origin, so a zone
carried in `.y` would come back as `1 - zone`. That flip is why the mirror
uses `.x`, and asserting `.y` here would just encode glTFast's convention.

## When it fails

The failure the contract predicts is an empty or short UV0 — glTFast pruning a
UV set no material samples. The reference shader's header
(`docs/reference/VoxelithUberURP.shader`) lists the fixes in preference order:
assign the material at import time, use an import callback, or route through
Blender. Until this gate passes on your glTFast version, treat per-zone tint as
unproven; whole-model `_BaseColor` tint always works.

## What is checked in

Only the manifest, the package lock, `ProjectVersion.txt` and `Assets/` —
Unity rebuilds the rest. The lock file is what pins the glTFast version the
gate ran against, so a version bump shows up as a diff and demands a re-run.
