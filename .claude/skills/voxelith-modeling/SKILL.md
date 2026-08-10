---
name: voxelith-modeling
description: Build, edit, inspect and export voxel models with Voxelith — its headless CLI (`voxelith exec / inspect / render / generators / eval / bake`) and its MCP tool set. Use this skill whenever a task involves making a voxel model or asset (a castle, a tree, a sword, terrain, a game prop), editing or inspecting a `.vxlt` project, producing `.glb` / `.obj` / `.vox` from voxels, batch-generating asset variants, or answering whether a model is actually connected / symmetric / hollow. Use it even when the request just says "build me a voxel X" and never names Voxelith, as long as a Voxelith checkout or binary is available. It covers choosing between the CLI and the MCP path, the ops format, and the modeling technique that keeps a first attempt from being wrong.
---

# Building voxel models with Voxelith

Voxelith turns a JSON batch of drawing operations into a voxel model and
answers questions about the result: exact counts, exact coordinates, a
rendered picture, and structural measurements no picture gives you
reliably. Everything here runs with **no GPU and no window**, so it works
wherever you do.

Coordinates are integer cells, **Y is up**, and every region is
**inclusive on both ends** — `min [0,0,0] max [1,1,1]` is 8 cells, not 1.

**Before writing your first batch, read `references/ops.md`.** It is the
op vocabulary, the report shape and the error codes. Guessing at op names
costs a round trip, because unknown fields are refused rather than
ignored — which is a feature, but only if you read the message.

## Getting the binary

```bash
cargo build --release --no-default-features   # headless: no wgpu/winit/egui
./target/release/voxelith generators          # confirm it runs
```

Drop `--no-default-features` only if you also want the GUI editor. Use
`--release` either way: meshing and procedural generation are CPU-heavy
and a debug build is slow enough to change how you work.

If a `voxelith` binary is already on PATH, use it directly — the examples
below do.

## Pick your path first

Three ways to drive the same primitives. The choice is not cosmetic; it
changes what you can do in one step.

| Situation | Use | Why |
|---|---|---|
| A human has the model open in the Voxelith editor right now | **in-editor bridge** — `voxelith --agent-port 8737`, connect over HTTP | You edit the world they are looking at, sharing their undo stack. No file in the middle, so no race. 7 tools, no file operations. |
| Iterating on **one** model, wanting to look at it repeatedly | **MCP** — `voxelith mcp` | The document is resident: undo reaches back through earlier batches, the selection persists, and `render_views` hands you images inline. |
| Batch work, variants, anything that should be reproducible | **CLI** — `voxelith exec` | An `ops.json` is a file: diff it, commit it, re-run it, loop over it in a shell. Costs no context. |

**What MCP genuinely buys you**, measured rather than assumed:

- `apply_ops` answers with the report *and* a description of the same
  world, so "did that work?" is never a second call. The CLI needs
  `--describe`, and after a failure you re-run to find out.
- `render_views` returns the PNG inline. Apply → look → fix is three
  calls, not five plus a file read.
- `undo` across calls. When a shape comes out wrong, one `undo` puts you
  back; the CLI has to re-run the whole batch from the saved file.
- The cost: **call `new_project` between unrelated models.** One resident
  document means the last model's terrain is still there otherwise.

**What the CLI genuinely buys you:**

- **Context.** MCP tool schemas ride in your context every single turn.
  The CLI costs zero tokens until you run it.
- **Loops.** "Twenty variants, one seed each" is twenty tool calls over
  MCP and four lines of shell here.
- **Reproducibility.** The ops file *is* the record of how the model was
  made. It survives the conversation.

When in doubt for a single model: MCP if it is available, CLI otherwise.
Both take the identical ops batch, so switching later costs nothing.

## The loop

```bash
voxelith exec ops.json --out hut.vxlt --describe     # build
voxelith exec more.json --in hut.vxlt --out hut.vxlt # keep editing
voxelith render hut.vxlt --view iso                  # look at it
```

Write a batch → run it → **read the report** → look at the result → fix.
State lives in the `.vxlt`, so a human can open it in the editor at any
point and take over.

stdout is JSON and nothing else; logs go to stderr. `{"ok": true, …}`
with exit 0, or `{"ok": false, "error": {…}}` with exit 1.

A batch is **atomic** — a failure writes nothing, so fix the op named by
`op_index` and resend. It is **sequential** — each op sees the previous
ones' results. And it lands as **one undo entry**, which is why a human
can Ctrl+Z your work as a unit.

Use `--dry-run` when a batch is large or you are unsure about a
coordinate. It is a real preview, not a prediction: `--describe` and
`--slice` alongside it show the model **as the batch would leave it**.

**This is how you check a batch before committing it, and it replaces
writing your own verifier.** A dry run reports the full `structure`
block, so two halves that miss each other by a cell come back as
`components: 2` with nothing written to disk:

```bash
voxelith exec ops.json --dry-run --describe    # components, floating parts, symmetry — no write
```

Reach for that before hand-rolling a flood fill over your own cell list.
The tool already owns the question, and its answer is the one you will be
graded against.

## Check in this order

**1. `notes` in the report.** This is where a generator says it degraded
something. It is easy to skip and it is never noise.

**2. `structure.components` from `--describe`.** A one-cell gap is the
single most common way a model looks finished and isn't — the blade that
misses its crossguard reads as done from every angle and is two objects
to a mesher, an exporter and a physics engine. No render answers this.
Check `floating_components` in the same breath.

**3. `--slice` for geometry you can count.** One plane as ASCII is what
actually catches "the door is one cell too high". The first line states
the axis ranges and row order, so you never have to guess which way is
up.

**4. `voxelith render --view iso` to see it.** Orthographic, so a
straight wall looks straight and equal cells stay equal size.
`cells_per_pixel` in the report converts a pixel error back into cells.

The measurements are **measurements, never verdicts**. Two components is
wrong for a sword and right for a pair of boots; a floating part is a bug
in a chair and the entire point of a tree canopy. Decide what the numbers
should be for *this* object before you read them.

**And they answer "is it assembled correctly", never "does it look like
the thing".** For anything organic — a fish, a creature, a plant — the
rendered view is the primary judge. A fish with rectangular slab fins
passes every structural bar there is.

## Four rules that cost a retry every time

The rest are in `references/recipes.md`. These four are the ones worth
carrying without opening it:

- **`sphere` and `cylinder` include a cell when its centre is within
  `radius + 0.5`.** `radius: 1` is a solid 3×3. Size a curve by the span
  you want and then add one; estimating from `r²` gives you an arch about
  half as tall as you meant.
- **`hollow` is destructive on anything that tapers.** It clears cells
  whose six neighbours are all solid, so a spire of stacked solid discs
  becomes a stack of rings that touch nothing — one real case went from 1
  component to 61. Build shells from the start (`filled: false`) and the
  final `hollow` is a harmless no-op.
- **A `box` one cell thick with `filled: false` fills the whole face.** A
  one-cell-thick box *is* its own shell. For a rectangular outline, use
  twelve `line` ops.
- **`render --view front` looks along −Z, so it shows the model's +Z
  face.** Put a facade on +Z. A model that came out backwards takes one
  `mirror` on `z` over its bounding box, which leaves X symmetry intact.

## Prefer a graph when the shape is generated

If the shape comes from a generator rather than from your hand, send a
**pipeline graph** instead of placed voxels (`op: "graph"`). It is stored
*with the project*, so the human who opens the `.vxlt` afterwards finds
it in the editor's Graph panel with the sliders live — they can re-roll
the seed or push the terrain higher without you. Hand-placed voxels give
them a result they can only repaint.

Run `voxelith generators` first: it prints every generator with its
parameters at their default values **and a working `graph_template`**.
That listing is the format documentation — copy it, change what you care
about, send it back. Format details are in `references/ops.md`.

## Grade your own work

`evals/cases/*.json` and `evals/advanced/*.json` in the repo pair a
modeling task with the properties its result has to have:

```bash
voxelith eval evals/cases/connected-sword.json --project sword.vxlt
```

Every assertion is named with what was expected and what was measured;
exit code is zero only if all of them held. It is arithmetic on the same
numbers `--describe` gives you, so the bar you are graded against and the
feedback you build with cannot drift apart.

**Read the cases even when nobody asked you to run one.** They spell out
what "finished" tends to mean for a kind of object, including the one bar
that grades *method* rather than result: `graph_nodes` asks whether you
produced something that stays adjustable. A hand-placed model clears
every size and count bar and fails that one.

## References

- **`references/ops.md`** — the op vocabulary, voxel and `write_mode`
  format, transforms, generators, pipeline graphs, the report and
  description shape, and every error code. Read before your first batch.
- **`references/recipes.md`** — how to build specific things so they come
  out right: symmetry, shells and hollowing, curves and tapers, organic
  shapes, stairs, terrain with real relief. Read when you know what you
  are building.

## What this doesn't do

`exec --export` is the interactive File ▸ Export, headless: greedy mesh,
model space untouched. For engine-ready assets — pivot / up-axis /
unit-scale placement, Marching-Cubes smoothing, `gltfpack` compression,
whole directories at once — use `voxelith bake <spec.json>`.

Sockets (named attachment points) and the clipboard are not editable from
ops yet. They survive a load → edit → save round trip untouched.
