# The ops format

The complete edit vocabulary, the answers it gives back, and every way it
refuses. The same batch object drives the CLI (`voxelith exec`), the MCP
`apply_ops` tool and the in-editor bridge, so nothing here is specific to
one of them.

- [Commands](#commands)
- [The batch](#the-batch)
- [Ops](#ops)
- [Voxels](#voxels)
- [write_mode](#write_mode)
- [Transforms](#transforms)
- [Generators](#generators)
- [Pipeline graphs](#pipeline-graphs)
- [Reading the result](#reading-the-result)
- [describe](#describe)
- [slice](#slice)
- [render](#render)
- [When it refuses](#when-it-refuses)
- [Batch export: bake](#batch-export-bake)
- [Over MCP](#over-mcp)
- [The in-editor bridge](#the-in-editor-bridge)

## Commands

| Command | What it does |
|---|---|
| `exec <ops.json>` | Apply a batch. `--in <p.vxlt>` start from a project, `--out <p.vxlt>` save, `--export <f.glb\|f.obj\|f.vox>`, `--describe`, `--slice <json>`, `--dry-run` |
| `inspect <p.vxlt>` | Read-only: always describes, `--slice <json>` optional |
| `render <p.vxlt>` | Draw it as PNG. `--view <iso\|front\|…\|all>`, `--size <px>`, `--out <f.png>` |
| `generators` | Every callable generator with its parameters at their defaults, plus a working `graph_template` |
| `eval <cases>` | Grade finished models. `--project <p.vxlt>` for one, `--results <dir>` for a whole run. Exit 0 only when every case passed |
| `bake <spec.json>` | Batch-export projects to `.glb` with a placement transform. `--shard i/n` splits the work |

`--dry-run` refuses to be combined with `--out` / `--export`, since a dry
run writes nothing.

On PowerShell, quote JSON arguments with single quotes:
`--slice '{"axis":"y","index":1}'`.

## The batch

```jsonc
{
  "version": 1,                       // required
  "ops": [ /* 1..=256 ops */ ],
  "options": { "dry_run": false }     // optional
}
```

**Atomic** (any failure changes nothing), **sequential** (each op sees
the results of the ones before it), **one undo entry**. Unknown fields
are errors, not warnings — a misspelling is reported instead of silently
doing nothing.

## Ops

| `op` | Fields | Notes |
|---|---|---|
| `box` | `min`, `max`, `voxel`, `filled?=true`, `write_mode?` | `filled: false` leaves a 1-cell shell |
| `sphere` | `center`, `radius`, `voxel`, `filled?`, `write_mode?` | diameter is `2·radius + 1` |
| `cylinder` | `base`, `radius`, `height`, `axis?="y"`, `voxel`, `filled?`, `write_mode?` | `base` is the centre of the bottom face; extends `height` cells along +`axis` |
| `line` | `from`, `to`, `voxel`, `write_mode?` | 1 cell thick, both ends included; a diagonal costs its length, not its bounding box |
| `hollow` | `min`, `max` | clears cells whose 6 neighbours are all solid |
| `set_voxels` | `voxels: [[x,y,z,voxel], …]`, `write_mode?` | ≤ 4096 per op; for detail, not bulk |
| `generate` | `generator`, `params?`, `translate?`, `write_mode?` | see [Generators](#generators) |
| `graph` | `graph`, `apply?=true`, `translate?`, `write_mode?` | store a pipeline graph on the document and evaluate it |
| `graph_edit` | `edits`, `apply?=true`, `translate?`, `write_mode?` | change the graph the document already has |
| `select` | `min`, `max` | sets the session selection |
| `deselect` | — | |
| `rotate` | `axis`, `quarters` (1/2/3), `region?` | see [Transforms](#transforms) |
| `mirror` | `axis`, `region?` | flips a region's contents in place |
| `mirror_copy` | `axis`, `plane?`, `region?`, `write_mode?` | keeps the original and stamps a reflection |

There is no separate erase, paint or fill op: `"voxel": "air"` erases,
`write_mode: "only_solid"` paints, and `box` fills.

Two shapes behave in a way worth knowing before you size anything:

- **`sphere` and `cylinder` include a cell when its centre is within
  `radius + 0.5`, not `radius`.** `radius: 1` is a solid 3×3 — the
  diagonal cell is in — and a `radius: 3` arch has only three courses
  above its springing line (7 / 7 / 5 / 3 wide).
- **A `box` one cell thick with `filled: false` fills the whole face.**
  It isn't drawing an outline; a one-cell-thick box *is* its own shell.
  For a rectangular outline use twelve `line` ops. The one place the
  behaviour is useful: it states "this face is solid" in one op.

`hollow` is safe on a box-like shell of even wall thickness and
**destructive on a solid shape that tapers** — see `recipes.md`.

## Voxels

`"air"`, or an object:

```jsonc
{ "rgb": [200, 100, 50],   // required
  "material": 1,            // optional, default 1 (0 means air — use "air")
  "emissive": false,        // optional → glTF emissive material
  "metallic": false,        // optional → glTF metallic material
  "tint_zone": 0 }          // optional 0..=3, faction recolour zone
```

Every voxel in a Voxelith world is opaque. `"a": 255` is accepted;
anything else is refused rather than quietly forced.

## write_mode

| Value | Writes where |
|---|---|
| `replace` (default) | everywhere |
| `only_air` | the cell is currently empty — build around existing geometry |
| `only_solid` | the cell is currently solid — recolour without changing the silhouette |

`only_air` is what makes "build the trunk, then pack leaves around it"
work: the canopy fills in beside the trunk instead of overwriting it.
`only_solid` is the safe way to add decoration late — it cannot change
any structural measurement, so a recolouring pass cannot break a model
that was already passing.

## Transforms

`rotate` and `mirror` move a region's **contents**. If `region` is
omitted they use the current selection; with neither you get
`no_selection`.

- `rotate`: `quarters` is 1 = +90°, 2 = 180°, 3 = 270°, right-handed
  about `axis`. The region's `min` corner stays put and the extents swap
  for odd quarter turns (a 4×1×2 region rotated about Y becomes 2×1×4).
  If the op used the selection, the selection follows the result.
- `mirror_copy`: `plane` is the **seam between cell `plane - 1` and cell
  `plane`**, so cell `p` lands at `2·plane − 1 − p`. It defaults to the
  seam just past the region's `max` on that axis, which puts the copy
  flush against the original — the usual "build the left half, mirror it"
  move needs no `plane` at all. `plane: 0` mirrors across the world
  origin. Air is not copied, so a mirror stamps a shape without erasing
  what is on the far side.

Both consequences of a voxel being a cell rather than a point:

- **An even span's mirror plane falls between cells, not on one.** A
  model spanning x 0..23 mirrors about the seam between 11 and 12, so no
  single column sits on the axis.
- **Symmetry is cheaper to construct than to repair.** See `recipes.md`.

## Generators

`voxelith generators` prints every generator with its parameters at their
default values. That listing **is** the template: copy `default_params`,
change what you care about, send it back as `params`. Anything you leave
out keeps its default; a misspelled parameter is an error that lists the
real ones.

```json
{ "op": "generate", "generator": "builtin.lsystem_tree",
  "params": { "seed": 7, "iterations": 4 }, "translate": [14, 1, 4] }
```

Generators place their output where their own parameters say (often
around the origin); `translate` offsets the whole result. Check the
reported `world_aabb` afterwards — a tree at `iterations: 4` is over 100
cells tall, which is easy to not expect.

Three things about `builtin.perlin_terrain` that cost retries otherwise:

- **It centres itself.** `width`/`depth` of 32 lands on x/z `-16..15`;
  you do not need a `translate` to bring it to the origin.
- **`min_height` and `max_height` are both inclusive**, so
  `min_height: 0, max_height: 8` is nine cells of vertical span. For "at
  most 8 tall", write `max_height: 7`.
- **The height field is normalised, and stacking octaves squashes it
  toward the middle.** Averaging several octaves pulls the sum away from
  its own extremes, so at the default the bottom of the range is a solid
  floor and the top is never reached. **`octaves: 1` is the knob that
  fixes it** — measured over 32×32, `min_height: 0`, `max_height: 7`,
  `frequency: 0.10`, solid cells per layer y0..y7:

  | `octaves` | y0 | y1 | y2 | y3 | y4 | y5 | y6 | y7 |
  |---|---|---|---|---|---|---|---|---|
  | 1 | 1024 | 1009 | 887 | 732 | 521 | 293 | 141 | 2 |
  | 3 (default) | 1024 | 1024 | 1008 | 843 | 515 | 170 | 9 | 0 |

  Reach for more octaves when you want texture on the slopes, and pair it
  with a lower `frequency` so the landform still has room to move.

`builtin.wfc` is deliberately non-backtracking: over-constrained cells
fall back to empty or grass and say so in `notes`.

## Pipeline graphs

A graph composes generators and transforms into one pipeline. Prefer it
to hand-placed voxels whenever the shape is *generated* rather than
drawn: it survives as parameters a human can tune, it re-rolls with a
different seed, and you do not have to be right about coordinates the
first time.

`voxelith generators` prints a working `graph_template` beside the
generator list. **Read it before writing a graph** — it is where the node
format is defined, and it is accepted verbatim.

```json
{ "op": "graph",
  "graph": { "nodes": [
    { "id": 0, "kind": "builtin.perlin_terrain", "width": 32, "depth": 32 },
    { "id": 1, "kind": "filter", "input": 0, "predicate": { "y_above": 2 } },
    { "id": 2, "kind": "output", "input": 1 } ] } }
```

One flat object per node. `kind` is either a generator id (a source node,
taking that generator's parameters directly — only the ones you want to
differ) or one of the transforms:

| kind | inputs | what it does |
|---|---|---|
| `translate` | `input` | shifts everything by `dx` / `dy` / `dz` |
| `filter` | `input` | keeps voxels matching `predicate`: `{"y_above": n}`, `{"y_below": n}`, `{"matches_color": [r,g,b,a]}`, `{"inside_box": {"min": […], "max": […]}}` |
| `mask` | `subject`, `mask` | keeps `subject` voxels by what is in the same `(x, z)` column of `mask`: `{"mode": "above_column"}` for "trees only above the terrain" |
| `combine` | `a`, `b` | `{"op": "union" \| "difference" \| "intersect"}` |
| `output` | `input` | exactly one per graph — marks what the pipeline emits |

Ids are yours to choose and must be unique. Do not send `next_id`,
`output_node` or `position` — they are bookkeeping and layout, filled in
for you. Limits: 64 nodes, 8 source nodes, and each source is held to the
same size ceiling a `generate` op is.

The graph is **kept with the project**, so whoever opens the `.vxlt`
afterwards finds it in the editor's Graph panel with the sliders live.
That is the reason to send one.

To change a graph rather than resend it, read the current one back from
`--describe` (it comes back whole) and edit it:

```json
{ "op": "graph_edit", "edits": [
    { "edit": "set_params", "id": 0, "params": { "seed": 99 } },
    { "edit": "add_node", "node": { "id": 3, "kind": "translate", "dy": 8 } },
    { "edit": "connect", "target": 3, "slot": 0, "source": 1 } ] }
```

Six edits: `add_node`, `remove_node`, `set_params`, `connect`,
`disconnect`, `clear`. They run in order against a copy — if one fails,
none of them happened, and the message says which. A `connect` that would
close a cycle is refused.

Both ops evaluate the graph and write the result by default. Pass
`"apply": false` to change the graph without writing anything, which is
how you build one up over several batches. Running twice writes twice —
`undo` if you meant to replace.

## Reading the result

```jsonc
{ "ok": true,
  "report": { "version": 1, "dry_run": false, "applied_ops": 3,
              "changed_voxels": 1234,          // writes that changed something
              "voxel_count": 5678,             // solid voxels in the world after
              "world_aabb": { "min": [...], "max": [...] },
              "selection": null,
              "notes": ["op[3] builtin.wfc: 2 cells fell back to empty"] },
  "description": { /* --describe */ },
  "slice": [ /* --slice, one string per row */ ],
  "saved": "hut.vxlt",
  "exported": { "path": "hut.glb", "format": "glb", "vertices": 900,
                "triangles": 600, "bytes": 41232, "notes": [] } }
```

`notes` is where a generator reports a degraded result — read it.

## describe

`--describe` adds voxel and chunk counts, the AABB and its size, the most
common colours, emissive / metallic / tint-zone tallies, sockets, the
selection, undo depth, the document's pipeline graph, and a `structure`
block. `structure` is `null` on a document over two million voxels, which
is too big to measure cheaply.

```jsonc
"structure": {
  "components": 2,              // connected parts, faces only (a corner touch is not a join)
  "largest_component": 1204,
  "loose_parts": [ { "voxels": 8, "aabb": { "min": [...], "max": [...] } } ],
  "floating_components": 1,     // parts that never reach the lowest layer
  "enclosed": 340,              // SOLID voxels fully surrounded — what `hollow` removes
  "footprint": 42,              // solid cells on the lowest layer — what touches the ground
  "cavities": { "count": 1, "voxels": 96 },   // AIR the model seals in completely
  "symmetry": [ { "axis": "x", "mismatched": 12, "ratio": 0.0099 }, … ] }
```

**Three of these get called "hollow" and mean different things.** A solid
cube has one `enclosed` solid and no cavity. Hollow it out and the
readings swap: no enclosed solid, one cavity. Punch a hole through to the
outside and the cavity is gone too — open space, like the gap under an
arch, reaches the surface and is not sealed air at all. Reach for
`enclosed` when you mean "wasted interior an export would carry",
`cavities` when you mean "a sealed room, or a bubble hidden in the mesh".

`footprint` separates a span from a wall. An arch and a wall of the same
bounding box agree on component count, floating parts and voxel count;
the arch stands on two piers and the wall stands on its whole base. It is
meaningless for a model that does not sit on the ground — a fish's lowest
layer is one cell of tail tip, and reads 1 forever.

**Six-connectivity is stricter than it looks** when you repeat a part
with a rotation between copies. Fan-shaped stair treads 45° apart must be
**wider than their angular step** — 64° of tread for a 45° step — so
consecutive treads share cells in (x, z) one level apart and meet face to
face. Treads that exactly tile the circle touch edge to edge, and edge
contact is not connection: the result reads as a pile of separate pieces.

`symmetry` is measured against the model's own bounding box (`min + max −
p`), not the world origin.

## slice

```jsonc
{ "axis": "y",                       // plane normal; "y" is the top-down view
  "index": 1,                        // coordinate along that axis
  "region": { "min": [...], "max": [...] },   // optional window, ≤128 per side
  "mode": "solid" }                  // or "color" for a letter per colour + legend
```

The first line of the output states the axis ranges and row order, so you
never have to guess which way is up.

## render

```bash
voxelith render hut.vxlt                              # one iso view, 256px, hut-iso.png
voxelith render hut.vxlt --view front --out door.png  # one view, named file
voxelith render hut.vxlt --view all --size 384        # hut-iso.png, hut-front.png, …
```

Seven viewpoints: `iso` (default — shows all three dimensions at once)
plus `front`, `back`, `left`, `right`, `top`, `bottom`, or `all`.
**`front` looks along −Z**, so the camera sits on the +Z side and you see
the model's +Z face. Projections are orthographic, so a straight wall
looks straight and equal cells stay equal size. The light follows the
camera, so `back` and `bottom` are lit as well as `front` is.

```jsonc
{ "ok": true,
  "views": [ { "view": "iso", "path": "hut-iso.png", "size": 256, "bytes": 11841,
               "framing": { "bounds": [[-8,0,-9],[8,8,8]],  // inclusive cells
                            "cells_per_pixel": 0.099,
                            "right": [0.707, 0, -0.707],    // world axes, in-image
                            "up": [-0.408, 0.816, -0.408],
                            "forward": [-0.577, -0.577, -0.577] },
               "empty": false } ] }
```

`cells_per_pixel` is what makes a picture actionable: measure an error in
pixels, multiply, and you have it in cells.

Two ways a picture can be background without your model being gone:
`empty: true` means the project held no voxels. `truncated: true` means
the ray walk gave up after a fixed number of steps, so a scene too big to
cross along the view direction drew as nothing — the model is almost
certainly built at coordinates far from where you meant, so check
`world_aabb`.

A size outside `1..=1024` is refused (`invalid_size`), not quietly
resized.

## When it refuses

```jsonc
{ "ok": false, "error": { "code": "region_too_large", "op_index": 2,
                          "message": "region 200×200×200 is 8000000 cells; …" } }
```

`op_index` points at the op to fix. Nothing was written — fix that one op
and resend the batch.

That holds for the whole run, not just the batch: everything about an
export that can be checked without writing is checked first, and the two
writes then run `--export` before `--out`. A run that fails leaves the
project file as it was, so re-sending is safe even in the `--in x --out x`
loop.

| Code | Meaning |
|---|---|
| `unsupported_version` | `version` isn't 1 |
| `invalid_argument` | a field's value isn't usable (bad `quarters`, alpha ≠ 255, `tint_zone` > 3, …) |
| `too_many_ops` / `too_many_voxels` | over 256 ops, or over 4096 voxels in one `set_voxels` |
| `region_too_large` | one op is too big |
| `cell_budget_exceeded` | the batch touches over 8,388,608 cells |
| `coordinate_out_of_range` | a coordinate is outside ±1,048,576 |
| `world_too_large` | the batch would allocate over 2048 new chunks — scattered writes are expensive |
| `no_selection` | a transform got neither `region` nor a selection |
| `unknown_generator` / `invalid_params` | the message lists what's valid |
| `invalid_graph` | a duplicate id, a wire to a node that isn't there, a cycle, or no single `output` |
| `graph_too_large` | over 64 nodes, over 8 source nodes, or sources covering more cells than one batch may |
| `generator_failed` | the generator ran and refused its own parameters |
| `slice_too_large` | narrow it with `region` |
| `unknown_view` / `no_views` / `invalid_size` | `render` arguments; the message lists the views |
| `invalid_slice` | checked before the batch runs, so nothing is lost |
| `ops_unreadable` / `invalid_ops_json` / `input_unreadable` | the files, before any work started |
| `dry_run_with_output` | a dry run writes nothing, so it can't be combined with `--out` / `--export` |
| `save_failed` / `export_failed` / `unsupported_export_format` | the write at the end |

## Batch export: bake

`exec --export` is the interactive File ▸ Export, headless: greedy mesh,
model space untouched. `voxelith bake <spec.json>` is the other export
path — the one that places an asset for an engine and handles many files
at once.

```jsonc
{
  "defaults": {                  // every field optional; items override field by field
    "mesher": "greedy",          // only 'greedy'; Marching Cubes is `smoothing`
    "smoothing": "none",         // none | light | heavy
    "pivot": "origin",           // origin | base-center | feet | center
    "up_axis": "y",              // y | z
    "unit_scale": 1.0,           // finite and positive
    "optimize": "none",          // none | meshopt (shells out to gltfpack)
    "optimize_args": null        // explicit gltfpack args, replacing the safe defaults
  },
  "items": [
    { "src": "models/hut.vxlt", "out": "dist/hut.glb", "pivot": "feet" },
    { "srcDir": "models/props", "outDir": "dist/props" }   // one item per .vxlt found
  ]
}
```

The placement transform is a single lossless **root node** — vertex data
is untouched, and an identity transform is byte-identical to a plain
export.

- **One bad model never sinks the batch.** A per-item failure is recorded
  in that item's report; only a spec-level problem (unreadable spec, bad
  `--shard`, unreadable `srcDir`) aborts before anything runs.
- **Exit codes**: 0 all good, 1 some item failed, 2 the spec itself was
  unusable.
- A `<out>.report.json` lands beside each output (`farm.glb` →
  `farm.report.json`).
- `optimize: "meshopt"` needs `gltfpack` on PATH and degrades to a note
  if it is missing. The default arguments deliberately skip quantization,
  which would corrupt the integer tint zone carried in `TEXCOORD_0.x`.
- `--shard i/n` processes every n-th item, for CI fan-out.
- Unknown keys are ignored for forward compatibility but **reported** —
  a typo'd `"smoothng"` produces an asset that isn't what you asked for,
  so read the warnings.

## Over MCP

`voxelith mcp` serves the same primitives as a tool set — stdio by
default, Streamable HTTP at `/mcp` with `--http`. **Eleven tools**:
`new_project`, `open_project`, `save_project`, `apply_ops`,
`list_generators`, `describe`, `slice`, `render_views`, `undo`, `redo`,
`export`.

The ops format is identical, and the tool schema is generated from the
same types, so a client sees the whole op union without reading this
page.

- **The document is resident.** One session stays open across calls, so
  undo reaches back through earlier batches and the selection persists.
  **Call `new_project` between unrelated models** or the last one is
  still there.
- **`apply_ops` answers with the report *and* a description.** Under
  `dry_run` both come from the preview.
- **`render_views` hands back images, not paths.**
  `{"views": ["front", "top"], "size": 256}`; omit both for one 256px
  isometric view.

Every path resolves inside the server's root (`--root`, default the
working directory); anything outside comes back as `path_refused`.

### Letting a human watch

`voxelith mcp --checkpoint` writes the document back to its own file
after every edit that changed it, so somebody with that `.vxlt` open in
the editor sees each step land. `apply_ops`, `undo` and `redo` then carry
a `checkpoint` field:

```json
{ "ok": true, "voxel_count": 195, "checkpoint": { "saved": true } }
```

The field is absent without the flag, and a dry run never reports one.
Before the document has a file, `saved` is `false` with a `detail` saying
to call `save_project` once. A failed write is also `saved: false` with a
reason and the call still succeeds — the edit is in the session either
way; what it means is that the human is now looking at a stale world.

**One writer at a time.** If the person at the editor has unsaved edits,
their copy wins and the reload is refused. Nothing merges the two.

## The in-editor bridge

The checkpoint path passes a file back and forth. The editor also hosts
an MCP server of its own with no file in the middle: you edit the world
the human is looking at.

```
voxelith --agent-port 8737          # or: Agent panel ▸ Start
claude mcp add --transport http voxelith http://127.0.0.1:8737/mcp
```

Loopback only. **Seven tools** — `apply_ops`, `describe`, `slice`,
`render_views`, `list_generators`, `undo`, `redo`.

- **One undo stack, shared with the person.** Your batch is one entry on
  the same history as their brush strokes. They can Ctrl+Z your step and
  your `undo` can take back theirs — deliberate, but it means
  `undo_depth` moves when they work.
- **No file tools.** Someone is sitting at this document; where it saves
  is theirs to decide. Ask them rather than looking for a tool.
- **`apply_ops` may wait, and may be refused.** The editor can be set to
  ask its user first: your batch goes up in their viewport as translucent
  geometry and the call waits. `"review": "accepted"` means yes, `"auto"`
  means the editor was not asking. A no comes back as an error with code
  `rejected` — ask what they want different instead of resending as-is.
- **Refusals specific to this server**: `review_pending` (a previous
  batch is still waiting), `world_changed` (they edited while yours
  waited — describe the current world and send it again),
  `editor_unavailable` (the editor closed or the bridge was switched
  off).
- Every answer carries `unsaved_changes`. If true, the human has work in
  front of them that is not on disk — worth knowing before suggesting
  they close or reload anything.
- `render_views` here is still a CPU render of the voxels, not a
  screenshot: it ignores where the human has pointed their camera. Sizes
  stop at 512, because these renders run on the editor's frame loop.
