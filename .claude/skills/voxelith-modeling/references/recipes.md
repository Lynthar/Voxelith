# Building things that come out right

Technique, not format — `ops.md` has the format. Everything here was paid
for in retries: these are the places where the obvious first attempt is
wrong, and what to do instead.

- [Symmetry](#symmetry)
- [Shells and hollowing](#shells-and-hollowing)
- [Curves, arches and domes](#curves-arches-and-domes)
- [Tapers and organic shapes](#tapers-and-organic-shapes)
- [Repetition with rotation](#repetition-with-rotation)
- [Terrain with real relief](#terrain-with-real-relief)
- [Building in layers](#building-in-layers)
- [Which way is the front](#which-way-is-the-front)
- [Knowing when it's done](#knowing-when-its-done)
- [A worked example](#a-worked-example)

## Symmetry

**Construct it; don't repair it.** `symmetry.mismatched` counts cells
that have no partner across the model's own bounding-box mid-plane. If
you build freely and fix afterwards, you are hunting single cells. Three
ways to make the count zero by construction, in order of preference:

1. **Use odd spans.** A part spanning x −7..7 has its mirror plane
   through the centre of column 0, so every cell has a partner. Choose
   the span before you choose anything else and the problem never
   appears.
2. **Issue ops in mirrored pairs.** Every `box` at `[x0, x1]` matched by
   one at `[−x1, −x0]`. A tiny helper that emits both from one call keeps
   the batch readable.
3. **Build one half and `mirror_copy` it.** The default `plane` already
   sits flush past the region's `max`, which is the seam you want, so no
   `plane` argument is needed.

**An even span's mirror plane falls between cells.** A model spanning x
0..23 mirrors about the seam between 11 and 12, so no single column sits
on the axis. One cylinder centred anywhere is off by a cell whichever way
you round it — use **two cylinders of equal radius at x = 11 and x = 12**
and union them. This is the usual reason a "centred" tower reports
mismatched cells.

Symmetry is measured against the model's **own bounding box**, not the
world origin, so a model built off-centre is still judged fairly. But it
also means adding one asymmetric part moves the bounding box and can put
everything else off-axis at once.

## Shells and hollowing

**If the model should end up hollow, build it hollow.** `box` and
`cylinder` take `filled: false`. Do that from the start and the closing
`hollow` op is a harmless no-op instead of demolition.

**`hollow` clears any cell whose six neighbours are all solid**, which is
exactly right for a box-like shell of even wall thickness and
**destructive on anything that tapers**. A spire built from stacked solid
discs (r = 4, 3, 2, 1) has middle layers whose neighbours are all solid;
hollowing clears them and leaves a stack of rings that touch nothing. One
real case went from 1 connected component to 61, with 60 floating — and
it looked fine in a render until the numbers were read.

A tapered top is better built as a **solid cap with battlements around
it**: no enclosed interior to begin with, so nothing to hollow.

Run `describe` after hollowing anything that is not a shell. `components`
tells you immediately.

Watch which reading you actually want:

| You mean | Read |
|---|---|
| wasted interior an export would carry | `enclosed` |
| a sealed room, or a bubble hidden in the mesh | `cavities` |
| the gap under an arch | neither — it reaches the outside |

## Curves, arches and domes

**A cell is included when its centre is within `radius + 0.5`.** So
`radius: 1` is a solid 3×3, and a `radius: 3` arch has only three courses
above its springing line (7 / 7 / 5 / 3 wide).

Size a curve by the **span you want, then add one**. Estimating from `r²`
gives you an arch about half as tall as you meant — the most common cause
of "the arch looks like a chamfered square hole".

For an arch, two cylinders make a cleaner opening than a sphere: place
the axis along the span, pick the radius from the half-width plus one,
and put the springing line where the piers stop. Cut the opening with a
second cylinder of `"voxel": "air"` rather than trying to draw the curve
in solid.

## Tapers and organic shapes

**Stack segmented cylinders with changing radius.** A fish body is
`cylinder axis: "x"` in segments of radius 1 → 2 → 3 → 4 → 3 → 2 → 1.
**Denser segments read much better**: one cell per segment beats four.

**Cylinders only give round cross-sections.** For a flattened body, carve
the outer Z with two `air` boxes after building — that is what makes a
fish look like a fish instead of a log.

**Fins, wings and leaves want to be one cell thick.** A single-cell sheet
reads as a fin; a rectangular slab reads as a slab and passes every
structural check anyway. Open a fan tail as a **staircase of columns**
widening outward, with a notch in the middle for a fork.

For organic work the **render is the primary judge**. The structural
numbers answer "is it assembled correctly", never "does it look like the
thing". A fish with rectangular slab fins scores a clean pass on every
bar there is.

## Repetition with rotation

Six-connectivity is stricter than it looks. Two parts joined only along
an edge or a corner are **separate components**.

**Spiral stairs**: make each tread a fan whose **angular width exceeds
the angular step**, so consecutive treads share cells in (x, z) one level
apart and meet face to face. Treads that exactly tile the circle touch
edge to edge and the result reads as a pile of loose pieces.

**Specify that overlap in cells of arc, not in degrees.** A fixed extra
angle buys arc in proportion to radius, so it is generous at the rim and
close to nothing at the newel — which is exactly where every tread also
has to grip the column. Pick an overlap of one to two cells and solve for
the half-width at each radius; the wedge comes out crisp outside and
collared near the middle. Two independent builds arrived at this
separately, one widening only the innermost ring, the other holding the
arc constant at about 1.7 cells (42° of half-width at the newel, 26° at
the rim).

Whatever you choose, **check connectivity before you commit the batch —
with a dry run, not with your own flood fill**:

```bash
voxelith exec stairs.json --dry-run --describe
```

That reports `components`, `loose_parts` and `floating_components` for
the world the batch would produce, and writes nothing. Computing a shape
like this in a script is reasonable; re-implementing the connectivity
test beside it is not, and the built-in answer is the one the eval cases
grade against.

**One thing `components: 1` does not prove**: that every part touches the
part you meant. A chain of treads holding on to each other and gripping
the newel once reads exactly like twenty treads each gripping it. When
the task names a specific contact, check that contact directly — the
global count cannot see it.

The same rule governs any ring of rotated copies: overlap adjacent
copies, or add a hub that every copy penetrates.

## Terrain with real relief

`builtin.perlin_terrain` **centres itself** — `width`/`depth` of 32 lands
on x/z −16..15, no `translate` needed.

**Turn `octaves` down to 1 before you touch anything else.** Stacking
octaves averages them, which pulls the height field toward the middle of
its range: at the default the bottom layers come out as a fully solid
floor and the top layer is never reached at all, so the terrain reads as
a flat terraced plate. Measured over 32×32 with `min_height: 0`,
`max_height: 7`, `frequency: 0.10` — solid cells per layer y0..y7:

| `octaves` | y0 | y1 | y2 | y3 | y4 | y5 | y6 | y7 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 1009 | 887 | 732 | 521 | 293 | 141 | 2 |
| 3 (default) | 1024 | 1024 | 1008 | 843 | 515 | 170 | 9 | 0 |

Then tune `frequency` for the size of the hills — around 0.10 gives
hills a few cells across at this scale.

**Do not reach for a second noise source to get relief — it does the
opposite.** `combine` with `{"op": "union"}` takes the higher column, and
the maximum of two noise fields is both higher and less varied than
either one, so unioning two terrains raises the floor and flattens what
you were trying to exaggerate. Measured on 32×32, solid cells per layer:

| | y0 | y1 | y2 | y3 | y4 | y5 | y6 | y7 |
|---|---|---|---|---|---|---|---|---|
| one source | 1024 | 1024 | 1011 | 931 | 737 | 439 | 197 | 50 |
| union of two | 1024 | 1024 | 1011 | 972 | 891 | 700 | 412 | 87 |

Every middle layer fills in: the low ground rises and the patch reads as
a plateau. A low-amplitude "detail" source is even less use — it can only
raise ground the base already buried, measured at 4 and 33 voxels of
change. `intersect` fails the other way, taking the minimum and shaving
the top layer off entirely.

Union is for combining **different things** — terrain with a rock, a
trunk with its canopy — not two of the same thing. For terrain, one
tuned source is the answer.

Spend the effort on `frequency` and `seed` instead, and pick them by
rendering rather than by reasoning — at 32×32 a frequency near 0.09
reads as a maze of small ridges and 0.04 as a plateau, with the
interesting range in between. That is art direction, and the render is
the only judge of it.

Remember `min_height` and `max_height` are both inclusive: for "at most 8
tall", write `max_height: 7`.

**Send terrain as a graph, not as placed voxels.** It is generated rather
than drawn, so the human who opens the project gets sliders instead of a
result. This is also the one eval bar that grades method — `graph_nodes`.

## Building in layers

`write_mode` is what keeps a second pass from destroying the first:

- **`only_air`** — build the trunk, then pack the canopy around it. The
  leaves fill in beside the trunk instead of overwriting it.
- **`only_solid`** — add colour and detail late. It cannot change any
  structural measurement, so a recolouring pass cannot break a model that
  was already passing.

Order matters more than it looks: structure first with `replace`, mass
around it with `only_air`, decoration last with `only_solid`.

## Which way is the front

**`render --view front` looks along −Z**, so the camera sits on the +Z
side and you see the model's **+Z** face. Put a facade, a face, or a door
on +Z.

A model that came out backwards takes one `mirror` on `z` over its whole
bounding box. That leaves X symmetry untouched, so nothing else has to be
redone.

## Knowing when it's done

Decide what the numbers should be for **this** object before you read
them — they are measurements, not verdicts:

| Reading | Wrong for | Right for |
|---|---|---|
| `components: 2` | a sword | a pair of boots |
| a floating part | a chair | a tree canopy |
| `enclosed > 0` | anything you will export | a model still being built |
| `footprint: 1` | a building | a fish (its lowest layer is one cell of tail) |

Check in this order: `notes` → `components` and `floating_components` →
`slice` for anything you need to count → `render --view iso` to see it.

And grade yourself when a case fits:

```bash
voxelith eval evals/cases/connected-sword.json --project sword.vxlt
```

Reading the cases is useful even without running one — they state what
"finished" means for a kind of object, in the same numbers you build
with.

## A worked example

A castle keep, 15 ops, one pass, every bar cleared. What made it work:

- **Everything a shell from the start.** Main tower `box filled: false`,
  corner turrets `cylinder filled: false`. The closing `hollow` was
  therefore a no-op rather than demolition.
- **Every span odd.** The tower spans x/z −7..7 and the turret centres
  sit exactly on ±7, so both mirror planes fall through cell centres and
  `mismatched` was zero without a single correction.
- **Flat roof with battlements, not a cone.** A tapered solid top is
  precisely the shape `hollow` destroys, and battlements have no interior
  to hollow anyway.
- **`describe` read afterwards, not assumed.** It reported 20 cavity
  cells in the turret shafts — not required by the task, so not a
  problem, but worth knowing before exporting.

The general shape of that: choose spans and wall style **before** the
first op, so the properties you will be judged on are consequences of the
construction rather than repairs on top of it.
