# Eval cases

Each file in `cases/` is one modeling task plus the properties its
result has to have. They exist to answer a question a screenshot can't:
**is the thing an agent just built actually correct?**

Two tiers, graded the same way:

- **`cases/`** — six everyday objects. A sword, a crate, a chair. This
  tier answers "is the interface usable at all"; a capable agent with
  the docs in front of it clears all six.
- **`advanced/`** — five that push on what a cube grid is worst at:
  overhangs and curves, repetition-with-rotation, two-axis symmetry at
  a scale that forces hollowing, a height range one generator cannot
  reach, and a five-part assembly.

```bash
# grade one finished project against one case
cargo run --release -- eval evals/cases/connected-sword.json --project sword.vxlt

# grade a whole run: one <case-id>.vxlt per case in a directory
cargo run --release -- eval evals/cases --results run-2026-08-08/
cargo run --release -- eval evals/advanced --results run-2026-08-08/
```

The report is JSON on stdout, one entry per assertion with what was
expected and what was measured. The exit code is zero only when every
case passed, so a suite gates a run the way a test command does.

## What is being graded

The properties come from `describe` — connected components, floating
parts, enclosed interior, per-axis symmetry, bounding size, voxel count,
and the size of the project's pipeline graph. Grading is arithmetic on
those numbers. There is no model in the loop: whether two halves of a
sword are joined is not a matter of opinion, and a judge with opinions
is exactly what this layer exists to replace.

An agent can run the same command on its own work. That is the point of
grading on the numbers it already reads while building — "check your own
work" and "did it pass" can't drift apart.

## Writing a case

```jsonc
{
  "id": "connected-sword",          // also the result's file name in suite mode
  "task": "Build a sword ...",      // handed to the agent verbatim
  "notes": "Why this case exists",  // not graded
  "expect": {
    "components": { "max": 1 },     // every field optional, min/max both optional
    "floating_components": { "max": 0 },
    "symmetry": [{ "axis": "x", "max_ratio": 0.02 }],
    "size": [{ "min": 2, "max": 6 }, { "min": 16, "max": 26 }, { "min": 1, "max": 5 }],
    "voxel_count": { "min": 60 },
    "enclosed": { "max": 0 },       // no wasted solid interior
    "cavities": { "max": 0 },       // no sealed air pocket
    "footprint": { "max": 60 },     // how much touches the ground
    "emissive": { "min": 1 },
    "graph_nodes": { "min": 2 }
  }
}
```

### Picking the right "hollow" bar

Three readings get called hollow and mean different things. Choosing
wrong fails correct work — it has already happened twice here:

| | solid 3³ | hollow shell | shell with a hole |
|---|---|---|---|
| `enclosed` (solid, no exposed face) | 1 | 0 | 0 |
| `cavities` (air sealed in) | 0 | 1 | 0 |

`enclosed` is "interior an export would carry for nothing".
`cavities` is "a sealed room, or a bubble inside the mesh". Open space
— the gap under an arch, a doorway — is **neither**: it reaches the
outside, so no metric of enclosure sees it. What catches an arch is
`footprint`, the solid cells on the lowest layer: an arch stands on two
piers, a wall stands on its whole base, and every other reading agrees
between them.

Two rules keep a case honest:

**Assert only what the task asked for.** A case that grades symmetry
without requesting it is measuring whether the model guessed the
grader's mind. If you add a bar, add the sentence that asks for it.

This is the rule that has actually been broken here, twice, both times
in the same shape: a standard the author held in mind and never wrote
into the task. `arched-bridge` graded interior hollowness that its task
never mentioned, and failed a correct stone arch. `castle-keep` said
"no solid fill inside" and then graded every voxel at a wall junction,
which is not what anyone reads that phrase to mean. When a case fails,
read the task text first and ask whether it really demanded the thing
that failed.

**The numbers are measurements, not verdicts.** `tree-with-canopy` is
deliberately silent about floating parts, because clumps of leaves that
don't touch are a tree — the same reading that fails `standing-chair`.
A grader that hard-coded "one component, nothing floating" would mark a
correct tree wrong. The case decides what its own numbers mean.

`graph_nodes` is the one bar that grades *method* rather than result: it
asks whether the model produced something that stays adjustable, and it
is checkable only because the pipeline graph travels with the project. A
hand-placed model passes every size and count bar and fails this one.

## Running the agent

Not this tool's job. Driving a model, sampling it k times, pinning its
temperature — that belongs to whoever is measuring, and it changes with
the model. What lives here is the part that shouldn't: the bar.
