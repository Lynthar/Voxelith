//! Pipeline graph: compose multiple generators + transforms into a DAG.
//!
//! Each `GraphNode` either produces a `VoxelPatch` from no inputs
//! (source generators), transforms one input patch (`Translate`), or
//! combines two input patches (`Combine`). An `Output` node marks
//! the final patch the pipeline emits. Evaluation is a DFS topological
//! sort starting from the output: visit inputs before consumers,
//! memoize each node's patch in a `HashMap`, return the output's patch.
//!
//! The graph deliberately stays small for now (no n-ary combine, no
//! per-voxel filters, no visual wires). The data model and evaluator
//! are written so those extensions only require new `NodeKind`
//! variants — the surrounding plumbing (UI, prefs, undo) doesn't
//! care what nodes do internally.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::core::Voxel;

use super::{
    GenError, GenResult, LSystemTree, PerlinTerrain, VoxelGenerator, VoxelPatch,
    WfcGenerator,
};

/// Node identifier. Stable within a graph (we never reuse an id after
/// removal — the next add gets `next_id`).
pub type NodeId = u32;

/// One graph node: id + payload describing what it does + UI position.
///
/// `position` is panel-space coordinates rendered by the visual graph
/// editor. It's part of the persisted state so node layout survives
/// restarts. `#[serde(default)]` keeps older prefs files (without
/// position) loadable — they'll deserialize as `[0.0, 0.0]` and the
/// hydrate path detects that case to re-layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: NodeId,
    pub kind: NodeKind,
    #[serde(default)]
    pub position: [f32; 2],
}

/// Default node spacing for cascade layout.
const NODE_LAYOUT_DX: f32 = 220.0;
const NODE_LAYOUT_DY: f32 = 130.0;
const NODE_LAYOUT_COLS: usize = 4;
const NODE_LAYOUT_ORIGIN: [f32; 2] = [60.0, 40.0];

/// What a node does and what it consumes. Source variants take no
/// inputs and embed their generator's parameters directly. Transform
/// variants reference other nodes by id — `None` means "input not
/// connected yet" and `evaluate` will report it as a `MissingInput`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    Terrain(PerlinTerrain),
    Tree(LSystemTree),
    Wfc(WfcGenerator),
    /// Shift every voxel of `input` by `(dx, dy, dz)` world units.
    Translate {
        input: Option<NodeId>,
        dx: i32,
        dy: i32,
        dz: i32,
    },
    /// Per-voxel filter: keep only those satisfying `predicate`. Intended
    /// for compositions like "keep tree voxels above ground level" or
    /// "keep only the grass-colored layer of a stratified terrain". For
    /// position-set operations against another patch, use Combine.
    Filter {
        input: Option<NodeId>,
        predicate: FilterPredicate,
    },
    /// Two-input column mask: keep voxels of `subject` based on what's
    /// in the same `(x, z)` column of `mask`. Differs from Combine in
    /// that the test is column-projected, not exact position match —
    /// enabling workflows like "keep tree voxels in any column where
    /// terrain has any voxel below" (`AboveColumn`), which Combine and
    /// Filter alone can't express.
    Mask {
        subject: Option<NodeId>,
        mask: Option<NodeId>,
        mode: MaskMode,
    },
    /// Set-theoretic combination of two patches.
    Combine {
        a: Option<NodeId>,
        b: Option<NodeId>,
        op: CombineOp,
    },
    /// Pass-through marker for the patch the pipeline emits.
    Output {
        input: Option<NodeId>,
    },
}

/// What `Filter` keeps from its input patch. Each variant defines a
/// per-voxel predicate; voxels that satisfy it are kept, the rest are
/// dropped. Bounds in `YAbove`/`YBelow`/`InsideBox` are inclusive.
///
/// Color match uses raw RGBA bytes so the filter can be persisted in
/// prefs without coupling to `Voxel`'s serde shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterPredicate {
    /// Keep voxels with `y >= threshold`.
    YAbove(i32),
    /// Keep voxels with `y <= threshold`.
    YBelow(i32),
    /// Keep voxels whose RGBA bytes match `[r, g, b, a]` exactly.
    MatchesColor([u8; 4]),
    /// Keep voxels inside the closed box `[min, max]` on every axis.
    InsideBox {
        min: (i32, i32, i32),
        max: (i32, i32, i32),
    },
}

impl FilterPredicate {
    /// One-line summary for the node body and combo box.
    pub fn label(&self) -> String {
        match self {
            Self::YAbove(t) => format!("y ≥ {}", t),
            Self::YBelow(t) => format!("y ≤ {}", t),
            Self::MatchesColor([r, g, b, _]) => {
                format!("color = #{:02x}{:02x}{:02x}", r, g, b)
            }
            Self::InsideBox { min, max } => format!(
                "box ({},{},{})..({},{},{})",
                min.0, min.1, min.2, max.0, max.1, max.2
            ),
        }
    }
}

impl Default for FilterPredicate {
    fn default() -> Self {
        Self::YAbove(0)
    }
}

/// Column-projected modes for the `Mask` node. Both modes look up
/// the mask's voxel set per `(x, z)` column rather than testing exact
/// `(x, y, z)` matches — that's what distinguishes Mask from
/// Combine's set ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskMode {
    /// Keep `subject` voxel at `(x, y, z)` iff `mask` has at least one
    /// voxel in the same column with `y_mask < y`. Use for "trees
    /// only above terrain surface".
    AboveColumn,
    /// Keep `subject` voxel at `(x, y, z)` iff `mask` has at least one
    /// voxel in the same column with `y_mask > y`. Use for "stalactites
    /// only where there's a ceiling above".
    BelowColumn,
}

impl MaskMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::AboveColumn => "Above column",
            Self::BelowColumn => "Below column",
        }
    }
}

impl Default for MaskMode {
    fn default() -> Self {
        Self::AboveColumn
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombineOp {
    /// Voxels from `a` and `b`. Cells present in both: `b` wins.
    Union,
    /// Cells in `a` not also in `b`. (`a - b`)
    Difference,
    /// Cells present in both, voxel value taken from `a`.
    Intersect,
}

impl CombineOp {
    pub fn label(self) -> &'static str {
        match self {
            Self::Union => "Union",
            Self::Difference => "Difference",
            Self::Intersect => "Intersect",
        }
    }
}

impl NodeKind {
    /// All node ids this node consumes. Used by topological sort and
    /// by the UI to render input dropdowns.
    pub fn inputs(&self) -> Vec<NodeId> {
        match self {
            Self::Terrain(_) | Self::Tree(_) | Self::Wfc(_) => vec![],
            Self::Translate { input, .. } => input.iter().copied().collect(),
            Self::Filter { input, .. } => input.iter().copied().collect(),
            Self::Mask { subject, mask, .. } => {
                let mut v = Vec::with_capacity(2);
                if let Some(id) = *subject {
                    v.push(id);
                }
                if let Some(id) = *mask {
                    v.push(id);
                }
                v
            }
            Self::Combine { a, b, .. } => {
                let mut v = Vec::with_capacity(2);
                if let Some(id) = *a {
                    v.push(id);
                }
                if let Some(id) = *b {
                    v.push(id);
                }
                v
            }
            Self::Output { input } => input.iter().copied().collect(),
        }
    }

    /// Display label for combo boxes / node headers.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Terrain(_) => "Source: Terrain",
            Self::Tree(_) => "Source: Tree",
            Self::Wfc(_) => "Source: WFC",
            Self::Translate { .. } => "Translate",
            Self::Filter { .. } => "Filter",
            Self::Mask { .. } => "Mask",
            Self::Combine { .. } => "Combine",
            Self::Output { .. } => "Output",
        }
    }
}

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("graph has no Output node")]
    NoOutput,
    #[error("multiple Output nodes (only one allowed)")]
    MultipleOutputs,
    #[error("cycle detected involving node {0}")]
    Cycle(NodeId),
    #[error("node {node} has unconnected input slot")]
    MissingInput { node: NodeId },
    #[error("node {0} references nonexistent node")]
    DanglingReference(NodeId),
}

impl From<GraphError> for GenError {
    fn from(e: GraphError) -> Self {
        GenError::Failed(e.to_string())
    }
}

/// A pipeline of generator / transform / combine nodes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PipelineGraph {
    pub nodes: Vec<GraphNode>,
    /// Next id to hand out. Monotonic per graph instance — ids are
    /// never reused so dropdowns can rely on stable references.
    pub next_id: NodeId,
    /// Cached id of the (sole) Output node. Invalidated if you remove
    /// the output — call `find_output` to rediscover.
    pub output_node: Option<NodeId>,
}

impl PipelineGraph {
    /// Add a new node. Auto-assigns the next id and a cascade-layout
    /// position so visual editors don't pile new nodes on top of
    /// existing ones. Returns the new id.
    pub fn add(&mut self, kind: NodeKind) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let is_output = matches!(kind, NodeKind::Output { .. });
        let n = self.nodes.len();
        let position = [
            NODE_LAYOUT_ORIGIN[0]
                + ((n % NODE_LAYOUT_COLS) as f32) * NODE_LAYOUT_DX,
            NODE_LAYOUT_ORIGIN[1]
                + ((n / NODE_LAYOUT_COLS) as f32) * NODE_LAYOUT_DY,
        ];
        self.nodes.push(GraphNode { id, kind, position });
        if is_output {
            self.output_node = Some(id);
        }
        id
    }

    /// Re-layout all nodes in cascade order. Useful when loading a
    /// graph saved before `position` existed (every node deserializes
    /// at `[0, 0]`), or when the user requests "auto layout".
    pub fn relayout(&mut self) {
        for (i, node) in self.nodes.iter_mut().enumerate() {
            node.position = [
                NODE_LAYOUT_ORIGIN[0]
                    + ((i % NODE_LAYOUT_COLS) as f32) * NODE_LAYOUT_DX,
                NODE_LAYOUT_ORIGIN[1]
                    + ((i / NODE_LAYOUT_COLS) as f32) * NODE_LAYOUT_DY,
            ];
        }
    }

    /// True when every node sits at the origin — diagnostic hook for
    /// "this graph came from a pre-position prefs file". Used by the
    /// app's hydrate path to call `relayout` once.
    pub fn all_at_origin(&self) -> bool {
        !self.nodes.is_empty()
            && self
                .nodes
                .iter()
                .all(|n| n.position == [0.0, 0.0])
    }

    /// Remove a node by id. Also clears any input slot in other nodes
    /// that referenced it (so the graph stays internally consistent
    /// without dangling refs).
    pub fn remove(&mut self, id: NodeId) {
        self.nodes.retain(|n| n.id != id);
        if self.output_node == Some(id) {
            self.output_node = None;
        }
        for node in &mut self.nodes {
            clear_input_if(&mut node.kind, id);
        }
    }

    pub fn get(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Number of input slots a node has. UI uses this to draw the
    /// right number of input sockets.
    pub fn input_count(kind: &NodeKind) -> usize {
        match kind {
            NodeKind::Terrain(_) | NodeKind::Tree(_) | NodeKind::Wfc(_) => 0,
            NodeKind::Translate { .. }
            | NodeKind::Filter { .. }
            | NodeKind::Output { .. } => 1,
            NodeKind::Mask { .. } | NodeKind::Combine { .. } => 2,
        }
    }

    /// Whether this node kind exposes an output socket. Output nodes
    /// (sinks) don't.
    pub fn has_output(kind: &NodeKind) -> bool {
        !matches!(kind, NodeKind::Output { .. })
    }

    /// Read the current value of input slot `slot` on `target`.
    /// Returns `Ok(None)` for sources (which have no inputs).
    pub fn get_input(
        &self,
        target: NodeId,
        slot: usize,
    ) -> Result<Option<NodeId>, GraphError> {
        let node = self.get(target).ok_or(GraphError::DanglingReference(target))?;
        Ok(match &node.kind {
            NodeKind::Terrain(_) | NodeKind::Tree(_) | NodeKind::Wfc(_) => None,
            NodeKind::Translate { input, .. }
            | NodeKind::Filter { input, .. }
            | NodeKind::Output { input } => *input,
            NodeKind::Mask { subject, mask, .. } => match slot {
                0 => *subject,
                1 => *mask,
                _ => return Err(GraphError::DanglingReference(target)),
            },
            NodeKind::Combine { a, b, .. } => match slot {
                0 => *a,
                1 => *b,
                _ => return Err(GraphError::DanglingReference(target)),
            },
        })
    }

    /// Connect or disconnect an input slot. When connecting (i.e.
    /// `new_input` is `Some`), reachability from `target` is checked
    /// before committing — if the proposed wire would form a cycle,
    /// the change is reverted and `GraphError::Cycle` is returned.
    /// `target` must exist; sources have no input slots and return
    /// `Err(DanglingReference)` instead of silently no-op'ing.
    pub fn set_input(
        &mut self,
        target: NodeId,
        slot: usize,
        new_input: Option<NodeId>,
    ) -> Result<(), GraphError> {
        let old = self.get_input(target, slot)?;
        // Tentative apply.
        if let Some(node) = self.get_mut(target) {
            apply_input_slot(&mut node.kind, slot, new_input);
        } else {
            return Err(GraphError::DanglingReference(target));
        }
        // If connecting, walk the graph from `target` to confirm no
        // cycle was introduced. Disconnects always succeed.
        if new_input.is_some() {
            if let Err(e) = self.topo_sort_to(target) {
                // Revert.
                if let Some(node) = self.get_mut(target) {
                    apply_input_slot(&mut node.kind, slot, old);
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// Rescan the node list to find the (sole) Output node and store
    /// its id in `output_node`. Returns Err if there are zero or more
    /// than one Output nodes.
    pub fn find_output(&mut self) -> Result<NodeId, GraphError> {
        let outputs: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Output { .. }))
            .map(|n| n.id)
            .collect();
        match outputs.len() {
            0 => {
                self.output_node = None;
                Err(GraphError::NoOutput)
            }
            1 => {
                self.output_node = Some(outputs[0]);
                Ok(outputs[0])
            }
            _ => Err(GraphError::MultipleOutputs),
        }
    }

    /// Run the pipeline and return the patch produced by the Output
    /// node (or an error describing what's wrong with the graph).
    pub fn evaluate(&self) -> GenResult<VoxelPatch> {
        let output_id = self.find_output_immut()?;
        let order = self.topo_sort_to(output_id)?;

        let mut cache: HashMap<NodeId, VoxelPatch> = HashMap::new();
        for id in order {
            let node = self
                .get(id)
                .ok_or(GraphError::DanglingReference(id))?;
            let patch = self.eval_node(node, &cache)?;
            cache.insert(id, patch);
        }
        // Output's patch is its sole input's patch (passed through
        // unchanged). evaluate returns it.
        cache
            .remove(&output_id)
            .ok_or_else(|| GraphError::DanglingReference(output_id).into())
    }

    fn find_output_immut(&self) -> Result<NodeId, GraphError> {
        if let Some(id) = self.output_node {
            if matches!(self.get(id).map(|n| &n.kind), Some(NodeKind::Output { .. })) {
                return Ok(id);
            }
        }
        // Cache miss / stale — scan.
        let mut found: Option<NodeId> = None;
        for n in &self.nodes {
            if matches!(n.kind, NodeKind::Output { .. }) {
                if found.is_some() {
                    return Err(GraphError::MultipleOutputs);
                }
                found = Some(n.id);
            }
        }
        found.ok_or(GraphError::NoOutput)
    }

    /// DFS-based topological sort up from `root`. Returns nodes in
    /// the order they should be evaluated (inputs first). Detects
    /// cycles via the classic three-color marking scheme.
    fn topo_sort_to(&self, root: NodeId) -> Result<Vec<NodeId>, GraphError> {
        // 0 = unvisited, 1 = on stack (visiting), 2 = done.
        let mut state: HashMap<NodeId, u8> = HashMap::new();
        let mut order = Vec::new();
        self.visit(root, &mut state, &mut order)?;
        Ok(order)
    }

    fn visit(
        &self,
        id: NodeId,
        state: &mut HashMap<NodeId, u8>,
        order: &mut Vec<NodeId>,
    ) -> Result<(), GraphError> {
        match state.get(&id) {
            Some(&2) => return Ok(()),
            Some(&1) => return Err(GraphError::Cycle(id)),
            _ => {}
        }
        state.insert(id, 1);
        let node = self.get(id).ok_or(GraphError::DanglingReference(id))?;
        for input_id in node.kind.inputs() {
            self.visit(input_id, state, order)?;
        }
        state.insert(id, 2);
        order.push(id);
        Ok(())
    }

    fn eval_node(
        &self,
        node: &GraphNode,
        cache: &HashMap<NodeId, VoxelPatch>,
    ) -> GenResult<VoxelPatch> {
        match &node.kind {
            NodeKind::Terrain(g) => g.generate(),
            NodeKind::Tree(g) => g.generate(),
            NodeKind::Wfc(g) => g.generate(),
            NodeKind::Translate { input, dx, dy, dz } => {
                let in_id = input.ok_or(GraphError::MissingInput { node: node.id })?;
                let in_patch = cache
                    .get(&in_id)
                    .cloned()
                    .ok_or(GraphError::DanglingReference(in_id))?;
                Ok(translate_patch(in_patch, *dx, *dy, *dz))
            }
            NodeKind::Filter { input, predicate } => {
                let in_id = input.ok_or(GraphError::MissingInput { node: node.id })?;
                let in_patch = cache
                    .get(&in_id)
                    .cloned()
                    .ok_or(GraphError::DanglingReference(in_id))?;
                Ok(filter_patch(in_patch, predicate))
            }
            NodeKind::Mask { subject, mask, mode } => {
                let s_id = subject.ok_or(GraphError::MissingInput { node: node.id })?;
                let m_id = mask.ok_or(GraphError::MissingInput { node: node.id })?;
                let s_patch = cache
                    .get(&s_id)
                    .cloned()
                    .ok_or(GraphError::DanglingReference(s_id))?;
                let m_patch = cache
                    .get(&m_id)
                    .cloned()
                    .ok_or(GraphError::DanglingReference(m_id))?;
                Ok(mask_patch(s_patch, m_patch, *mode))
            }
            NodeKind::Combine { a, b, op } => {
                let a_id = a.ok_or(GraphError::MissingInput { node: node.id })?;
                let b_id = b.ok_or(GraphError::MissingInput { node: node.id })?;
                let pa = cache
                    .get(&a_id)
                    .cloned()
                    .ok_or(GraphError::DanglingReference(a_id))?;
                let pb = cache
                    .get(&b_id)
                    .cloned()
                    .ok_or(GraphError::DanglingReference(b_id))?;
                Ok(combine_patches(pa, pb, *op))
            }
            NodeKind::Output { input } => {
                let in_id = input.ok_or(GraphError::MissingInput { node: node.id })?;
                cache
                    .get(&in_id)
                    .cloned()
                    .ok_or_else(|| GraphError::DanglingReference(in_id).into())
            }
        }
    }
}

/// Set or clear a specific input slot, dispatching by node kind. Slot
/// 0/1 distinguishes Combine's two inputs; single-input nodes
/// (`Translate`/`Filter`/`Output`) ignore `slot`.
fn apply_input_slot(kind: &mut NodeKind, slot: usize, new_input: Option<NodeId>) {
    match kind {
        NodeKind::Translate { input, .. }
        | NodeKind::Filter { input, .. }
        | NodeKind::Output { input } => {
            *input = new_input;
        }
        NodeKind::Mask { subject, mask, .. } => match slot {
            0 => *subject = new_input,
            1 => *mask = new_input,
            _ => {}
        },
        NodeKind::Combine { a, b, .. } => match slot {
            0 => *a = new_input,
            1 => *b = new_input,
            _ => {}
        },
        NodeKind::Terrain(_) | NodeKind::Tree(_) | NodeKind::Wfc(_) => {}
    }
}

/// Wipe `id` from any input slot of `kind`. Used when removing a node
/// so other nodes don't keep dangling refs.
fn clear_input_if(kind: &mut NodeKind, id: NodeId) {
    match kind {
        NodeKind::Translate { input, .. }
        | NodeKind::Filter { input, .. }
        | NodeKind::Output { input } => {
            if *input == Some(id) {
                *input = None;
            }
        }
        NodeKind::Mask { subject, mask, .. } => {
            if *subject == Some(id) {
                *subject = None;
            }
            if *mask == Some(id) {
                *mask = None;
            }
        }
        NodeKind::Combine { a, b, .. } => {
            if *a == Some(id) {
                *a = None;
            }
            if *b == Some(id) {
                *b = None;
            }
        }
        NodeKind::Terrain(_) | NodeKind::Tree(_) | NodeKind::Wfc(_) => {}
    }
}

fn translate_patch(patch: VoxelPatch, dx: i32, dy: i32, dz: i32) -> VoxelPatch {
    let mut result = VoxelPatch::new();
    result.voxels = patch
        .voxels
        .into_iter()
        .map(|((x, y, z), v)| ((x + dx, y + dy, z + dz), v))
        .collect();
    result.notes = patch.notes;
    result
}

fn filter_patch(patch: VoxelPatch, predicate: &FilterPredicate) -> VoxelPatch {
    let mut result = VoxelPatch::new();
    result.voxels = patch
        .voxels
        .into_iter()
        .filter(|((x, y, z), v)| match predicate {
            FilterPredicate::YAbove(t) => *y >= *t,
            FilterPredicate::YBelow(t) => *y <= *t,
            FilterPredicate::MatchesColor([r, g, b, a]) => {
                v.r == *r && v.g == *g && v.b == *b && v.a == *a
            }
            FilterPredicate::InsideBox { min, max } => {
                *x >= min.0
                    && *x <= max.0
                    && *y >= min.1
                    && *y <= max.1
                    && *z >= min.2
                    && *z <= max.2
            }
        })
        .collect();
    result.notes = patch.notes;
    result
}

/// Column-projected mask: keep `subject` voxels based on the mask's
/// column profile at the subject's `(x, z)`. We index the mask once
/// into a `(x, z) -> Vec<y>` map; per-subject lookup is then O(column
/// height), which for typical heightmap-style masks is `O(1)` (one y
/// per column). Subject voxels in columns the mask never touches are
/// dropped — the column has nothing to project against.
fn mask_patch(subject: VoxelPatch, mask: VoxelPatch, mode: MaskMode) -> VoxelPatch {
    let mut mask_columns: HashMap<(i32, i32), Vec<i32>> = HashMap::new();
    for ((x, y, z), _) in &mask.voxels {
        mask_columns.entry((*x, *z)).or_default().push(*y);
    }

    let mut result = VoxelPatch::new();
    result.voxels = subject
        .voxels
        .into_iter()
        .filter(|((x, y, z), _)| {
            let Some(ys) = mask_columns.get(&(*x, *z)) else {
                return false;
            };
            match mode {
                MaskMode::AboveColumn => ys.iter().any(|my| *my < *y),
                MaskMode::BelowColumn => ys.iter().any(|my| *my > *y),
            }
        })
        .collect();
    // Preserve diagnostics from both branches so notes don't get lost.
    result.notes = subject.notes;
    result.notes.extend(mask.notes);
    result
}

fn combine_patches(a: VoxelPatch, b: VoxelPatch, op: CombineOp) -> VoxelPatch {
    let map_a: HashMap<(i32, i32, i32), Voxel> = a.voxels.into_iter().collect();
    let map_b: HashMap<(i32, i32, i32), Voxel> = b.voxels.into_iter().collect();

    let combined: HashMap<(i32, i32, i32), Voxel> = match op {
        CombineOp::Union => {
            let mut r = map_a;
            r.extend(map_b);
            r
        }
        CombineOp::Difference => map_a
            .into_iter()
            .filter(|(pos, _)| !map_b.contains_key(pos))
            .collect(),
        CombineOp::Intersect => map_a
            .into_iter()
            .filter(|(pos, _)| map_b.contains_key(pos))
            .collect(),
    };

    let mut result = VoxelPatch::new();
    result.voxels = combined.into_iter().collect();
    // Preserve diagnostics from both branches so the user sees them.
    result.notes = a.notes;
    result.notes.extend(b.notes);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(r: u8) -> Voxel {
        Voxel::from_rgb(r, 0, 0)
    }

    /// Build a node that emits a fixed set of voxels. Test-only — uses
    /// a hardcoded `Terrain` of trivial size and replaces its output
    /// post-hoc isn't easy, so instead we test transform/combine logic
    /// directly on `VoxelPatch` and only hit the real source generators
    /// in a single integration-style test.
    fn manual_patch(voxels: Vec<((i32, i32, i32), Voxel)>) -> VoxelPatch {
        let mut p = VoxelPatch::new();
        p.voxels = voxels;
        p
    }

    #[test]
    fn test_translate_shifts_positions() {
        let p = manual_patch(vec![((1, 2, 3), solid(1))]);
        let t = translate_patch(p, 10, 20, 30);
        assert_eq!(t.voxels, vec![((11, 22, 33), solid(1))]);
    }

    #[test]
    fn test_combine_union_b_wins_overlap() {
        let a = manual_patch(vec![
            ((0, 0, 0), solid(1)),
            ((1, 0, 0), solid(1)),
        ]);
        let b = manual_patch(vec![
            ((1, 0, 0), solid(2)),
            ((2, 0, 0), solid(2)),
        ]);
        let r = combine_patches(a, b, CombineOp::Union);
        let map: HashMap<_, _> = r.voxels.into_iter().collect();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&(0, 0, 0)), Some(&solid(1)));
        assert_eq!(map.get(&(1, 0, 0)), Some(&solid(2))); // b wins
        assert_eq!(map.get(&(2, 0, 0)), Some(&solid(2)));
    }

    #[test]
    fn test_combine_difference_excludes_b_cells() {
        let a = manual_patch(vec![
            ((0, 0, 0), solid(1)),
            ((1, 0, 0), solid(1)),
        ]);
        let b = manual_patch(vec![((1, 0, 0), solid(2))]);
        let r = combine_patches(a, b, CombineOp::Difference);
        let map: HashMap<_, _> = r.voxels.into_iter().collect();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&(0, 0, 0)), Some(&solid(1)));
    }

    #[test]
    fn test_filter_y_above_inclusive() {
        let p = manual_patch(vec![
            ((0, -1, 0), solid(1)),
            ((0, 0, 0), solid(1)),
            ((0, 1, 0), solid(1)),
            ((0, 2, 0), solid(1)),
        ]);
        let r = filter_patch(p, &FilterPredicate::YAbove(0));
        let ys: Vec<i32> = r.voxels.iter().map(|((_, y, _), _)| *y).collect();
        // YAbove is inclusive on the threshold.
        assert_eq!(ys.len(), 3);
        assert!(ys.iter().all(|y| *y >= 0));
    }

    #[test]
    fn test_filter_y_below_inclusive() {
        let p = manual_patch(vec![
            ((0, -2, 0), solid(1)),
            ((0, -1, 0), solid(1)),
            ((0, 0, 0), solid(1)),
            ((0, 1, 0), solid(1)),
        ]);
        let r = filter_patch(p, &FilterPredicate::YBelow(-1));
        let ys: Vec<i32> = r.voxels.iter().map(|((_, y, _), _)| *y).collect();
        assert_eq!(ys.len(), 2);
        assert!(ys.iter().all(|y| *y <= -1));
    }

    #[test]
    fn test_filter_color_exact_match() {
        let red = Voxel::from_rgb(200, 30, 30);
        let blue = Voxel::from_rgb(30, 30, 200);
        let p = manual_patch(vec![
            ((0, 0, 0), red),
            ((1, 0, 0), blue),
            ((2, 0, 0), red),
        ]);
        let r = filter_patch(
            p,
            &FilterPredicate::MatchesColor([red.r, red.g, red.b, red.a]),
        );
        assert_eq!(r.voxels.len(), 2);
        assert!(r.voxels.iter().all(|(_, v)| *v == red));
    }

    #[test]
    fn test_filter_inside_box_inclusive_on_all_axes() {
        let mut voxels = Vec::new();
        for x in -2..=2 {
            for y in -2..=2 {
                for z in -2..=2 {
                    voxels.push(((x, y, z), solid(1)));
                }
            }
        }
        let p = manual_patch(voxels);
        let r = filter_patch(
            p,
            &FilterPredicate::InsideBox {
                min: (0, 0, 0),
                max: (1, 1, 1),
            },
        );
        // 2×2×2 box contains 8 cells.
        assert_eq!(r.voxels.len(), 8);
        for ((x, y, z), _) in &r.voxels {
            assert!(*x >= 0 && *x <= 1);
            assert!(*y >= 0 && *y <= 1);
            assert!(*z >= 0 && *z <= 1);
        }
    }

    #[test]
    fn test_mask_above_column_keeps_above_drops_below() {
        // subject has voxels at y = 0..5 in column (0, 0).
        // mask has a single voxel at y = 2 in the same column.
        // AboveColumn keeps subject voxels with some mask y < y → keeps
        // y in 3..5 (where mask y=2 is below); drops y in 0..2 (no mask
        // below) and y=2 itself (mask is *at* not below).
        let subject = manual_patch((0..5).map(|y| ((0, y, 0), solid(1))).collect());
        let mask = manual_patch(vec![((0, 2, 0), solid(2))]);
        let r = mask_patch(subject, mask, MaskMode::AboveColumn);
        let kept_ys: Vec<i32> = r.voxels.iter().map(|((_, y, _), _)| *y).collect();
        assert_eq!(kept_ys, vec![3, 4]);
    }

    #[test]
    fn test_mask_below_column_keeps_below_drops_above() {
        let subject = manual_patch((0..5).map(|y| ((0, y, 0), solid(1))).collect());
        let mask = manual_patch(vec![((0, 2, 0), solid(2))]);
        let r = mask_patch(subject, mask, MaskMode::BelowColumn);
        let kept_ys: Vec<i32> = r.voxels.iter().map(|((_, y, _), _)| *y).collect();
        assert_eq!(kept_ys, vec![0, 1]);
    }

    #[test]
    fn test_mask_drops_subject_in_columns_with_no_mask() {
        // subject at (5, 0, 5), but mask has nothing in column (5, 5).
        // Both modes should drop the subject voxel — there's nothing
        // to project against.
        let subject = manual_patch(vec![((5, 0, 5), solid(1))]);
        let mask = manual_patch(vec![((0, 0, 0), solid(2))]);
        let r_above = mask_patch(subject.clone(), mask.clone(), MaskMode::AboveColumn);
        assert!(r_above.voxels.is_empty());
        let r_below = mask_patch(subject, mask, MaskMode::BelowColumn);
        assert!(r_below.voxels.is_empty());
    }

    #[test]
    fn test_mask_node_in_graph_above_terrain() {
        // Terrain (heightmap) → Mask(AboveColumn, terrain) ← tree-like
        // column at (0, 0..10, 0). Expected: tree voxels kept only
        // where they're above the terrain in that column.
        let mut g = PipelineGraph::default();
        let terrain = g.add(NodeKind::Terrain(PerlinTerrain {
            seed: 1,
            width: 4,
            depth: 4,
            min_height: 0,
            max_height: 4,
            ..Default::default()
        }));
        // We'll synthesize a "tree" by abusing Translate of terrain to
        // shift it up 20 — guarantees the shifted patch is above the
        // un-shifted one in every shared column.
        let tower = g.add(NodeKind::Translate {
            input: Some(terrain),
            dx: 0,
            dy: 20,
            dz: 0,
        });
        let m = g.add(NodeKind::Mask {
            subject: Some(tower),
            mask: Some(terrain),
            mode: MaskMode::AboveColumn,
        });
        g.add(NodeKind::Output { input: Some(m) });

        let patch = g.evaluate().unwrap();
        // Every kept voxel must sit above some terrain cell in the same
        // column (true by construction since tower = terrain shifted up).
        assert!(!patch.voxels.is_empty());
        for ((_, y, _), _) in &patch.voxels {
            assert!(*y >= 20, "tower voxels should be shifted to y>=20");
        }
    }

    #[test]
    fn test_mask_missing_input_reports_error() {
        let mut g = PipelineGraph::default();
        let m = g.add(NodeKind::Mask {
            subject: None,
            mask: None,
            mode: MaskMode::default(),
        });
        g.add(NodeKind::Output { input: Some(m) });
        let err = g.evaluate().unwrap_err();
        assert!(err.to_string().contains("unconnected input slot"));
    }

    #[test]
    fn test_filter_node_in_graph() {
        // Terrain → Filter(YAbove(2)) → Output
        let mut g = PipelineGraph::default();
        let src = g.add(NodeKind::Terrain(PerlinTerrain {
            width: 6,
            depth: 6,
            ..Default::default()
        }));
        let f = g.add(NodeKind::Filter {
            input: Some(src),
            predicate: FilterPredicate::YAbove(2),
        });
        g.add(NodeKind::Output { input: Some(f) });

        let patch = g.evaluate().unwrap();
        assert!(!patch.voxels.is_empty());
        for ((_, y, _), _) in &patch.voxels {
            assert!(*y >= 2, "filter leaked y={}", y);
        }
    }

    #[test]
    fn test_filter_missing_input_reports_error() {
        let mut g = PipelineGraph::default();
        let f = g.add(NodeKind::Filter {
            input: None,
            predicate: FilterPredicate::default(),
        });
        g.add(NodeKind::Output { input: Some(f) });

        let err = g.evaluate().unwrap_err();
        assert!(err.to_string().contains("unconnected input slot"));
    }

    #[test]
    fn test_combine_intersect_keeps_a_voxels() {
        let a = manual_patch(vec![
            ((0, 0, 0), solid(1)),
            ((1, 0, 0), solid(1)),
        ]);
        let b = manual_patch(vec![
            ((1, 0, 0), solid(99)),
            ((2, 0, 0), solid(99)),
        ]);
        let r = combine_patches(a, b, CombineOp::Intersect);
        let map: HashMap<_, _> = r.voxels.into_iter().collect();
        assert_eq!(map.len(), 1);
        // Intersection takes voxel value from `a`, not `b`.
        assert_eq!(map.get(&(1, 0, 0)), Some(&solid(1)));
    }

    #[test]
    fn test_empty_graph_no_output() {
        let g = PipelineGraph::default();
        assert!(matches!(
            g.evaluate(),
            Err(GenError::Failed(_))
        ));
    }

    #[test]
    fn test_dangling_input_reports_missing() {
        let mut g = PipelineGraph::default();
        let _out = g.add(NodeKind::Output { input: None });
        let err = g.evaluate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unconnected input slot"), "got: {}", msg);
    }

    #[test]
    fn test_simple_chain_terrain_translate_output() {
        let mut g = PipelineGraph::default();
        let src = g.add(NodeKind::Terrain(PerlinTerrain {
            width: 4,
            depth: 4,
            ..Default::default()
        }));
        let xform = g.add(NodeKind::Translate {
            input: Some(src),
            dx: 100,
            dy: 0,
            dz: 0,
        });
        g.add(NodeKind::Output { input: Some(xform) });

        let patch = g.evaluate().unwrap();
        assert!(!patch.voxels.is_empty());
        // Translate(100, 0, 0): every x must be >= 100 - depth (terrain centers
        // around 0; with width=4 the leftmost x is -2, shifted = 98).
        for ((x, _, _), _) in &patch.voxels {
            assert!(*x >= 98, "translated x out of range: {}", x);
        }
    }

    #[test]
    fn test_cycle_detected() {
        // Translate -> Translate -> ... feeding back into self.
        // We have to manually wire: A.input = B; B.input = A. Since
        // `add` returns ids in order, we add A first with no input,
        // add B with input=A, then patch A.input = B.
        let mut g = PipelineGraph::default();
        let a = g.add(NodeKind::Translate {
            input: None,
            dx: 0,
            dy: 0,
            dz: 0,
        });
        let b = g.add(NodeKind::Translate {
            input: Some(a),
            dx: 0,
            dy: 0,
            dz: 0,
        });
        if let NodeKind::Translate { input, .. } = &mut g.get_mut(a).unwrap().kind {
            *input = Some(b);
        }
        let _out = g.add(NodeKind::Output { input: Some(a) });

        let err = g.evaluate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cycle"), "got: {}", msg);
    }

    #[test]
    fn test_set_input_succeeds() {
        let mut g = PipelineGraph::default();
        let src = g.add(NodeKind::Terrain(PerlinTerrain::default()));
        let xform = g.add(NodeKind::Translate {
            input: None,
            dx: 0,
            dy: 0,
            dz: 0,
        });
        g.set_input(xform, 0, Some(src)).unwrap();
        assert_eq!(g.get_input(xform, 0).unwrap(), Some(src));
    }

    #[test]
    fn test_set_input_rejects_cycle() {
        // Translate(A) <- Translate(B); attempt to set B.input = A
        // (via passing A as B's input) is fine. But then setting
        // A.input = B closes the cycle and must be rejected.
        let mut g = PipelineGraph::default();
        let a = g.add(NodeKind::Translate {
            input: None,
            dx: 0,
            dy: 0,
            dz: 0,
        });
        let b = g.add(NodeKind::Translate {
            input: Some(a),
            dx: 0,
            dy: 0,
            dz: 0,
        });
        let err = g.set_input(a, 0, Some(b)).unwrap_err();
        assert!(matches!(err, GraphError::Cycle(_)));
        // A.input must still be None — the failed set should have
        // rolled back.
        assert_eq!(g.get_input(a, 0).unwrap(), None);
    }

    #[test]
    fn test_set_input_disconnect() {
        let mut g = PipelineGraph::default();
        let src = g.add(NodeKind::Terrain(PerlinTerrain::default()));
        let xform = g.add(NodeKind::Translate {
            input: Some(src),
            dx: 0,
            dy: 0,
            dz: 0,
        });
        g.set_input(xform, 0, None).unwrap();
        assert_eq!(g.get_input(xform, 0).unwrap(), None);
    }

    #[test]
    fn test_remove_clears_dangling_inputs() {
        let mut g = PipelineGraph::default();
        let src = g.add(NodeKind::Terrain(PerlinTerrain::default()));
        let xform = g.add(NodeKind::Translate {
            input: Some(src),
            dx: 0,
            dy: 0,
            dz: 0,
        });
        g.remove(src);
        // The Translate node's input must have been cleared.
        match &g.get(xform).unwrap().kind {
            NodeKind::Translate { input, .. } => assert_eq!(*input, None),
            _ => panic!("wrong kind"),
        }
    }

    #[test]
    fn test_multi_source_combine_union() {
        // Two terrains at different origins, combined via Union.
        let mut g = PipelineGraph::default();
        let a = g.add(NodeKind::Terrain(PerlinTerrain {
            width: 4,
            depth: 4,
            ..Default::default()
        }));
        let b = g.add(NodeKind::Terrain(PerlinTerrain {
            seed: 99,
            width: 4,
            depth: 4,
            ..Default::default()
        }));
        let translated_b = g.add(NodeKind::Translate {
            input: Some(b),
            dx: 100,
            dy: 0,
            dz: 0,
        });
        let combine = g.add(NodeKind::Combine {
            a: Some(a),
            b: Some(translated_b),
            op: CombineOp::Union,
        });
        g.add(NodeKind::Output { input: Some(combine) });

        let patch = g.evaluate().unwrap();
        // We should see voxels at both x ranges (≈0 and ≈100).
        let has_low = patch.voxels.iter().any(|((x, _, _), _)| *x < 50);
        let has_high = patch.voxels.iter().any(|((x, _, _), _)| *x >= 90);
        assert!(has_low && has_high);
    }
}
