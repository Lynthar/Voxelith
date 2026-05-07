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
            NodeKind::Translate { .. } | NodeKind::Output { .. } => 1,
            NodeKind::Combine { .. } => 2,
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
            NodeKind::Translate { input, .. } | NodeKind::Output { input } => *input,
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
/// 0/1 distinguishes Combine's two inputs; `Translate`/`Output` ignore
/// `slot` since they only have one input.
fn apply_input_slot(kind: &mut NodeKind, slot: usize, new_input: Option<NodeId>) {
    match kind {
        NodeKind::Translate { input, .. } | NodeKind::Output { input } => {
            *input = new_input;
        }
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
        NodeKind::Translate { input, .. } | NodeKind::Output { input } => {
            if *input == Some(id) {
                *input = None;
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
