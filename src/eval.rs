//! The eval set: what "good enough" means, written as assertions a
//! machine can check.
//!
//! An eval case is a task handed to an agent plus the properties the
//! result has to have. The properties are the same ones `describe`
//! already measures — connected components, floating parts, enclosed
//! interior, symmetry, size — which is the point: the numbers an agent
//! reads while it works are the numbers it is graded on, so "check your
//! own work" and "did it pass" cannot drift apart.
//!
//! Grading is deliberately **code-based, not model-based**. A judge
//! that is itself a language model would bring the failure mode this
//! whole layer exists to remove: an opinion where a measurement will
//! do. Whether two halves of a sword are actually joined is not a
//! matter of taste, and no rendered view answers it reliably.
//!
//! What this does *not* do is run the agent. Driving a model, sampling
//! it k times, pinning its temperature — all of that belongs to whoever
//! is doing the measuring, and it changes with the model. This module
//! takes a finished `.vxlt` and says whether it meets the bar, which is
//! the part that should stay the same across every model and every year.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::agent_ops::{self, AxisSpec, Description, DocumentView};
use crate::editor::Socket;
use crate::io;
use crate::procgen::PipelineGraph;

/// One task and the bar its result has to clear.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalCase {
    /// Stable id. In suite mode this is also the file name the result
    /// is looked up under (`<id>.vxlt`).
    pub id: String,
    /// The task, as it would be given to an agent. Verbatim — a case
    /// that is graded on symmetry has to *ask* for symmetry, or it is
    /// measuring whether the model guessed the grader's mind.
    pub task: String,
    /// Why this case exists and what it catches. Not graded; read by
    /// whoever is looking at a failure.
    #[serde(default)]
    pub notes: Option<String>,
    pub expect: Expect,
}

/// The properties a result must have. Every field is optional: a case
/// asserts what its task actually asked for and stays quiet about the
/// rest. Asserting more than the task demanded grades the model on
/// guessing rather than on building.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expect {
    #[serde(default)]
    pub voxel_count: Option<Bound>,
    /// Extents along x / y / z, in cells.
    #[serde(default)]
    pub size: Option<[Bound; 3]>,
    /// Connected components, 6-connectivity. `{"max": 1}` is "one solid
    /// piece" — the single most common way a model looks finished and
    /// isn't.
    #[serde(default)]
    pub components: Option<Bound>,
    /// Parts that never reach the lowest layer. `{"max": 0}` for things
    /// that stand on the ground; left unset for anything with a canopy.
    #[serde(default)]
    pub floating_components: Option<Bound>,
    /// Fully-surrounded interior voxels. `{"max": 0}` is "hollow, ready
    /// to export".
    #[serde(default)]
    pub enclosed: Option<Bound>,
    #[serde(default)]
    pub symmetry: Vec<SymmetryExpect>,
    /// Solid cells touching the lowest layer. A low ceiling is how a
    /// case asks for something that *spans* rather than sits — an arch
    /// and a wall are otherwise indistinguishable to every other bar.
    #[serde(default)]
    pub footprint: Option<Bound>,
    /// Sealed air pockets. `{"min": 1}` asks for a room; `{"max": 0}`
    /// asks for a model with no bubble hidden inside the export.
    #[serde(default)]
    pub cavities: Option<Bound>,
    /// Voxels flagged emissive / metallic. `{"min": 1}` is how a case
    /// checks that material flags are reachable at all — geometry can
    /// be right while every material knob went untouched.
    #[serde(default)]
    pub emissive: Option<Bound>,
    #[serde(default)]
    pub metallic: Option<Bound>,
    /// Nodes in the project's pipeline graph. `{"min": 2}` is how a
    /// case says *"generate this, don't hand-place it"* — and it is
    /// checkable only because the graph travels with the project.
    #[serde(default)]
    pub graph_nodes: Option<Bound>,
}

/// Inclusive range. Either end may be omitted for a one-sided bound.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bound {
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
}

impl Bound {
    fn holds(&self, actual: f64) -> bool {
        self.min.is_none_or(|min| actual >= min) && self.max.is_none_or(|max| actual <= max)
    }

    /// How the bound reads in a report: `<= 1`, `>= 80`, `80..2000`.
    fn describe(&self) -> String {
        match (self.min, self.max) {
            (Some(min), Some(max)) => format!("{}..{}", number(min), number(max)),
            (Some(min), None) => format!(">= {}", number(min)),
            (None, Some(max)) => format!("<= {}", number(max)),
            (None, None) => "anything".to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SymmetryExpect {
    pub axis: AxisSpec,
    /// Largest share of voxels allowed to have no mirror image. `0.0`
    /// demands exact symmetry; a couple of percent tolerates a handle
    /// or an inscription on one side.
    pub max_ratio: f64,
}

/// One assertion and how it came out.
#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub name: String,
    pub expected: String,
    pub actual: String,
    pub pass: bool,
}

/// A graded case.
#[derive(Debug, Clone, Serialize)]
pub struct CaseReport {
    pub id: String,
    pub pass: bool,
    /// Absent when the result was graded, present when there was
    /// nothing to grade (no file for this case, or it wouldn't load).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub checks: Vec<Check>,
}

/// A whole suite.
#[derive(Debug, Clone, Serialize)]
pub struct SuiteReport {
    pub passed: usize,
    pub total: usize,
    pub cases: Vec<CaseReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalError {
    pub code: &'static str,
    pub message: String,
}

impl EvalError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Same envelope as the other headless commands: one shape to parse
    /// whichever way the run went.
    pub fn to_json(&self) -> String {
        #[derive(Serialize)]
        struct Envelope<'a> {
            ok: bool,
            error: &'a EvalError,
        }
        serde_json::to_string_pretty(&Envelope {
            ok: false,
            error: self,
        })
        .unwrap_or_else(|_| r#"{"ok":false,"error":{"code":"internal","message":""}}"#.into())
    }
}

impl SuiteReport {
    pub fn to_json(&self) -> String {
        #[derive(Serialize)]
        struct Envelope<'a> {
            ok: bool,
            #[serde(flatten)]
            report: &'a SuiteReport,
        }
        serde_json::to_string_pretty(&Envelope {
            // `ok` is "the run completed", not "everything passed" —
            // the same meaning it has everywhere else here. Whether the
            // bar was cleared is `passed == total`, and the exit code.
            ok: true,
            report: self,
        })
        .unwrap_or_else(|_| r#"{"ok":false}"#.into())
    }
}

/// What to grade, and against what.
#[derive(Debug, Clone, Default)]
pub struct EvalRequest {
    /// A case file, or a directory of them.
    pub cases: PathBuf,
    /// The single result to grade. Mutually exclusive with `results`;
    /// this is the form an agent uses to check its own work, where the
    /// file it just saved has whatever name it likes.
    pub project: Option<PathBuf>,
    /// A directory holding one `<case-id>.vxlt` per case.
    pub results: Option<PathBuf>,
}

/// Load the cases, grade each one, and report.
pub fn run_eval(request: &EvalRequest) -> Result<SuiteReport, EvalError> {
    let cases = load_cases(&request.cases)?;
    if cases.is_empty() {
        return Err(EvalError::new(
            "no_cases",
            format!("no eval cases in {}", request.cases.display()),
        ));
    }
    let single = match (&request.project, &request.results) {
        (Some(_), Some(_)) => {
            return Err(EvalError::new(
                "conflicting_results",
                "pass either --project (one result) or --results (a directory), not both",
            ))
        }
        (None, None) => {
            return Err(EvalError::new(
                "no_results",
                "nothing to grade: pass --project <file.vxlt> or --results <dir>",
            ))
        }
        (Some(project), None) => {
            if cases.len() > 1 {
                return Err(EvalError::new(
                    "ambiguous_result",
                    format!(
                        "--project grades one case, but {} were loaded; use --results <dir> \
                         with one <case-id>.vxlt per case",
                        cases.len()
                    ),
                ));
            }
            Some(project.clone())
        }
        (None, Some(_)) => None,
    };

    let mut reports = Vec::with_capacity(cases.len());
    for case in &cases {
        let path = match &single {
            Some(path) => path.clone(),
            None => request
                .results
                .as_ref()
                .expect("checked above")
                .join(format!("{}.vxlt", case.id)),
        };
        reports.push(grade_file(case, &path));
    }

    Ok(SuiteReport {
        passed: reports.iter().filter(|r| r.pass).count(),
        total: reports.len(),
        cases: reports,
    })
}

/// Read one case file or a directory of them, sorted by id so a suite
/// report reads the same way twice.
fn load_cases(path: &Path) -> Result<Vec<EvalCase>, EvalError> {
    let mut files = Vec::new();
    if path.is_dir() {
        let entries = std::fs::read_dir(path).map_err(|e| {
            EvalError::new(
                "cases_unreadable",
                format!("could not read {}: {e}", path.display()),
            )
        })?;
        for entry in entries {
            // A directory entry that can't be read is not "no case
            // here": suite mode reports a missing result as a failed
            // case, so a case that silently vanished from the *set*
            // would shrink the bar instead of failing anything.
            let entry = entry
                .map_err(|e| {
                    EvalError::new(
                        "cases_unreadable",
                        format!("could not read an entry in {}: {e}", path.display()),
                    )
                })?
                .path();
            // Case-insensitive: the cases are data files, and on
            // Windows `.JSON` is the same file the author meant.
            let is_case = entry
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("json"));
            if is_case {
                files.push(entry);
            }
        }
        files.sort();
    } else {
        files.push(path.to_path_buf());
    }

    let mut cases = Vec::with_capacity(files.len());
    for file in files {
        let text = std::fs::read_to_string(&file).map_err(|e| {
            EvalError::new(
                "cases_unreadable",
                format!("could not read {}: {e}", file.display()),
            )
        })?;
        let case: EvalCase = serde_json::from_str(&text).map_err(|e| {
            EvalError::new(
                "case_invalid",
                format!("{} is not a valid eval case: {e}", file.display()),
            )
        })?;
        cases.push(case);
    }
    cases.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(cases)
}

/// Grade one case against one project file. A missing or unreadable
/// result is a failed case, not a failed run — "the agent didn't get
/// there" is an outcome the suite has to be able to report.
fn grade_file(case: &EvalCase, path: &Path) -> CaseReport {
    let (world, state, _) = match io::load_world_with_state(path) {
        Ok(loaded) => loaded,
        Err(e) => {
            return CaseReport {
                id: case.id.clone(),
                pass: false,
                error: Some(format!("could not load {}: {e}", path.display())),
                checks: Vec::new(),
            }
        }
    };
    let sockets: Vec<Socket> = state
        .sockets
        .iter()
        .map(|s| Socket::new(s.name.clone(), s.position, s.normal))
        .collect();
    let description = agent_ops::describe(DocumentView {
        world: &world,
        selection: None,
        sockets: &sockets,
        graph: &state.graph,
        undo_depth: 0,
        redo_depth: 0,
    });
    grade(case, &description)
}

/// Grade a description against a case. Pure — this is what the tests
/// drive, and what a caller with a live session could use directly.
pub fn grade(case: &EvalCase, description: &Description) -> CaseReport {
    let mut checks = Vec::new();
    let expect = &case.expect;

    if let Some(bound) = expect.voxel_count {
        checks.push(check("voxel_count", bound, description.voxel_count as f64));
    }
    if let Some(bounds) = &expect.size {
        match description.size {
            Some(size) => {
                for (axis, (bound, actual)) in ["size.x", "size.y", "size.z"]
                    .iter()
                    .zip(bounds.iter().zip(size.iter()))
                {
                    checks.push(check(axis, *bound, *actual as f64));
                }
            }
            // An empty document has no extent to measure, and treating
            // that as `[0, 0, 0]` passes any bound written as a ceiling
            // alone — "at most 10 cells across" is true of nothing at
            // all. Same rule as the structural pass below: a bar that
            // was never measured doesn't get to report a pass.
            None => checks.push(Check {
                name: "size".to_string(),
                expected: "measurable".to_string(),
                actual: "not measured (the document is empty)".to_string(),
                pass: false,
            }),
        }
    }

    // Everything below needs the structural pass. It is skipped on
    // documents too big to measure, and a case that asks about
    // structure can't be graded without it — say so rather than
    // reporting a pass nobody checked.
    let structure = description.structure.as_ref();
    let wants_structure = expect.components.is_some()
        || expect.floating_components.is_some()
        || expect.enclosed.is_some()
        || expect.footprint.is_some()
        || expect.cavities.is_some()
        || !expect.symmetry.is_empty();
    match (structure, wants_structure) {
        (None, true) => checks.push(Check {
            name: "structure".to_string(),
            expected: "measurable".to_string(),
            actual: "not measured (document too large)".to_string(),
            pass: false,
        }),
        (Some(structure), _) => {
            if let Some(bound) = expect.components {
                checks.push(check("components", bound, structure.components as f64));
            }
            if let Some(bound) = expect.floating_components {
                checks.push(check(
                    "floating_components",
                    bound,
                    structure.floating_components as f64,
                ));
            }
            if let Some(bound) = expect.enclosed {
                checks.push(check("enclosed", bound, structure.enclosed as f64));
            }
            if let Some(bound) = expect.footprint {
                checks.push(check("footprint", bound, structure.footprint as f64));
            }
            if let Some(bound) = expect.cavities {
                match &structure.cavities {
                    Some(cavities) => checks.push(check("cavities", bound, cavities.count as f64)),
                    // Same rule as `structure` itself: a bar nobody
                    // measured must not read as a bar that was cleared.
                    None => checks.push(Check {
                        name: "cavities".to_string(),
                        expected: bound.describe(),
                        actual: "not measured (bounding box too large)".to_string(),
                        pass: false,
                    }),
                }
            }
            for want in &expect.symmetry {
                let axis = axis_name(want.axis);
                let measured = structure
                    .symmetry
                    .iter()
                    .find(|s| s.axis == axis)
                    .map(|s| s.ratio)
                    .unwrap_or(1.0);
                checks.push(Check {
                    name: format!("symmetry.{axis}"),
                    expected: format!("<= {}", number(want.max_ratio)),
                    actual: number(measured),
                    pass: measured <= want.max_ratio,
                });
            }
        }
        (None, false) => {}
    }

    if let Some(bound) = expect.emissive {
        checks.push(check("emissive", bound, description.emissive as f64));
    }
    if let Some(bound) = expect.metallic {
        checks.push(check("metallic", bound, description.metallic as f64));
    }
    if let Some(bound) = expect.graph_nodes {
        let nodes = description
            .graph
            .as_ref()
            .map(|g: &PipelineGraph| g.nodes.len())
            .unwrap_or(0);
        checks.push(check("graph_nodes", bound, nodes as f64));
    }

    CaseReport {
        id: case.id.clone(),
        pass: checks.iter().all(|c| c.pass),
        error: None,
        checks,
    }
}

fn check(name: &str, bound: Bound, actual: f64) -> Check {
    // A bound that can't be met, or can't be missed, is an authoring
    // mistake rather than a judgment about the model — and the failure
    // mode of the empty one is the dangerous direction: `{}` reads as a
    // bar in the case file and grades every result as a pass. Failing
    // it here names the case and the field; the alternative, refusing
    // to load the whole suite, punishes the eleven cases that are fine.
    let unusable = match (bound.min, bound.max) {
        (None, None) => Some("names neither min nor max, so it asserts nothing".to_string()),
        (Some(min), Some(max)) if min > max => Some(format!(
            "asks for {} .. {}, which nothing can satisfy",
            number(min),
            number(max)
        )),
        _ => None,
    };
    if let Some(why) = unusable {
        return Check {
            name: name.to_string(),
            expected: "a usable bound".to_string(),
            actual: format!("the case {why}"),
            pass: false,
        };
    }
    Check {
        name: name.to_string(),
        expected: bound.describe(),
        actual: number(actual),
        pass: bound.holds(actual),
    }
}

fn axis_name(axis: AxisSpec) -> &'static str {
    match axis {
        AxisSpec::X => "x",
        AxisSpec::Y => "y",
        AxisSpec::Z => "z",
    }
}

/// Render a number the way a person reads it: counts without a decimal
/// point, ratios with just enough digits to be meaningful.
fn number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        format!("{value:.4}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Voxel, World};

    fn case(expect_json: &str) -> EvalCase {
        let text = format!(r#"{{"id":"t","task":"build something","expect":{expect_json}}}"#);
        serde_json::from_str(&text).expect("case should parse")
    }

    fn describe_of(world: &World) -> Description {
        agent_ops::describe(DocumentView {
            world,
            selection: None,
            sockets: &[],
            graph: &PipelineGraph::default(),
            undo_depth: 0,
            redo_depth: 0,
        })
    }

    fn filled(cells: impl IntoIterator<Item = (i32, i32, i32)>) -> World {
        let mut world = World::new();
        for (x, y, z) in cells {
            world.set_voxel(x, y, z, Voxel::from_rgb(140, 140, 140));
        }
        world
    }

    #[test]
    fn a_one_cell_gap_fails_the_connectivity_bar() {
        // The case this whole layer was built for: two halves that look
        // joined in every render and are not.
        let joined = filled((0..6).map(|x| (x, 0, 0)));
        let split = filled((0..3).chain(4..7).map(|x| (x, 0, 0)));
        let case = case(r#"{"components":{"max":1}}"#);

        assert!(grade(&case, &describe_of(&joined)).pass);
        let failed = grade(&case, &describe_of(&split));
        assert!(!failed.pass);
        assert_eq!(failed.checks[0].actual, "2");
        assert_eq!(failed.checks[0].expected, "<= 1");
    }

    #[test]
    fn a_case_only_grades_what_it_asked_for() {
        // An empty `expect` passes anything. Cases assert what their
        // task demanded and stay quiet about the rest, so a model isn't
        // graded on guessing unstated requirements.
        let report = grade(&case("{}"), &describe_of(&filled([(0, 0, 0)])));
        assert!(report.pass);
        assert!(report.checks.is_empty());
    }

    #[test]
    fn symmetry_is_graded_as_a_share_not_a_count() {
        // A 4-wide bar with a two-cell bump on one side: 2 of 6 voxels
        // have no mirror image.
        let world = filled((0..4).map(|x| (x, 0, 0)).chain([(0, 1, 0), (1, 1, 0)]));
        let lenient = grade(
            &case(r#"{"symmetry":[{"axis":"x","max_ratio":0.5}]}"#),
            &describe_of(&world),
        );
        assert!(lenient.pass, "2 of 6 is inside a 50% tolerance");
        let strict = grade(
            &case(r#"{"symmetry":[{"axis":"x","max_ratio":0.1}]}"#),
            &describe_of(&world),
        );
        assert!(!strict.pass);
        assert_eq!(strict.checks[0].name, "symmetry.x");
    }

    #[test]
    fn a_hollow_bar_is_the_difference_between_solid_and_shell() {
        let solid =
            filled((0..3).flat_map(|x| (0..3).flat_map(move |y| (0..3).map(move |z| (x, y, z)))));
        let case = case(r#"{"enclosed":{"max":0}}"#);
        assert!(
            !grade(&case, &describe_of(&solid)).pass,
            "a solid 3³ has an interior"
        );

        let mut shell = solid.deep_clone();
        shell.set_voxel(1, 1, 1, Voxel::AIR);
        assert!(grade(&case, &describe_of(&shell)).pass);
    }

    #[test]
    fn asking_for_a_graph_is_how_a_case_says_generate_rather_than_place() {
        use crate::procgen::{NodeKind, PerlinTerrain};

        let world = filled([(0, 0, 0)]);
        let hand_placed = agent_ops::describe(DocumentView {
            world: &world,
            selection: None,
            sockets: &[],
            graph: &PipelineGraph::default(),
            undo_depth: 0,
            redo_depth: 0,
        });
        let mut graph = PipelineGraph::default();
        let src = graph.add(NodeKind::Terrain(PerlinTerrain::default()));
        graph.add(NodeKind::Output { input: Some(src) });
        let generated = agent_ops::describe(DocumentView {
            world: &world,
            selection: None,
            sockets: &[],
            graph: &graph,
            undo_depth: 0,
            redo_depth: 0,
        });

        let case = case(r#"{"graph_nodes":{"min":2}}"#);
        assert!(!grade(&case, &hand_placed).pass);
        assert!(grade(&case, &generated).pass);
    }

    #[test]
    fn a_structural_bar_on_an_unmeasured_document_fails_rather_than_passes() {
        // `structure` is `None` on an empty document. A case asking
        // about components must not read that as "zero problems".
        let empty = describe_of(&World::new());
        assert!(empty.structure.is_none());
        let report = grade(&case(r#"{"components":{"max":1}}"#), &empty);
        assert!(!report.pass);
        assert_eq!(report.checks[0].name, "structure");
    }

    /// The cases in the repository are data, so nothing but this
    /// notices when one is malformed — and the id has to match the file
    /// name, because suite mode looks a result up as `<id>.vxlt`. A
    /// mismatch there reports "the agent never attempted this case"
    /// when the truth is that the case is misfiled.
    #[test]
    fn every_case_in_the_repository_loads_and_is_named_after_its_id() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        for (folder, least) in [("evals/cases", 6), ("evals/advanced", 6)] {
            let dir = root.join(folder);
            let cases = load_cases(&dir).expect("the shipped cases must load");
            assert!(
                cases.len() >= least,
                "{folder} has only {} cases",
                cases.len()
            );
            for case in &cases {
                let file = dir.join(format!("{}.json", case.id));
                assert!(
                    file.exists(),
                    "case {:?} should live in {}",
                    case.id,
                    file.display()
                );
                assert!(
                    !case.task.trim().is_empty(),
                    "case {:?} has no task",
                    case.id
                );
            }
        }
    }

    /// What `arched-bridge` needed all along. A wall and a span can
    /// agree on every other bar; ground contact is where they differ.
    #[test]
    fn a_footprint_bar_is_what_separates_a_span_from_a_wall() {
        let wall = filled((0..6).flat_map(|x| (0..4).map(move |y| (x, y, 0))));
        let span = filled(
            [
                (0, 0, 0),
                (0, 1, 0),
                (0, 2, 0),
                (5, 0, 0),
                (5, 1, 0),
                (5, 2, 0),
            ]
            .into_iter()
            .chain((0..6).map(|x| (x, 3, 0))),
        );
        let case = case(r#"{"footprint":{"max":2}}"#);
        assert!(!grade(&case, &describe_of(&wall)).pass);
        assert!(grade(&case, &describe_of(&span)).pass);
    }

    #[test]
    fn material_flags_are_gradable_because_geometry_can_be_right_without_them() {
        let mut world = filled([(0, 0, 0), (1, 0, 0)]);
        let case = case(r#"{"emissive":{"min":1}}"#);
        assert!(!grade(&case, &describe_of(&world)).pass);

        let mut lamp = Voxel::from_rgb(255, 220, 120);
        lamp.set_emissive(true);
        world.set_voxel(1, 0, 0, lamp);
        assert!(grade(&case, &describe_of(&world)).pass);
    }

    #[test]
    fn bounds_read_the_way_a_person_writes_them() {
        assert_eq!(
            Bound {
                min: None,
                max: Some(1.0)
            }
            .describe(),
            "<= 1"
        );
        assert_eq!(
            Bound {
                min: Some(80.0),
                max: None
            }
            .describe(),
            ">= 80"
        );
        assert_eq!(
            Bound {
                min: Some(80.0),
                max: Some(2000.0)
            }
            .describe(),
            "80..2000"
        );
    }
}
