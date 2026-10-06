//! The binary's own wiring: which flags reach which subcommand, and the
//! exit codes a script gates on. What runs behind them is tested in the
//! library.

use std::path::Path;
use std::process::{Command, Output};

fn voxelith(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_voxelith"))
        .args(args)
        .current_dir(dir)
        .output()
        .expect("the binary runs")
}

fn stdout_json(output: &Output) -> serde_json::Value {
    serde_json::from_slice(&output.stdout).expect("stdout is one JSON envelope")
}

/// A 2×2×2 cube saved as `cube.vxlt` in `dir`.
fn cube_project(dir: &Path) {
    std::fs::write(
        dir.join("cube.json"),
        r#"{"version":1,"ops":[{"op":"box","min":[0,0,0],"max":[1,1,1],"voxel":{"rgb":[1,2,3]}}]}"#,
    )
    .unwrap();
    let made = voxelith(dir, &["exec", "cube.json", "--out", "cube.vxlt"]);
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stdout)
    );
}

fn case(dir: &Path, name: &str, min_voxels: u64) {
    let case = format!(
        r#"{{"id":"{name}","task":"a cube","expect":{{"voxel_count":{{"min":{min_voxels}}}}}}}"#
    );
    std::fs::write(dir.join(format!("{name}.json")), case).unwrap();
}

#[test]
fn eval_exits_zero_only_when_every_case_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    cube_project(dir);
    case(dir, "fits", 8);
    case(dir, "too-small", 9);

    let pass = voxelith(dir, &["eval", "fits.json", "--project", "cube.vxlt"]);
    assert_eq!(pass.status.code(), Some(0));
    assert_eq!(stdout_json(&pass)["passed"], serde_json::json!(1));

    // A failed grade is still a completed run: `ok` stays true.
    let fail = voxelith(dir, &["eval", "too-small.json", "--project", "cube.vxlt"]);
    assert_eq!(fail.status.code(), Some(1));
    let report = stdout_json(&fail);
    assert_eq!(report["ok"], serde_json::json!(true));
    assert_eq!(report["passed"], serde_json::json!(0));

    let unusable = voxelith(dir, &["eval", "fits.json"]);
    assert_eq!(unusable.status.code(), Some(1));
    assert_eq!(stdout_json(&unusable)["ok"], serde_json::json!(false));
}

#[test]
fn inspect_describes_and_slices_without_writing() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    cube_project(dir);
    let before = std::fs::read(dir.join("cube.vxlt")).unwrap();

    let out = voxelith(dir, &["inspect", "cube.vxlt", "--slice", r#"{"index":0}"#]);
    assert!(out.status.success());
    let body = stdout_json(&out);
    assert_eq!(body["description"]["voxel_count"], serde_json::json!(8));
    // The header line, then one row per z: a 2×2 floor at y = 0.
    assert_eq!(body["slice"][1], serde_json::json!("##"));
    assert_eq!(body["slice"][2], serde_json::json!("##"));
    assert_eq!(std::fs::read(dir.join("cube.vxlt")).unwrap(), before);
}
