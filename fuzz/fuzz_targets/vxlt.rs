#![no_main]

use std::io::Write;

use flate2::write::GzEncoder;
use flate2::Compression;
use libfuzzer_sys::fuzz_target;
use voxelith::io::Project;

// The input is the decompressed body: mutated gzip bytes would almost
// all stop at the CRC, before any of the parsing this target is after.
fn wrap(body: &[u8]) -> Vec<u8> {
    let mut header = b"VXLT".to_vec();
    header.extend_from_slice(&1u32.to_le_bytes());
    let mut gz = GzEncoder::new(header, Compression::none());
    gz.write_all(body).expect("writing to a Vec cannot fail");
    gz.finish().expect("writing to a Vec cannot fail")
}

fuzz_target!(|body: &[u8]| {
    let file = wrap(body);
    let Ok(project) = Project::load(&mut &file[..]) else {
        return;
    };
    let Ok(world) = project.to_world() else {
        return;
    };
    for (_, chunk) in world.chunks() {
        for (_, voxel) in chunk.read().iter_solid() {
            assert_eq!(
                voxel.a, 255,
                "a .vxlt load put a translucent voxel in the world"
            );
        }
    }
});
