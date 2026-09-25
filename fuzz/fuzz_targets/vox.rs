#![no_main]

use libfuzzer_sys::fuzz_target;

// An `Err` is the right answer to a malformed file. A panic, a hang, a
// runaway allocation or a see-through voxel in the world is a finding.
fuzz_target!(|data: &[u8]| {
    let Ok(import) = voxelith::io::import_vox(&mut &data[..], true) else {
        return;
    };
    for (_, chunk) in import.world.chunks() {
        for (_, voxel) in chunk.read().iter_solid() {
            assert_eq!(
                voxel.a, 255,
                "a .vox import put a translucent voxel in the world"
            );
        }
    }
});
