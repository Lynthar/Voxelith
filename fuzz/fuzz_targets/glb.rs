#![no_main]

use libfuzzer_sys::fuzz_target;

// A fixed low resolution keeps each run cheap; the parse, the scene walk
// and the texture decode are what this target is after.
const RESOLUTION: u32 = 32;

fuzz_target!(|data: &[u8]| {
    let Ok(patch) = voxelith::io::voxelize_glb(data, RESOLUTION) else {
        return;
    };
    for (_, voxel) in patch.voxels.iter().filter(|(_, v)| v.is_solid()) {
        assert_eq!(voxel.a, 255, "a GLB import produced a translucent voxel");
    }
});
