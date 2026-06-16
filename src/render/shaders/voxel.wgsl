// Voxel rendering shader
// Supports basic lighting with ambient and directional components

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    // Per-vertex Ambient Occlusion in [0, 1]. 0 = fully occluded
    // (sharp inside corner), 1 = no occlusion. Bilinearly
    // interpolated across the quad by the rasterizer.
    @location(3) ao: f32,
    // Faction tint zone — present in the vertex layout for GLB export;
    // unused by rendering, declared so the pipeline accepts the vertex
    // buffer's @location(4) attribute.
    @location(4) tint_zone: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) ao: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_position = in.position;
    out.normal = in.normal;
    out.color = in.color;
    out.ao = in.ao;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Light direction (sun-like, from upper right)
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));

    // Ambient light
    let ambient_strength = 0.3;
    let ambient = ambient_strength;

    // Diffuse lighting
    let diff = max(dot(in.normal, light_dir), 0.0);
    let diffuse = diff * 0.7;

    // Combine lighting
    let lighting = ambient + diffuse;

    // Ambient occlusion: maps the per-vertex AO factor to a
    // brightness multiplier. ao=0 (fully occluded inside corner)
    // → ambient_min; ao=1 (open space) → 1.0 (no darkening).
    // ambient_min is set above 0.5 because the mesher already
    // bakes face-direction shading into vertex color (NegY pressed
    // to 0.6) — stacking aggressive AO on top would tip dark
    // corners into near-black.
    let ambient_min = 0.5;
    let ao_factor = ambient_min + (1.0 - ambient_min) * in.ao;

    // Apply lighting + AO to color
    var result = in.color.rgb * lighting * ao_factor;

    // Simple fog based on distance from camera. Tuned for ~256³
    // scenes — fog stays out of the way at typical editing zoom
    // and only kicks in when you really pull back. Editor's mouse-
    // raycast (in app/input.rs) uses the same scale so anything
    // visibly clear is also click-reachable.
    let camera_pos = camera.camera_pos.xyz;
    let dist = length(in.world_position - camera_pos);
    let fog_start = 200.0;
    let fog_end = 800.0;
    let fog_color = vec3<f32>(0.1, 0.1, 0.15);
    let fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    result = mix(result, fog_color, fog_factor);

    return vec4<f32>(result, in.color.a);
}
