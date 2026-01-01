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
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_position = in.position;
    out.normal = in.normal;
    out.color = in.color;
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

    // Apply lighting to color
    var result = in.color.rgb * lighting;

    // Simple fog based on distance from camera
    let camera_pos = camera.camera_pos.xyz;
    let dist = length(in.world_position - camera_pos);
    let fog_start = 100.0;
    let fog_end = 300.0;
    let fog_color = vec3<f32>(0.1, 0.1, 0.15);
    let fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    result = mix(result, fog_color, fog_factor);

    return vec4<f32>(result, in.color.a);
}
