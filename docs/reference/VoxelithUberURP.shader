// VoxelithUberURP.shader — reference Universal Render Pipeline (URP) shader
// for consuming a Voxelith-exported .glb. This file *is* the engine
// consumption contract: what a stock URP project has to do to read
// everything src/io/gltf.rs writes.
//
// What it reads:
//   COLOR_0    (vec4) — voxel RGBA with per-vertex AO pre-multiplied into
//                       RGB (linear). Used directly as albedo.
//   TEXCOORD_0 (vec2) — .x carries the faction tint zone (0..3); the
//                       glTFast-readable mirror of the custom _TINTZONE
//                       attribute (glTFast drops custom attributes). .y is 0.
//   material        — the export splits geometry into plain / metallic /
//                       emissive primitives; assign this shader per material
//                       and set Metallic / Emission to match.
//
// Faction recolor: per-vertex zone 0 = no tint, 1 = primary, 2 = secondary,
// 3 = reserved. Final albedo = COLOR_0.rgb * _BaseColor * zoneTint(zone).
// Set _PrimaryColor/_SecondaryColor per faction at runtime (e.g. via
// MaterialPropertyBlock) to recolor the same mesh for different teams.
//
// Scene layout: a plain export puts the mesh and socket nodes at the scene
// root. A *placed* export (`voxelith bake` with any non-identity pivot,
// up-axis or unit-scale) wraps them in one extra node named
// "Voxelith_root" carrying the placement transform; children keep their
// own transforms and vertex data is never rewritten, so an identity
// placement is byte-identical to a plain export. Three consequences:
//   - Instantiate the whole scene, not a hand-picked node. Reading "the
//     first mesh node" skips the wrapper and puts the asset at the wrong
//     scale and origin. glTFast, Godot and Blender all apply it when the
//     scene is imported as a whole.
//   - A socket's world position comes from the full parent chain, wrapper
//     included. Compose ancestor transforms (or read the engine's world
//     matrix); a socket's local translation is not model space.
//   - The wrapper's presence is not a stable signal — it appears only when
//     a placement was requested. Key on the name if you must find it;
//     "Voxelith_root" is reserved and never used for geometry or sockets.
//
// The tint zone rides in TEXCOORD_0 because glTFast drops custom attributes
// like _TINTZONE, and glTFast prunes UV channels no material samples, which
// would take the mirror with it. This shader SAMPLES TEXCOORD_0, so keep a
// material that reaches the mesh at import time. If every zone arrives as 0,
// UV0 was pruned.
//
// tools/gltfast-gate/ settles the import half automatically: it bakes a
// fixture carrying zones 1-3 (no gltfpack, so a compression bug cannot pass
// for an import bug), hands it to real glTFast both as an Asset and through
// the runtime API, and fails if UV0 is missing or a zone never arrives.
// Verified as of glTFast 6.20.0 on Unity 6000.3.19f1: UV0 survives both
// import paths with no material assigned at import time.
//
// The gate cannot see your materials, so confirm the rest by eye once:
//   1. `python tools/gltfast-gate/run.py` - red here means the zone never
//      reaches Unity at all, and nothing below will help.
//   2. Import a baked .glb into your Unity 6 URP project, assign this shader
//      to the submeshes, set _PrimaryColor red and _SecondaryColor blue.
//   3. PASS: zone-1 voxels render red, zone-2 blue.
//   4. FAIL (all one colour) with the gate green: your import pruned UV0. In
//      preference order - assign the material *at import* (import material
//      remap, or an IMaterialGenerator callback) so a material samples UV0;
//      or a thin import callback copying UV0 into a second vertex-colour
//      channel before pruning; or route through Blender, which preserves all
//      attributes.
// Until step 3 passes in your project, treat per-zone tint as unproven -
// whole-model _BaseColor tint always works.
//
// Targets URP 12–17 (Unity 2022 LTS .. Unity 6). Stable URP HLSL APIs only.

Shader "Voxelith/VoxelithUberURP"
{
    Properties
    {
        [Header(Base)]
        _BaseColor       ("Base Tint (multiplies all)", Color) = (1,1,1,1)

        [Header(Faction Tint Zones)]
        _PrimaryColor    ("Primary (zone 1)",   Color) = (1,1,1,1)
        _SecondaryColor  ("Secondary (zone 2)", Color) = (1,1,1,1)
        _ReservedColor   ("Reserved (zone 3)",  Color) = (1,1,1,1)

        [Header(PBR)]
        _Metallic        ("Metallic", Range(0,1)) = 0
        _Smoothness      ("Smoothness", Range(0,1)) = 0.5

        [Header(Emission)]
        [HDR] _EmissionColor ("Emission Color", Color) = (0,0,0,0)
        // Multiplies emission by COLOR_0.rgb so emissive voxels glow in
        // their own color (core glTF emissiveFactor can't be per-vertex).
        [Toggle(_EMISSION_TINT_BY_VERTEX)] _EmissionTintByVertex ("Emission x Vertex Color", Float) = 1
    }

    SubShader
    {
        Tags
        {
            "RenderType"       = "Opaque"
            "RenderPipeline"   = "UniversalPipeline"
            "Queue"            = "Geometry"
            "UniversalMaterialType" = "Lit"
        }
        LOD 300

        // ------------------------------------------------------------------
        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex   vert
            #pragma fragment frag

            // URP lighting keywords.
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile_fragment _ _SCREEN_SPACE_OCCLUSION
            #pragma multi_compile _ LIGHTMAP_ON
            #pragma multi_compile_fog
            #pragma multi_compile_instancing

            #pragma shader_feature_local_fragment _EMISSION_TINT_BY_VERTEX

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float4 _PrimaryColor;
                float4 _SecondaryColor;
                float4 _ReservedColor;
                float  _Metallic;
                float  _Smoothness;
                float4 _EmissionColor;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                float4 color      : COLOR;        // COLOR_0 (RGBA, AO in RGB)
                float2 uv         : TEXCOORD0;    // .x = tint zone
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 positionWS  : TEXCOORD0;
                float3 normalWS    : TEXCOORD1;
                float4 color       : TEXCOORD2;
                float2 uv          : TEXCOORD3;
                float  fogFactor   : TEXCOORD4;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // Select the faction tint for a per-vertex zone (0..3).
            half3 ZoneTint(float zoneRaw)
            {
                int zone = (int)round(zoneRaw);
                if (zone == 1) return _PrimaryColor.rgb;
                if (zone == 2) return _SecondaryColor.rgb;
                if (zone == 3) return _ReservedColor.rgb;
                return half3(1, 1, 1); // zone 0: no faction tint
            }

            Varyings vert (Attributes IN)
            {
                Varyings OUT = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_TRANSFER_INSTANCE_ID(IN, OUT);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                VertexPositionInputs pos = GetVertexPositionInputs(IN.positionOS.xyz);
                VertexNormalInputs   nrm = GetVertexNormalInputs(IN.normalOS);

                OUT.positionHCS = pos.positionCS;
                OUT.positionWS  = pos.positionWS;
                OUT.normalWS    = nrm.normalWS;
                OUT.color       = IN.color;
                OUT.uv          = IN.uv;
                OUT.fogFactor   = ComputeFogFactor(pos.positionCS.z);
                return OUT;
            }

            half4 frag (Varyings IN) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                // Albedo: vertex color (AO already baked into RGB) × base
                // tint × per-zone faction tint.
                half3 albedo = IN.color.rgb * _BaseColor.rgb * ZoneTint(IN.uv.x);

                // Emission: white emissiveFactor on the emissive material;
                // optionally tinted by the voxel color so it glows in-hue.
                half3 emission = _EmissionColor.rgb;
                #ifdef _EMISSION_TINT_BY_VERTEX
                    emission *= IN.color.rgb;
                #endif

                SurfaceData surface = (SurfaceData)0;
                surface.albedo     = albedo;
                surface.metallic   = _Metallic;
                surface.smoothness = _Smoothness;
                surface.occlusion  = 1.0; // AO is baked into albedo, not a map
                surface.emission   = emission;
                surface.alpha      = 1.0;

                InputData inputData = (InputData)0;
                inputData.positionWS        = IN.positionWS;
                inputData.normalWS          = normalize(IN.normalWS);
                inputData.viewDirectionWS   = GetWorldSpaceNormalizeViewDir(IN.positionWS);
                inputData.shadowCoord       = TransformWorldToShadowCoord(IN.positionWS);
                inputData.fogCoord          = IN.fogFactor;
                inputData.normalizedScreenSpaceUV = GetNormalizedScreenSpaceUV(IN.positionHCS);

                half4 color = UniversalFragmentPBR(inputData, surface);
                color.rgb = MixFog(color.rgb, IN.fogFactor);
                return color;
            }
            ENDHLSL
        }

        // ------------------------------------------------------------------
        // Shadow casting — lets these assets cast shadows in URP.
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On
            ZTest LEqual
            ColorMask 0
            Cull Back

            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex   shadowVert
            #pragma fragment shadowFrag
            #pragma multi_compile_instancing
            #pragma multi_compile _ _CASTING_PUNCTUAL_LIGHT_SHADOW

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

            float3 _LightDirection;
            float3 _LightPosition;

            struct ShadowAttributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct ShadowVaryings
            {
                float4 positionHCS : SV_POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            float4 GetShadowPositionHClip(ShadowAttributes IN)
            {
                float3 positionWS = TransformObjectToWorld(IN.positionOS.xyz);
                float3 normalWS   = TransformObjectToWorldNormal(IN.normalOS);

                #if _CASTING_PUNCTUAL_LIGHT_SHADOW
                    float3 lightDirectionWS = normalize(_LightPosition - positionWS);
                #else
                    float3 lightDirectionWS = _LightDirection;
                #endif

                float4 positionCS = TransformWorldToHClip(ApplyShadowBias(positionWS, normalWS, lightDirectionWS));
                #if UNITY_REVERSED_Z
                    positionCS.z = min(positionCS.z, UNITY_NEAR_CLIP_VALUE);
                #else
                    positionCS.z = max(positionCS.z, UNITY_NEAR_CLIP_VALUE);
                #endif
                return positionCS;
            }

            ShadowVaryings shadowVert (ShadowAttributes IN)
            {
                ShadowVaryings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                OUT.positionHCS = GetShadowPositionHClip(IN);
                return OUT;
            }

            half4 shadowFrag (ShadowVaryings IN) : SV_Target
            {
                return 0;
            }
            ENDHLSL
        }
    }

    FallBack "Universal Render Pipeline/Lit"
}
