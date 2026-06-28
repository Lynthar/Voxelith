// VoxelithUberURP.shader — reference Universal Render Pipeline (URP) shader
// for consuming a Voxelith-exported .glb. Reference implementation for
// docs/GAME_PIPELINE_ROADMAP.md §3.2 (the engine consumption contract).
//
// What it reads (the §3.2 attribute contract):
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
// IMPORTANT (§3.2 open risk): glTFast prunes UV channels that no material
// samples. This shader SAMPLES TEXCOORD_0, but it must be assigned to the
// mesh *before/while* importing (set it as the import material, or use a
// glTFast import callback) so the channel is kept. If the zone arrives as
// all-zero, the UV0 channel was pruned — see roadmap §3.2 for the Blender-
// bridge fallback. Verified working as of: <fill in glTFast version>.
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
