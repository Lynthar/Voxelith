// The glTFast end-to-end gate: the tint zone that src/io/gltf.rs mirrors into
// TEXCOORD_0.x must still be in the Unity mesh's UV0 after a real glTFast
// import. glTFast is the outside source of truth here — nothing in this file
// may assert against Voxelith's own code. Driven by tools/gltfast-gate/run.py.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using GLTFast;
using UnityEditor;
using UnityEngine;

static class GltfastGate
{
    const string ImportDir = "Assets/GltfastGate";

    static int s_Failures;

    public static async void Run()
    {
        var code = 1;
        try
        {
            code = await Check();
        }
        catch (Exception e)
        {
            Console.WriteLine("GATE ERROR " + e);
        }
        EditorApplication.Exit(code);
    }

    static async Task<int> Check()
    {
        var path = Arg("-glb");
        if (path == null)
        {
            Console.WriteLine("GATE ERROR missing -glb <file>");
            return 2;
        }
        var wanted = (Arg("-zones") ?? "1,2,3").Split(',').Select(int.Parse).ToArray();

        var pkg = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(GltfImport).Assembly);
        var pipeline = UnityEngine.Rendering.GraphicsSettings.defaultRenderPipeline;
        Console.WriteLine($"glTFast {pkg?.version ?? "?"} | Unity {Application.unityVersion} | "
            + $"pipeline {(pipeline == null ? "built-in" : pipeline.GetType().Name)}\n");

        // Both ways a project consumes a .glb: dropped into Assets/ (a
        // ScriptedImporter, which is what the reference shader's procedure
        // describes) and loaded at runtime through the GltfImport API.
        Meshes("editor import", ImportAsAsset(path), wanted);
        Meshes("runtime import", await ImportAtRuntime(path), wanted);

        AssetDatabase.DeleteAsset(ImportDir);
        Console.WriteLine(s_Failures == 0 ? "\nGATE PASS" : $"\n{s_Failures} GATE FAILURES");
        return s_Failures == 0 ? 0 : 1;
    }

    static Mesh[] ImportAsAsset(string glb)
    {
        Directory.CreateDirectory(ImportDir);
        var dst = $"{ImportDir}/gate.glb";
        File.Copy(glb, dst, true);
        AssetDatabase.ImportAsset(dst, ImportAssetOptions.ForceSynchronousImport);
        return AssetDatabase.LoadAllAssetsAtPath(dst).OfType<Mesh>().ToArray();
    }

    static async Task<Mesh[]> ImportAtRuntime(string glb)
    {
        // UninterruptedDeferAgent: the default one spreads work over frames via
        // a DontDestroyOnLoad object, which an editor script may not create.
        var gltf = new GltfImport(deferAgent: new UninterruptedDeferAgent());
        var loaded = await gltf.LoadFile(glb);
        Check("runtime import: the .glb loads", loaded, glb);
        return loaded ? gltf.GetMeshes() ?? Array.Empty<Mesh>() : Array.Empty<Mesh>();
    }

    static void Meshes(string via, Mesh[] meshes, int[] wanted)
    {
        Check($"{via}: produced meshes", meshes.Length > 0);
        var zones = new HashSet<int>();
        foreach (var mesh in meshes)
        {
            var uv = mesh.uv;
            // Short-circuit: with no UV0 every check below would pass vacuously.
            var survived = uv != null && uv.Length == mesh.vertexCount && uv.Length > 0;
            Check($"{via}: [{mesh.name}] UV0 survived", survived,
                $"{uv?.Length ?? 0} uvs for {mesh.vertexCount} vertices");
            if (!survived) continue;
            foreach (var v in uv) zones.Add(Mathf.RoundToInt(v.x));
            // Only .x is load-bearing: glTFast flips V for Unity's texture
            // origin, so a zone carried in .y would come back as 1 - zone.
            Check($"{via}: [{mesh.name}] UV0.x holds whole numbers",
                uv.All(v => Mathf.Abs(v.x - Mathf.Round(v.x)) < 0.001f));
        }
        var got = string.Join(",", zones.OrderBy(z => z));
        foreach (var zone in wanted) Check($"{via}: zone {zone} reached UV0.x", zones.Contains(zone), $"got [{got}]");
    }

    static void Check(string what, bool ok, string detail = null)
    {
        if (!ok) s_Failures++;
        Console.WriteLine($"{(ok ? "PASS" : "FAIL")} {what}{(ok || detail == null ? "" : ": " + detail)}");
    }

    static string Arg(string name)
    {
        var args = Environment.GetCommandLineArgs();
        var i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }
}
