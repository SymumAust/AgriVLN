// FieldWorld.cs  -  open field + crop beds + one test obstacle + a small robot, with the same lock-step bridge to Python
//                   as OrchardWorld.cs. Built for the go / slow / stop / detour task (professor's plan, Nov 2026).
//
// HOW TO USE (Unity 6, URP project, the same OrchardLoop project):
//   1. Put this file in Assets/Scripts/ next to OrchardWorld.cs.
//   2. File > New Scene > "Basic (URP)" > Create, save it as Assets/Scenes/Field.unity.
//   3. Hierarchy: right-click > Create Empty, name it "Field", Inspector > Add Component > FieldWorld.
//   4. Press Play. You see the field from behind the robot (W/S/A/D drives it when no Python is connected).
//      Python (field_client.py) chooses the scenario on every reset, so the Inspector "Scenario" is only the first view.
//
// World: Y-up. The robot drives along +Z on a path centred at x = 0. Crop beds (long green boxes, cropHeight high) run
// along Z on both sides; the first bed's inner edge is cropOffset from the path centre. The free strip between the
// path and the crops is the "side space" for a detour (driving into the crops = crop damage, not allowed).
//
// Scenarios (one obstacle at z = obs_z, x = obs_x):
// Camera height and tilt can also be set by Python on every reset (cam_h > 0), for the camera-mounting experiment.
//
//   none            nothing on the path                                    -> correct: go all the way
//   person_stand    capsule 1.7 m tall, 0.5 m wide                          -> stop (people: always stop)
//   person_lie      capsule lying across the path, 1.7 m long, 0.4 m high   -> stop
//   person_moving   standing capsule that walks in from the left at 1 m/s when the robot is movingTrigger seconds away
//   rock            0.6 m wide, 0.42 m high                                 -> detour if side space, else stop
//   drum            0.6 m wide, 0.9 m high                                  -> detour if side space, else stop
//   ditch           1 m long gap across the whole field, 1 m deep           -> stop (terrain hazard)
//   grass           2.4 x 1.5 m patch of 0.35 m blades, PASSABLE            -> go (hard-but-go: sensors see it, robot can cross)
//
// Every reply: JSON header (pose, camera, ground-truth objects) + JPEG (if asked) + depth grid (if asked):
//   depth grid = depthW x depthH rays through the image (row 0 = top), z-depth along the camera axis like a stereo camera,
//   0 = no valid depth (sky or beyond 30 m). float32 little-endian.

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

public class FieldWorld : MonoBehaviour
{
    [Header("Field")]
    public float pathLength = 80f;
    public float cropOffset = 2.0f;         // inner edge of the first crop bed, measured from the path centre (m)
    public float cropHeight = 0.5f;
    public int cropBedsEachSide = 4;
    public string scenario = "person_stand";
    public float obstacleZ = 30f;
    public float obstacleX = 0f;
    public int seed = 1;

    [Header("Robot")]
    public float startZ = 12f;
    public float camHeight = 0.28f;
    public float camTiltDeg = 0f;           // positive = camera looks down
    public float hFovDeg = 102f;
    public int imgW = 640, imgH = 360;
    public int depthW = 64, depthH = 36;
    public float movingTrigger = 5f;        // person_moving starts walking when the robot is this many seconds away
    public float walkSpeed = 1.0f;

    [Header("Bridge to Python")]
    public int port = 5555;

    const float FrontOffset = 0.215f;       // robot front bumper ahead of the robot centre (Jackal-sized body)

    // --- state
    Transform robot, sceneRoot; Camera robotCam; RenderTexture rt; Texture2D readTex;
    float px, pz, psi, simTime, lastV;
    string curScenario; float curObsZ, curObsX;
    bool clientConnected;
    Material groundMat, pathMat, leafMat, personMat, rockMat, drumMat, grassMat, ditchMat;

    class Obj { public string kind; public float x, z, hw, hl, h; public bool pass; public Transform tr; }
    readonly List<Obj> objs = new List<Obj>();
    Obj mover; bool moverWalking; float moverTargetX;
    string hudText = "", hudLabel = "";     // live view: what Python decided (drawn on the Game view only, never in the robot camera)
    Material lightMat;

    class Req { public string cmd; public float v, w, dt, x0, z0, yaw, obs_z, obs_x, cam_h, tilt; public bool image, depth; public string scenario, hud, label; }
    class Job { public Req req; public byte[] reply; public ManualResetEventSlim done = new ManualResetEventSlim(false); }
    readonly ConcurrentQueue<Job> jobs = new ConcurrentQueue<Job>();
    bool busy;
    Thread server; volatile bool running;

    // ------------------------------------------------------------------ build
    void Start()
    {
        Application.runInBackground = true;
        groundMat = Mat(new Color(0.58f, 0.48f, 0.36f), 0.25f, 60f);
        pathMat = Mat(new Color(0.46f, 0.37f, 0.27f), 0.25f, 20f);
        leafMat = Mat(new Color(0.20f, 0.45f, 0.15f), 0.45f, 8f);
        grassMat = Mat(new Color(0.42f, 0.60f, 0.22f), 0.2f, 1f);
        personMat = Mat(new Color(0.85f, 0.35f, 0.20f), 0f, 1f);
        rockMat = Mat(new Color(0.45f, 0.45f, 0.43f), 0.3f, 2f);
        drumMat = Mat(new Color(0.15f, 0.30f, 0.65f), 0f, 1f);
        ditchMat = Mat(new Color(0.30f, 0.24f, 0.18f), 0.25f, 10f);
        BuildRobot();
        BuildScene(scenario, obstacleZ, obstacleX);
        SetPose(0, startZ, 0);
        Debug.Log($"FieldWorld: scenario {scenario} at z = {obstacleZ} m, crops from |x| = {cropOffset} m. Bridge on port {port}.");
        running = true;
        server = new Thread(ServerLoop) { IsBackground = true }; server.Start();
    }

    Material Mat(Color c, float noise, float tiling)
    {
        var m = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        m.SetColor("_BaseColor", c); m.SetFloat("_Smoothness", 0.05f);
        if (noise > 0) { m.SetTexture("_BaseMap", NoiseTexture(Color.white, noise)); m.SetTextureScale("_BaseMap", new Vector2(tiling, tiling)); }
        return m;
    }

    GameObject Prim(PrimitiveType t, string name, Vector3 pos, Vector3 scale, Material m, Quaternion rot)
    {
        var g = GameObject.CreatePrimitive(t);
        g.name = name; g.transform.SetParent(sceneRoot);
        g.transform.position = pos; g.transform.rotation = rot; g.transform.localScale = scale;
        g.GetComponent<Renderer>().sharedMaterial = m;
        return g;
    }

    void BuildScene(string sc, float oz, float ox)
    {
        if (sceneRoot != null) DestroyImmediate(sceneRoot.gameObject);   // immediate, so old colliders never answer a ray
        sceneRoot = new GameObject("Scene").transform;
        objs.Clear(); mover = null; moverWalking = false;
        curScenario = sc; curObsZ = oz; curObsX = ox;
        UnityEngine.Random.InitState(seed);
        var flat = Quaternion.identity;

        // ground: one big slab, or two slabs with a 1 m gap (the ditch) and a floor 1 m below
        float z0 = -100f, z1 = 300f;
        if (sc == "ditch")
        {
            Slab(z0, oz - 0.5f); Slab(oz + 0.5f, z1);
            Prim(PrimitiveType.Cube, "ditch", new Vector3(0, -1.1f, oz), new Vector3(400, 0.2f, 1.0f), ditchMat, flat);
            objs.Add(new Obj { kind = "ditch", x = 0, z = oz, hw = 200, hl = 0.5f, h = -1f, pass = false });
        }
        else Slab(z0, z1);

        // path strip (visual only, no collider) and crop beds
        PathStrip(0, pathLength, sc == "ditch" ? oz : float.NaN);
        for (int side = -1; side <= 1; side += 2)
            for (int k = 0; k < cropBedsEachSide; k++)
            {
                float x = side * (cropOffset + 0.3f + 1.2f * k);
                Prim(PrimitiveType.Cube, "crop", new Vector3(x, cropHeight / 2, pathLength / 2), new Vector3(0.6f, cropHeight, pathLength), leafMat, flat);
            }

        // the obstacle
        switch (sc)
        {
            case "person_stand":
            case "person_moving":
            {
                float sx = sc == "person_moving" ? ox - 3f : ox;
                var g = Prim(PrimitiveType.Capsule, "person", new Vector3(sx, 0.85f, oz), new Vector3(0.5f, 0.85f, 0.5f), personMat, flat);
                var o = new Obj { kind = sc == "person_moving" ? "person_moving" : "person", x = sx, z = oz, hw = 0.25f, hl = 0.25f, h = 1.7f, pass = false, tr = g.transform };
                objs.Add(o);
                if (sc == "person_moving") { mover = o; moverTargetX = ox; }
                break;
            }
            case "person_lie":
            {
                var g = Prim(PrimitiveType.Capsule, "person", new Vector3(ox, 0.2f, oz), new Vector3(0.4f, 0.85f, 0.4f), personMat, Quaternion.Euler(0, 0, 90));
                objs.Add(new Obj { kind = "person_lying", x = ox, z = oz, hw = 0.85f, hl = 0.2f, h = 0.4f, pass = false, tr = g.transform });
                break;
            }
            case "rock":
            {
                var g = Prim(PrimitiveType.Sphere, "rock", new Vector3(ox, 0.19f, oz), new Vector3(0.6f, 0.46f, 0.6f), rockMat, flat);
                objs.Add(new Obj { kind = "rock", x = ox, z = oz, hw = 0.3f, hl = 0.3f, h = 0.42f, pass = false, tr = g.transform });
                break;
            }
            case "drum":
            {
                var g = Prim(PrimitiveType.Cylinder, "drum", new Vector3(ox, 0.45f, oz), new Vector3(0.6f, 0.45f, 0.6f), drumMat, flat);
                objs.Add(new Obj { kind = "drum", x = ox, z = oz, hw = 0.3f, hl = 0.3f, h = 0.9f, pass = false, tr = g.transform });
                break;
            }
            case "grass":
            {
                var patch = new GameObject("grass_patch").transform; patch.SetParent(sceneRoot);
                for (int i = 0; i < 500; i++)
                {
                    float bx = ox + UnityEngine.Random.Range(-1.2f, 1.2f), bz = oz + UnityEngine.Random.Range(-0.75f, 0.75f);
                    float bh = UnityEngine.Random.Range(0.25f, 0.4f);
                    var b = Prim(PrimitiveType.Cube, "grass", new Vector3(bx, bh / 2, bz), new Vector3(0.03f, bh, 0.03f), grassMat,
                                 Quaternion.Euler(UnityEngine.Random.Range(-15f, 15f), UnityEngine.Random.Range(0f, 180f), UnityEngine.Random.Range(-15f, 15f)));
                    b.transform.SetParent(patch, true);
                }
                objs.Add(new Obj { kind = "grass", x = ox, z = oz, hw = 1.2f, hl = 0.75f, h = 0.35f, pass = true, tr = patch });
                break;
            }
        }
        Physics.SyncTransforms();
    }

    void Slab(float za, float zb)
    {
        Prim(PrimitiveType.Cube, "ground", new Vector3(0, -0.1f, (za + zb) / 2), new Vector3(400, 0.2f, zb - za), groundMat, Quaternion.identity);
    }

    void PathStrip(float za, float zb, float gapZ)
    {
        var segs = new List<Vector2>();
        if (float.IsNaN(gapZ)) segs.Add(new Vector2(za, zb));
        else { segs.Add(new Vector2(za, gapZ - 0.5f)); segs.Add(new Vector2(gapZ + 0.5f, zb)); }
        foreach (var s in segs)
        {
            var q = Prim(PrimitiveType.Quad, "path", new Vector3(0, 0.003f, (s.x + s.y) / 2), new Vector3(1.2f, s.y - s.x, 1), pathMat, Quaternion.Euler(90, 0, 0));
            DestroyImmediate(q.GetComponent<Collider>());
        }
    }

    void BuildRobot()
    {
        robot = new GameObject("Robot").transform;
        var yellow = new Material(Shader.Find("Universal Render Pipeline/Lit")); yellow.SetColor("_BaseColor", new Color(0.95f, 0.75f, 0.1f));
        var black = new Material(Shader.Find("Universal Render Pipeline/Lit")); black.SetColor("_BaseColor", new Color(0.08f, 0.08f, 0.08f));
        var body = GameObject.CreatePrimitive(PrimitiveType.Cube);
        body.transform.SetParent(robot); body.transform.localPosition = new Vector3(0, 0.17f, 0); body.transform.localScale = new Vector3(0.30f, 0.15f, 0.43f);
        body.GetComponent<Renderer>().sharedMaterial = yellow;
        foreach (var (sx, sz) in new[] { (1, 1), (-1, 1), (1, -1), (-1, -1) })
        {
            var w = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            w.transform.SetParent(robot); w.transform.localPosition = new Vector3(sx * 0.185f, 0.098f, sz * 0.131f);
            w.transform.localRotation = Quaternion.Euler(0, 0, 90); w.transform.localScale = new Vector3(0.196f, 0.02f, 0.196f);
            w.GetComponent<Renderer>().sharedMaterial = black;
        }
        var lamp = GameObject.CreatePrimitive(PrimitiveType.Cube);          // status light: green go, yellow slow, red stop, blue detour
        lamp.transform.SetParent(robot); lamp.transform.localPosition = new Vector3(0, 0.3f, -0.13f); lamp.transform.localScale = new Vector3(0.12f, 0.08f, 0.08f);
        lightMat = new Material(Shader.Find("Universal Render Pipeline/Lit")); lightMat.SetColor("_BaseColor", Color.gray);
        lightMat.EnableKeyword("_EMISSION"); lamp.GetComponent<Renderer>().sharedMaterial = lightMat;
        foreach (var c in robot.GetComponentsInChildren<Collider>()) Destroy(c);
        foreach (var t in robot.GetComponentsInChildren<Transform>()) t.gameObject.layer = 2;   // Ignore Raycast

        var camGo = new GameObject("RobotCamera"); camGo.transform.SetParent(robot);
        camGo.transform.localPosition = new Vector3(0, camHeight, 0.2f);
        camGo.transform.localRotation = Quaternion.Euler(camTiltDeg, 0, 0);
        robotCam = camGo.AddComponent<Camera>();
        rt = new RenderTexture(imgW, imgH, 24); robotCam.targetTexture = rt;
        robotCam.aspect = imgW / (float)imgH;
        robotCam.fieldOfView = 2f * Mathf.Atan(Mathf.Tan(hFovDeg * Mathf.Deg2Rad / 2f) * imgH / imgW) * Mathf.Rad2Deg;
        robotCam.nearClipPlane = 0.05f; robotCam.farClipPlane = 300f;
        readTex = new Texture2D(imgW, imgH, TextureFormat.RGB24, false);
    }

    void SetPose(float x, float z, float heading)
    {
        px = x; pz = z; psi = heading;
        robot.position = new Vector3(px, 0, pz);
        robot.rotation = Quaternion.Euler(0, psi * Mathf.Rad2Deg, 0);
        Physics.SyncTransforms();
    }

    // drive with (v forward m/s, w = left-turn rate rad/s) for dt simulated seconds; the moving person walks in the same clock
    void Integrate(float v, float w, float dt)
    {
        lastV = v;
        float h = 0.01f;
        for (float t = 0; t < dt - 1e-6f; t += h)
        {
            float step = Mathf.Min(h, dt - t);
            psi -= w * step;
            px += v * Mathf.Sin(psi) * step; pz += v * Mathf.Cos(psi) * step;
            if (mover != null)
            {
                float gap = curObsZ - (pz + FrontOffset);
                if (!moverWalking && v > 0.05f && gap <= movingTrigger * v) moverWalking = true;
                if (moverWalking && mover.x < moverTargetX)
                {
                    mover.x = Mathf.Min(moverTargetX, mover.x + walkSpeed * step);
                    mover.tr.position = new Vector3(mover.x, mover.tr.position.y, mover.z);
                }
            }
        }
        simTime += dt;
        SetPose(px, pz, psi);
    }

    // ------------------------------------------------------------------ depth grid (like a stereo depth image, coarse)
    byte[] DepthGrid()
    {
        var d = new float[depthW * depthH];
        Vector3 o = robotCam.transform.position, f = robotCam.transform.forward;
        for (int j = 0; j < depthH; j++)
            for (int i = 0; i < depthW; i++)
            {
                float u = (i + 0.5f) / depthW, v = 1f - (j + 0.5f) / depthH;        // row 0 = top of the image
                Ray ray = robotCam.ViewportPointToRay(new Vector3(u, v, 0));
                float z = 0f;
                if (Physics.Raycast(ray, out RaycastHit hit, 40f, ~(1 << 2)))
                {
                    z = Vector3.Dot(hit.point - o, f);
                    if (z < 0.05f || z > 30f) z = 0f;
                }
                d[j * depthW + i] = z;
            }
        var b = new byte[d.Length * 4];
        Buffer.BlockCopy(d, 0, b, 0, b.Length);
        return b;
    }

    // ------------------------------------------------------------------ main thread
    void Update()
    {
        if (!busy && jobs.TryDequeue(out Job job)) { busy = true; StartCoroutine(Handle(job)); }
        if (!clientConnected) KeyboardDrive();
    }

    void LateUpdate()
    {
        var cam = Camera.main;
        if (cam != null && cam != robotCam)
        {
            cam.transform.position = robot.position + robot.rotation * new Vector3(0, 1.8f, -3.5f);
            cam.transform.LookAt(robot.position + robot.rotation * new Vector3(0, 0.3f, 2.5f));
        }
    }

    void KeyboardDrive()
    {
        float v = 0, w = 0;
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb != null)
        {
            if (kb.wKey.isPressed) v += 1.2f; if (kb.sKey.isPressed) v -= 0.6f;
            if (kb.aKey.isPressed) w += 0.8f; if (kb.dKey.isPressed) w -= 0.8f;
        }
#else
        if (Input.GetKey(KeyCode.W)) v += 1.2f; if (Input.GetKey(KeyCode.S)) v -= 0.6f;
        if (Input.GetKey(KeyCode.A)) w += 0.8f; if (Input.GetKey(KeyCode.D)) w -= 0.8f;
#endif
        if (v != 0 || w != 0) Integrate(v, w, Time.deltaTime);
    }

    IEnumerator Handle(Job job)
    {
        var q = job.req;
        if (q.cmd == "reset")
        {
            string sc = string.IsNullOrEmpty(q.scenario) ? curScenario : q.scenario;
            BuildScene(sc, q.obs_z, q.obs_x);
            if (q.cam_h > 0)                                   // Python may move the camera: height (m) and tilt (deg, + = down)
            {
                camHeight = q.cam_h; camTiltDeg = q.tilt;
                robotCam.transform.localPosition = new Vector3(0, camHeight, 0.2f);
                robotCam.transform.localRotation = Quaternion.Euler(camTiltDeg, 0, 0);
            }
            SetPose(q.x0, q.z0, q.yaw); simTime = 0; lastV = 0;
        }
        else if (q.cmd == "step" && q.dt > 0) Integrate(q.v, q.w, q.dt);

        if (q.hud != null) hudText = q.hud;
        if (q.label != null) { hudLabel = q.label; var lc = LabelColor(hudLabel); lightMat.SetColor("_BaseColor", lc); lightMat.SetColor("_EmissionColor", lc * 1.5f); }
        byte[] depth = q.depth ? DepthGrid() : null;
        byte[] jpg = null;
        if (q.image)
        {
            yield return new WaitForEndOfFrame();
            var prev = RenderTexture.active; RenderTexture.active = rt;
            readTex.ReadPixels(new Rect(0, 0, imgW, imgH), 0, 0); readTex.Apply();
            RenderTexture.active = prev;
            jpg = readTex.EncodeToJPG(92);
        }

        var ci = CultureInfo.InvariantCulture;
        Vector3 c = robotCam.transform.position;
        var sb = new StringBuilder();
        sb.Append("{");
        sb.AppendFormat(ci, "\"t\":{0:F3},\"x\":{1:F4},\"z\":{2:F4},\"yaw\":{3:F5},\"cam_x\":{4:F4},\"cam_y\":{5:F4},\"cam_z\":{6:F4},",
                        simTime, px, pz, psi, c.x, c.y, c.z);
        sb.AppendFormat(ci, "\"front_offset\":{0:F3},\"cam_height\":{1:F3},\"tilt_deg\":{2:F2},\"hfov_deg\":{3:F2},\"img_w\":{4},\"img_h\":{5},",
                        FrontOffset, camHeight, camTiltDeg, hFovDeg, imgW, imgH);
        sb.AppendFormat(ci, "\"depth_w\":{0},\"depth_h\":{1},\"crop_offset\":{2:F3},\"scenario\":\"{3}\",\"obs_z\":{4:F3},\"obs_x\":{5:F3},\"objects\":[",
                        depthW, depthH, cropOffset, curScenario, curObsZ, curObsX);
        for (int i = 0; i < objs.Count; i++)
        {
            var o = objs[i];
            if (i > 0) sb.Append(",");
            sb.AppendFormat(ci, "{{\"kind\":\"{0}\",\"x\":{1:F3},\"z\":{2:F3},\"hw\":{3:F3},\"hl\":{4:F3},\"h\":{5:F3},\"pass\":{6}}}",
                            o.kind, o.x, o.z, o.hw, o.hl, o.h, o.pass ? "true" : "false");
        }
        sb.Append("]}");
        byte[] hb = Encoding.UTF8.GetBytes(sb.ToString());
        var ms = new MemoryStream();
        WriteInt(ms, hb.Length); ms.Write(hb, 0, hb.Length);
        WriteInt(ms, jpg == null ? 0 : jpg.Length); if (jpg != null) ms.Write(jpg, 0, jpg.Length);
        WriteInt(ms, depth == null ? 0 : depth.Length); if (depth != null) ms.Write(depth, 0, depth.Length);
        job.reply = ms.ToArray();
        job.done.Set();
        busy = false;
    }

    static Color LabelColor(string l)
    {
        switch (l)
        {
            case "go": return new Color(0.2f, 0.8f, 0.3f);
            case "slow": return new Color(1.0f, 0.75f, 0.1f);
            case "stop": return new Color(0.9f, 0.15f, 0.1f);
            case "detour": return new Color(0.2f, 0.5f, 1.0f);
            default: return Color.gray;
        }
    }

    void OnGUI()
    {
        if (string.IsNullOrEmpty(hudLabel) && string.IsNullOrEmpty(hudText)) return;
        var big = new GUIStyle(GUI.skin.label) { fontSize = 40, fontStyle = FontStyle.Bold };
        big.normal.textColor = LabelColor(hudLabel);
        var small = new GUIStyle(GUI.skin.label) { fontSize = 20 };
        small.normal.textColor = Color.white;
        GUI.color = new Color(0, 0, 0, 0.6f);
        GUI.DrawTexture(new Rect(10, 10, 520, 190), Texture2D.whiteTexture);
        GUI.color = Color.white;
        GUI.Label(new Rect(24, 12, 500, 56), (hudLabel ?? "").ToUpper(), big);
        GUI.Label(new Rect(24, 70, 500, 130), hudText ?? "", small);
    }

    static void WriteInt(Stream s, int n)
    {
        s.WriteByte((byte)(n >> 24)); s.WriteByte((byte)(n >> 16)); s.WriteByte((byte)(n >> 8)); s.WriteByte((byte)n);
    }

    // ------------------------------------------------------------------ background thread: TCP server (one client)
    void ServerLoop()
    {
        var listener = new TcpListener(IPAddress.Loopback, port);
        listener.Start();
        while (running)
        {
            try
            {
                if (!listener.Pending()) { Thread.Sleep(50); continue; }
                using (var client = listener.AcceptTcpClient())
                using (var stream = client.GetStream())
                using (var reader = new StreamReader(stream, Encoding.UTF8))
                {
                    clientConnected = true;
                    string line;
                    while (running && (line = reader.ReadLine()) != null)
                    {
                        var job = new Job { req = JsonUtility.FromJson<Req>(line) };
                        jobs.Enqueue(job);
                        job.done.Wait();
                        stream.Write(job.reply, 0, job.reply.Length);
                    }
                }
            }
            catch (Exception e) { Debug.LogWarning("bridge: " + e.Message); }
            clientConnected = false;
        }
        listener.Stop();
    }

    void OnDestroy() { running = false; }

    static Texture2D NoiseTexture(Color tint, float strength)
    {
        var tex = new Texture2D(256, 256, TextureFormat.RGB24, true);
        var rnd = new System.Random(7);
        for (int y = 0; y < 256; y++)
            for (int x = 0; x < 256; x++)
            {
                float n = 1f - strength + strength * (float)rnd.NextDouble();
                tex.SetPixel(x, y, new Color(tint.r * n, tint.g * n, tint.b * n));
            }
        tex.Apply(); tex.wrapMode = TextureWrapMode.Repeat;
        return tex;
    }
}
