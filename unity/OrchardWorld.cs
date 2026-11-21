// OrchardWorld.cs  -  one script that builds an orchard + a Jackal-sized robot and opens a closed-loop bridge to Python.
//
// HOW TO USE (Unity 6, URP project):
//   1. Put this file in Assets/Scripts/.
//   2. Hierarchy: right-click > Create Empty, name it "World", Inspector > Add Component > OrchardWorld.
//   3. Press Play. You see the orchard from behind the robot.
//        - No Python connected: drive with the keyboard (click the Game view first): W/S = forward/back, A/D = turn.
//        - Python connected (loop_client.py): Python decides, Unity only moves the robot and answers with what it sees.
//
// THE LOOP (lock-step, so it is easy to follow and always repeatable):
//   Python sends one request  ->  Unity moves the robot for dt seconds of SIMULATED time with (v, w)
//   ->  the robot camera renders what it sees from the new pose  ->  Unity sends back: pose + edge depth + JPEG image
//   ->  Python decides the next (v, w)  ->  ...
//   Simulated time only advances when Python asks, so a slow model cannot "miss" anything unless Python decides that
//   the robot keeps driving while the model thinks (then Python sends a step with dt = the model's answer time).
//
// World: Unity is Y-up. The robot drives along +Z in the lane centred at x = 0. Hedge rows (sphere canopies) run along Z
// from z = 0 to z = rowLength at x = +-rowSpacing/2, +-3*rowSpacing/2, ... Optional "lookalike" corridor before the rows.
// Robot camera like the CitrusFarm ZED2i: 0.28 m above ground, 102 deg horizontal field of view, 640 x 360.
// Edge depth = the farm rule's measurement: median z-depth over the outer 15 % of the image width, rows 25-75 %,
// from a grid of rays through those pixels (Physics.Raycast); nothing beyond 20 m counts.

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

public class OrchardWorld : MonoBehaviour
{
    [Header("Orchard")]
    public int rowsEachSide = 4;            // hedge rows on each side of the lane
    public float rowSpacing = 6.4f;         // distance between row centres (CitrusFarm 13B ~ 6.4 m)
    public float canopyRadius = 1.5f;
    public float rowLength = 40f;
    public float treeGapFactor = 1.3f;      // tree spacing = canopyRadius * this (canopies overlap into a hedge)
    public bool lookalike = false;          // wide corridor of short hedges at z = -14 .. -8 (like the turning lane)
    public float lookalikeHalfWidth = 4.25f;// canopy faces this far from the lane centre
    public int seed = 1;
    public GameObject treePrefab;           // optional: drag a real tree model here; otherwise spheres are used

    [Header("Robot")]
    public float startZ = -10f;
    public float camHeight = 0.28f;
    public float hFovDeg = 102f;
    public int imgW = 640, imgH = 360;

    [Header("Bridge to Python")]
    public int port = 5555;

    // --- state
    Transform robot; Camera robotCam; RenderTexture rt; Texture2D readTex;
    float px, pz, psi;           // pose: x, z (m), heading psi (rad, 0 = +Z, positive = turned right)
    float simTime;
    float entranceZ;             // camera z of the row entrance (farm labelling rule)
    bool clientConnected;
    Material leafMat, groundMat, trunkMat;

    class Req { public string cmd; public float v, w, dt, x0, z0, yaw; public bool image; }
    class Job { public Req req; public byte[] reply; public ManualResetEventSlim done = new ManualResetEventSlim(false); }
    readonly ConcurrentQueue<Job> jobs = new ConcurrentQueue<Job>();
    bool busy;
    Thread server; volatile bool running;

    // ------------------------------------------------------------------ build the world
    void Start()
    {
        Application.runInBackground = true;      // keep running when the Unity window is not focused
        UnityEngine.Random.InitState(seed);
        var lit = Shader.Find("Universal Render Pipeline/Lit");
        leafMat = new Material(lit); leafMat.SetColor("_BaseColor", new Color(0.16f, 0.36f, 0.12f)); leafMat.SetTexture("_BaseMap", NoiseTexture(new Color(0.55f, 0.9f, 0.45f), 0.45f));
        leafMat.SetTextureScale("_BaseMap", new Vector2(3, 3)); leafMat.SetFloat("_Smoothness", 0.05f);
        groundMat = new Material(lit); groundMat.SetColor("_BaseColor", new Color(0.58f, 0.48f, 0.36f)); groundMat.SetTexture("_BaseMap", NoiseTexture(Color.white, 0.25f));
        groundMat.SetTextureScale("_BaseMap", new Vector2(60, 60)); groundMat.SetFloat("_Smoothness", 0.0f);
        trunkMat = new Material(lit); trunkMat.SetColor("_BaseColor", new Color(0.35f, 0.25f, 0.15f));

        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground"; ground.transform.localScale = new Vector3(30, 1, 30);   // 300 x 300 m
        ground.GetComponent<Renderer>().sharedMaterial = groundMat;

        var trees = new GameObject("Trees").transform;
        for (int k = -rowsEachSide; k < rowsEachSide; k++)
        {
            float x = (k + 0.5f) * rowSpacing;
            bool laneRow = (k == -1 || k == 0);           // the two rows around the robot's lane
            for (float z = canopyRadius; z <= rowLength - canopyRadius; z += canopyRadius * treeGapFactor)
                MakeTree(trees, new Vector3(x, 0, z), canopyRadius * UnityEngine.Random.Range(0.92f, 1.0f), laneRow ? "lane" : "row");
        }
        if (lookalike)
            for (int side = -1; side <= 1; side += 2)
                for (float z = -14f; z <= -8f; z += canopyRadius * treeGapFactor)
                {
                    float r = canopyRadius * UnityEngine.Random.Range(0.92f, 1.0f);
                    MakeTree(trees, new Vector3(side * (lookalikeHalfWidth + r), 0, z), r, "look");
                }

        BuildRobot();
        Physics.SyncTransforms();
        entranceZ = ComputeEntranceZ();
        Debug.Log($"OrchardWorld: rows every {rowSpacing} m, canopy {canopyRadius} m, row entrance (label) at camera z = {entranceZ:F2} m. Bridge on port {port}.");

        running = true;
        server = new Thread(ServerLoop) { IsBackground = true }; server.Start();
    }

    void MakeTree(Transform parent, Vector3 basePos, float r, string kind)
    {
        if (treePrefab != null)
        {
            var t = Instantiate(treePrefab, basePos, Quaternion.Euler(0, UnityEngine.Random.Range(0f, 360f), 0), parent);
            t.name = kind + "_tree";
            foreach (var c in t.GetComponentsInChildren<Collider>()) c.gameObject.name = kind + "_tree";
            if (t.GetComponentInChildren<Collider>() == null) { var sc = t.AddComponent<SphereCollider>(); sc.center = Vector3.up * r; sc.radius = r; }
            return;
        }
        var trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        trunk.name = kind + "_trunk"; trunk.transform.SetParent(parent);
        trunk.transform.position = basePos + Vector3.up * 0.45f; trunk.transform.localScale = new Vector3(0.16f, 0.45f, 0.16f);
        trunk.GetComponent<Renderer>().sharedMaterial = trunkMat;
        var crown = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        crown.name = kind + "_canopy"; crown.transform.SetParent(parent);
        crown.transform.position = basePos + Vector3.up * (0.85f * r); crown.transform.localScale = Vector3.one * 2 * r;
        crown.GetComponent<Renderer>().sharedMaterial = leafMat;
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
        foreach (var c in robot.GetComponentsInChildren<Collider>()) Destroy(c);           // rays and trees never hit the robot
        foreach (var t in robot.GetComponentsInChildren<Transform>()) t.gameObject.layer = 2; // Ignore Raycast

        var camGo = new GameObject("RobotCamera"); camGo.transform.SetParent(robot);
        camGo.transform.localPosition = new Vector3(0, camHeight, 0.2f); camGo.transform.localRotation = Quaternion.identity;
        robotCam = camGo.AddComponent<Camera>();
        rt = new RenderTexture(imgW, imgH, 24); robotCam.targetTexture = rt;
        robotCam.aspect = imgW / (float)imgH;
        float vfov = 2f * Mathf.Atan(Mathf.Tan(hFovDeg * Mathf.Deg2Rad / 2f) * imgH / imgW) * Mathf.Rad2Deg;
        robotCam.fieldOfView = vfov; robotCam.nearClipPlane = 0.05f; robotCam.farClipPlane = 300f;
        readTex = new Texture2D(imgW, imgH, TextureFormat.RGB24, false);
        SetPose(0, startZ, 0);
    }

    void SetPose(float x, float z, float heading)
    {
        px = x; pz = z; psi = heading;
        robot.position = new Vector3(px, 0, pz);
        robot.rotation = Quaternion.Euler(0, psi * Mathf.Rad2Deg, 0);
        Physics.SyncTransforms();
    }

    // drive with (v forward m/s, w = left-turn rate rad/s) for dt simulated seconds
    void Integrate(float v, float w, float dt)
    {
        float h = 0.01f;
        for (float t = 0; t < dt - 1e-6f; t += h)
        {
            float step = Mathf.Min(h, dt - t);
            psi -= w * step;
            px += v * Mathf.Sin(psi) * step; pz += v * Mathf.Cos(psi) * step;
        }
        simTime += dt;
        SetPose(px, pz, psi);
    }

    // ------------------------------------------------------------------ measurements
    void EdgeDepth(out float left, out float right)
    {
        left = Band(0f, 0.15f, 0.25f, 0.75f); right = Band(0.85f, 1f, 0.25f, 0.75f);
    }

    // v2 candidate: only the image rows ABOVE the horizon (rows 25-50 % from the top), so flat ground is never measured
    void EdgeDepthHigh(out float left, out float right)
    {
        left = Band(0f, 0.15f, 0.5f, 0.75f); right = Band(0.85f, 1f, 0.5f, 0.75f);
    }

    float Band(float u0, float u1, float v0, float v1)
    {
        var vals = new List<float>();
        Vector3 o = robotCam.transform.position, f = robotCam.transform.forward;
        for (int i = 0; i < 12; i++)
            for (int j = 0; j < 24; j++)
            {
                float u = u0 + (u1 - u0) * (i + 0.5f) / 12f, v = v0 + (v1 - v0) * (j + 0.5f) / 24f;   // viewport v: 0 = bottom
                Ray ray = robotCam.ViewportPointToRay(new Vector3(u, v, 0));
                if (Physics.Raycast(ray, out RaycastHit hit, 40f, ~(1 << 2)))
                {
                    float z = Vector3.Dot(hit.point - o, f);          // z-depth along the camera axis, like the ZED
                    if (z > 0.1f && z < 20f) vals.Add(z);
                }
            }
        if (vals.Count == 0) return float.NaN;
        vals.Sort();
        return vals.Count % 2 == 1 ? vals[vals.Count / 2] : 0.5f * (vals[vals.Count / 2 - 1] + vals[vals.Count / 2]);
    }

    // farm labelling rule: first camera position where a canopy of the two lane rows reaches BOTH image edges (horizon row)
    float ComputeEntranceZ()
    {
        float half = hFovDeg * Mathf.Deg2Rad / 2f;
        Vector3 dl = new Vector3(-Mathf.Sin(half), 0, Mathf.Cos(half)), dr = new Vector3(Mathf.Sin(half), 0, Mathf.Cos(half));
        for (float z = startZ - 5f; z < rowLength; z += 0.02f)
        {
            Vector3 o = new Vector3(0, camHeight, z + 0.2f);
            bool l = Physics.Raycast(o, dl, out RaycastHit hl, 40f, ~(1 << 2)) && hl.collider.name.StartsWith("lane");
            bool r = Physics.Raycast(o, dr, out RaycastHit hr, 40f, ~(1 << 2)) && hr.collider.name.StartsWith("lane");
            if (l && r) return z + 0.2f;
        }
        return float.NaN;
    }

    // ------------------------------------------------------------------ main thread: handle one request per frame
    void Update()
    {
        if (!busy && jobs.TryDequeue(out Job job)) { busy = true; StartCoroutine(Handle(job)); }
        if (!clientConnected) KeyboardDrive();
    }

    void LateUpdate()
    {
        var cam = Camera.main;                                      // chase view for you
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
        if (q.cmd == "reset") { SetPose(q.x0, q.z0, q.yaw); simTime = 0; }
        else if (q.cmd == "step" && q.dt > 0) Integrate(q.v, q.w, q.dt);
        EdgeDepth(out float lm, out float rm);
        EdgeDepthHigh(out float lh, out float rh);
        byte[] jpg = null;
        if (q.image)
        {
            yield return new WaitForEndOfFrame();                   // the robot camera has rendered the new pose
            var prev = RenderTexture.active; RenderTexture.active = rt;
            readTex.ReadPixels(new Rect(0, 0, imgW, imgH), 0, 0); readTex.Apply();
            RenderTexture.active = prev;
            jpg = readTex.EncodeToJPG(92);
        }
        var ci = CultureInfo.InvariantCulture;
        Vector3 c = robotCam.transform.position;
        string head = "{" + string.Format(ci,
            "\"t\":{0:F3},\"x\":{1:F4},\"z\":{2:F4},\"yaw\":{3:F5},\"cam_z\":{4:F4},\"left_m\":{5},\"right_m\":{6},\"entrance_z\":{7:F3},\"row_spacing\":{8:F2},\"lookalike\":{9},\"left_hi\":{10},\"right_hi\":{11}",
            simTime, px, pz, psi, c.z, float.IsNaN(lm) ? "null" : lm.ToString("F3", ci), float.IsNaN(rm) ? "null" : rm.ToString("F3", ci),
            entranceZ, rowSpacing, lookalike ? "true" : "false",
            float.IsNaN(lh) ? "null" : lh.ToString("F3", ci), float.IsNaN(rh) ? "null" : rh.ToString("F3", ci)) + "}";
        byte[] hb = Encoding.UTF8.GetBytes(head);
        var ms = new MemoryStream();
        WriteInt(ms, hb.Length); ms.Write(hb, 0, hb.Length);
        WriteInt(ms, jpg == null ? 0 : jpg.Length); if (jpg != null) ms.Write(jpg, 0, jpg.Length);
        job.reply = ms.ToArray();
        job.done.Set();
        busy = false;
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
