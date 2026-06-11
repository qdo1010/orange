# Integrating real-time JARVIS 3D pose into lime (branch: `lime-obb-jarvis`)

Status: **design + groundwork**. The 3D model side is done and validated (see
"Model side" below). This doc is the plan for wiring it into lime's capture app.

## TL;DR

lime already has every primitive JARVIS needs. The integration is a **bridge**,
not a reimplementation:

- lime produces 4 RGBA debayer frames **on the GPU** (NVDEC/debayer output).
- lime has per-camera calibration `CameraCalibResults{k, dist_coeffs, r, rvec,
  tvec, projection_mat}` (`src/realtime_tool.h:15`) — a near 1:1 match for red's
  `CameraParams`.
- lime already has a **multi-camera sync barrier + 3D thread**
  (`detection3d_proc`, `src/detect3d.cpp:16`) that waits for all cams, does
  triangulation, and reprojects 3D→2D for overlay. That is exactly the shape of
  the JARVIS 3D stage.

red already ships a **self-contained, header-only** HybridNet pipeline whose
device entry point does the whole thing (center-detect → triangulate center →
effTrack 2D → pad → hybrid3d) from N GPU frames + N `CameraParams`:

```cpp
// red/src/jarvis_hybridnet.h:1120
bool jarvis_hybridnet_predict_frame_device(
    JarvisHybridNetState &state,
    const std::vector<const uint8_t *> &camera_rgba_device,  // N RGBA32 dev ptrs
    const std::vector<int> &widths, const std::vector<int> &heights,
    const std::vector<CameraParams> &camera_params,          // red's Eigen struct
    AnnotationMap &annotations, SkeletonContext &skeleton, u32 frame_idx);
```

It fills `state.points3D_out` (J×3, world mm) and `state.confidences_out` (J)
*before* the annotation/overlay step. So lime can read those directly.

## Model side (DONE — 2026-06-11)

The 3D `hybrid3d` stage now exports and runs in real time. See the memory note
`jarvis-onnx-export-setup.md` for full detail. Summary:

- The blocker was a 4-vs-6 arg mismatch: upstream `ReprojectionLayer` wants
  extrinsic `[R|t]^T` + separate K^T + radial distortion; red's C++ feeds a single
  combined `cameraMatrices = P^T = (K·[R|t])^T` and 4 inputs only.
- Fixed **script-side only** (no red C++ change) in
  `red/scripts/export_jarvis_onnx.py`: new `RedReprojectionLayer` projects the
  voxel grid straight through `P` (pinhole, no lens distortion) with a 4-arg
  forward; clamp / trilinear-upsample(50³→100³) / gather / camera-average are
  upstream-verbatim.
- `hybrid3d.onnx` IO (mouse_merge_6kp: 4 cams, 6 joints):
  inputs `heatmaps_padded[B,4,6,354,354]`, `centerHM[B,4,2]`, `center3D[B,3]`,
  `cameraMatrices[B,4,4,3]`; outputs `points3D[B,6,3]`, `confidences[B,6]`
  (+ an unused `heatmap_final`).
- Validated: exports in the `jarvis` conda env; compiles with **TRT 8.6.1.6**
  (red's exact version) via `red/scripts/compile_tensorrt_engines.sh`; runs
  **~2.3 ms GPU / 434 qps** on an A6000.
- **Distortion caveat**: dropping lens distortion adds a reprojection error that
  grows away from each image's principal point (small for a centered animal). If
  real-data validation shows it matters, add `intrinsicMatrices` +
  `distortionCoefficients` engine inputs in `RedReprojectionLayer` and extend
  red's `jarvis_hn_assemble_hybrid3d_aux_inputs`.

Artifacts live in `realtime_jarvis_model/mouse_merge_6kp/onnx/`:
`center_detect`, `hybridnet_efftrack`, `hybrid3d` (.onnx + .engine), `manifest.json`.

## Architecture for lime

```
 capture thread (per cam, existing)
   └─ debayer → RGBA32 on GPU  ── notify ──┐
                                           ▼
                            JARVIS coordinator thread  (new; modeled on detection3d_proc)
                              waits until all N cams have a fresh RGBA frame
                              gathers N device ptrs + N CameraParams
                              calls jarvis_hybridnet_predict_frame_device(...)
                                (center → triangulate → effTrack → hybrid3d, all on GPU)
                              reads state.points3D_out / confidences_out
                              reproject 3D→2D per cam for overlay (cv::projectPoints,
                                like detect3d.cpp:89) → write into per-cam display
```

Why one coordinator thread (not per-camera like YOLO): the hybrid3d stage is
inherently multi-view — it needs all N 2D heatmaps together. red's predict entry
already runs center-detect + effTrack for **all** cameras in one batched call, so
the per-camera `FrameDetector` YOLO model is *replaced*, not duplicated, in this
mode. Per-camera threads just stage their latest RGBA device pointer.

## Concrete work items

### 1. Vendor red's JARVIS module — isolate the `CameraParams` clash
lime has its own `struct CameraParams` (`src/camera.h:17`, has `gpu_id`, no
`projection_mat`/Eigen) that **collides by name** with red's. So the vendored red
code must live in its own translation unit / namespace, never co-included with
lime's `camera.h`.

Dependency closure of `red/src/jarvis_hybridnet.h`:
`annotation.h, camera.h, red_math.h, skeleton.h, types.h, json.hpp,
jarvis_hybridnet_cuda.{h,cu}` (+ CUDA/TensorRT/NvInferPlugin). The TRT wrapper
`namespace jarvis_hn_trt` is *inside* `jarvis_hybridnet.h` (self-contained).
lime already bundles `src/json.hpp` (nlohmann) — satisfied.

Recommended: copy red's `jarvis_hybridnet.h`, `jarvis_hybridnet_cuda.{h,cu}`,
`camera.h`, `red_math.h`, `types.h`, and the minimal `annotation.h`/`skeleton.h`
into `lime/src/jarvis/` and wrap them in `namespace jarvis { ... }`. Compile the
adapter (item 2) and coordinator (item 4) as the *only* TUs that include
`jarvis/jarvis_hybridnet.h`; keep them separate from any TU that includes lime's
`camera.h`. (Alternative: lightly refactor red's predict to drop the
`AnnotationMap`/`SkeletonContext` params and the Stage-7 overlay write, since
`state.points3D_out` is already populated before it — removes the annotation.h /
skeleton.h dependency entirely. Cleaner but touches red code.)

### 2. Calibration adapter: lime `CameraCalibResults` → red `CameraParams`
Both hold `k, dist_coeffs, r, rvec, tvec, projection_mat`. lime's are `cv::Mat`;
red's are Eigen. A ~30-line `cv::Mat`→Eigen copy in a new
`src/jarvis/calib_bridge.cpp`. red builds `cameraMatrices = P^T` from
`projection_mat` internally (`jarvis_hybridnet.h:771`), so just fill
`projection_mat` (and `k`/`dist_coeffs`/`r`/`tvec` for the 3D→2D overlay
reprojection). lime's `projection_mat` is already `K·[R|t]` (`detect3d.cpp`
reprojection uses the same matrices), matching red's convention exactly.

### 3. New detect mode + config
- Add `Pose3D_Jarvis` to `enum DetectMode` (`src/video_capture.h:40`).
- Add a `jarvis_model` dir path to the camera JSON config next to `yolo`
  (`src/project.cpp:151`), pointing at the dir holding `*.engine` + `manifest.json`.
- Load once via red's `jarvis_hybridnet_load(state, model_dir, gpu_id)`
  (`jarvis_hybridnet.h:496`); it parses `manifest.json` (joints, cams, bbox, ROI,
  grid, mean/std) and deserializes the 3 engines.

### 4. JARVIS coordinator thread
Model on `detection3d_proc` (`src/detect3d.cpp:16`). Reuse the existing
`frame_detect_state` / `cv3d` barrier so it fires once all `Pose3D_Jarvis`
cameras have a fresh frame. Gather the N RGBA **device** pointers (the same
`device_image_ptr` handed to `FrameDetector::notify_frame_ready`,
`FrameDetector.h:22`) + N adapted `CameraParams`, call
`jarvis_hybridnet_predict_frame_device`, then reproject the J 3D keypoints to
each streaming camera with `cv::projectPoints` (`detect3d.cpp:89`) into a new
per-cam pose-overlay buffer.

### 5. Display + output
- Store `J` 3D keypoints + confidences in a new `Pose3d` struct (parallel to
  `Ball3d`, `realtime_tool.h:63`) inside `Detection3d`.
- Draw 2D skeleton overlay per cam (reuse the GPU OBB-draw pattern,
  `src/kernel.cuh`, or draw keypoint markers + skeleton edges from
  `manifest.json`'s `skeleton`).
- Optionally serialize 3D pose over the existing ENet/FlatBuffer channel like OBB
  (`opengldisplay.cpp:339`).

### 6. Build system
- Add `src/jarvis/jarvis_hybridnet_cuda.cu` to `CUDA_SOURCES` in `CMakeLists.txt:69`.
- Add adapter/coordinator `.cpp`s to the `orange` target sources. (Headless
  `orange_client` excludes `detect3d.cpp`/`FrameDetector.cpp` at
  `CMakeLists.txt:38` — decide whether pose runs headless too; if yes, don't
  exclude the coordinator.)
- lime builds CUDA for `sm_80` (`CMakeLists.txt:8`); the rig GPUs are A6000/A16
  (`sm_86`) — forward-compatible. **Engines are per-rig**: run
  `red/scripts/compile_tensorrt_engines.sh <onnx_dir>` on each rig (TRT 8.6.1.6).

## Implementation status (Option B built + validated, 2026-06-11)

The decomposed distributed pipeline is implemented and runs end-to-end on the
rig's real engines + real calibration. Module: `src/jarvis/`.

- `jarvis_trt.h` — TRT engine wrapper (vendored from red, `jarvis_hn_trt`).
- `jarvis_cuda.{h,cu}` — resize/crop/pad CUDA kernels (vendored from red).
- `red_math.h` — Eigen projection/undistort/DLT-triangulation (vendored).
- `jarvis_pose.h` — the Option-B pipeline:
  - `Cam2D` — center_detect + efftrack engines pinned to one camera GPU, with
    `launch_*/finish_*` split so all cameras run concurrently across GPUs.
  - `Hybrid3D` — the **3D network** (V2VNet reproject + soft-argmax) on the
    central GPU; gathers per-cam padded heatmaps via `cudaMemcpyPeerAsync`.
  - `PoseCoordinator` — phase 1 center-detect (all GPUs) → DLT triangulate →
    phase 2 efftrack (all GPUs) + gather → phase 3 hybrid3d. `predict_with_center`
    skips phase 1 (fallback when center-detect misses; reuse last center3D).
- `jarvis_calib.cpp` — manifest.json parse + calib loader for lime's OpenCV
  `camera_matrix/distortion_coefficients/tc_ext/rc_ext` YAMLs. **Reads the calib
  folder fresh each load** (rig is re-calibrated daily — point config at the live
  folder, e.g. `/home/ratan/src/realtime_jarvis_model/calibration`, never copy it).
- `jarvis_pose_offline.cpp` + `Makefile` — standalone smoke test (no Emergent
  SDK). `make run` loads engines on GPUs 0–3 + hybrid3d on GPU 4, synthesizes
  frames, runs the full distributed pass.

Note the **2D engines must be compiled at BATCH=1** for the per-camera path:
`HN_BATCH=1 red/scripts/compile_tensorrt_engines.sh <onnx_dir>` (hybrid3d's cam
axis is baked from the ONNX = 4, no batch flag).

**Measured (this rig, 4× A16 cams + A6000 central, fp32 engines):**
- Distributed pass **~50 ms/frame** (~20 Hz). Was 119 ms before overlapping the
  GPUs; launching all cameras then syncing cut it to 50 ms (wall-clock ≈ one
  efftrack instead of four). The 3D output is the hybrid3d V2VNet soft-argmax.
- **The bottleneck is efftrack (704² EfficientNet-medium) on the A16s.** Next
  speed levers, in order: (1) **FP16 engines** (`FP16=1` compile) ~halves
  efftrack → ~25–30 ms; (2) pipeline across frames; (3) throttle pose < capture
  fps. The A6000 central 3D stage is ~2.3 ms — negligible.

To build/run: `cd src/jarvis && make run`.

## Resource allocation (the #1 constraint: speed + don't starve other GPUs)

### Hardware (this rig)
| GPU | Model | VRAM | Interconnect |
|-----|-------|------|--------------|
| 0–3 | A16   | 15 GB each | mutually **PIX** (same PCIe switch, fast P2P), NUMA 0 |
| 4   | RTX A6000 | 49 GB | **SYS** to the A16s (P2P routes through host bridge) |

Current camera→GPU map (`example_config/*.json`): cams on gpu_id **1,2,3,4** —
each camera decodes/debayers on its own GPU; YOLO currently runs per-camera on
that same GPU. So GPUs 1–4 are busy with capture (+YOLO); **GPU 0 (A16) and most
of GPU 4 (A6000) are the spare capacity.**

### Cost of each JARVIS stage (mouse_merge_6kp, 4 cams)
- `center_detect` 320² ×4: light (<1 ms total).
- `hybridnet_efftrack` 704² ×4: **the heavy stage** (EfficientNet backbone at
  high res). This is the GPU cost to budget for.
- `hybrid3d` (reproject + V2VNet 100³→50³ + soft-argmax): **light, ~2.3 ms** on
  A6000 (measured), once per frame-set.

### Two viable placements

**Option A — Centralize the whole pipeline on the A6000 (GPU 4). Recommended.**
Gather the 4 camera RGBA frames to GPU 4 and run center+effTrack (batched ×4) +
hybrid3d there. The 4 A16s keep doing camera decode + YOLO **untouched**.
- Pros: protects the busy camera GPUs; 49 GB easily holds 3 engines + 704²
  activations + the 100³ volume; batching ×4 on a strong GPU is throughput-optimal;
  hybrid3d is cheap so the central GPU isn't stressed.
- Cons: must move 4 frames GPU→GPU 4 each inference. GPU 4 is **SYS** (through
  host) to the A16s, so peer copies route via host bandwidth — bounded, but the
  real cost lever. Mitigate by (a) throttling pose to < capture fps (e.g. 30 Hz
  pose vs 100 Hz record — pose rarely needs full frame rate), and (b) cropping on
  each source GPU first so only the 704² crop (not the full frame) is transferred.

**Option B — Distribute the 2D stages per camera, centralize only 3D.**
Run center+effTrack on **each camera's own gpu_id** (frame already resident — zero
frame transfer), then gather only the padded heatmaps (~3 MB/cam, 12 MB total) to
one "3D GPU" (A6000) for hybrid3d.
- Pros: no full-frame cross-GPU copy; per-cam effTrack runs in parallel so 2D
  latency = one effTrack, not four; tiny gather.
- Cons: adds the heavy effTrack to GPUs already doing decode+YOLO — the exact
  load the user wants to avoid. Loses ×4 batching. Needs red's monolithic predict
  decomposed into a per-cam 2D function + a gather/3D function (more code).

Recommendation: **start with Option A** (least disruptive to the camera GPUs,
simplest reuse of red's batched pipeline, A6000 is the obvious spare). Revisit B
only if the frame-gather bandwidth proves to be the bottleneck.

### Mandatory red-side changes for either option (resource-safety)
red's predict paths are written for a single dedicated GPU and will fight lime's
multi-GPU app as-is:
- `jarvis_hybridnet.h:869` and `:1161` hardcode `cudaSetDevice(0)` — **GPU 0 is an
  A16 doing camera work.** Parameterize to the JARVIS GPU (pass through from
  `jarvis_hybridnet_load`'s `gpu_device_id`, which is already honored at load,
  `:538`).
- `jarvis_hybridnet.h:1169` calls `cudaDeviceSynchronize()` — a **device-wide**
  barrier that stalls every other context on that GPU each frame. Replace with
  `cudaStreamSynchronize`/`cudaEvent` scoped to the JARVIS streams only.
- Keep all 3 engines on their own non-default, **lower-priority** CUDA stream(s)
  so camera decode/YOLO keep priority. Consider CUDA MPS if decode + inference
  contend on the same GPU.
- Throttle: drive the coordinator at a configurable pose rate (decouple from
  capture fps) so JARVIS can't monopolize the GPU or the PCIe bus under load.

## Gotchas / open questions

- **`CameraParams` name clash** (item 1) — the main structural constraint.
- **N cameras**: this model is **4 cams** (`manifest.json`). red's
  `compile_tensorrt_engines.sh` comments say "16 cams" (stale, from the 24kp rig)
  and its 2D-engine `HN_BATCH` defaults to 16 → set `HN_BATCH=4` for this model.
  hybrid3d bakes the cam axis from the ONNX (4), so it needs no shape flags.
- **6 vs 24 joints**: read `num_joints` from `manifest.json` at load — don't
  hardcode. This model emits 6 (`Snout, EarL, EarR, Neck, SpineL, TailBase`); the
  config's 24-name `KEYPOINT_NAMES` list is leftover — trust `num_joints=6` and
  the first 6 names.
- **Distortion** dropped in hybrid3d reprojection (see Model side caveat).
- **Replaces, not adds**: a `Pose3D_Jarvis` camera does not also run YOLO/OBB
  (the JARVIS pipeline owns its own 2D stages). YOLO OBB stays for cameras in the
  existing modes.

## Suggested implementation order

1. Vendor red `jarvis/` into lime + get it compiling in isolation (no runtime).
2. Calibration adapter + a standalone offline test (feed 4 saved frames + a
   calib YAML, print 3D keypoints) — validates the bridge before touching the
   live capture loop.
3. Detect mode + config + `jarvis_hybridnet_load` at startup.
4. Coordinator thread on the `cv3d` barrier; read `points3D_out`.
5. Overlay + 3D output struct + (optional) network serialize.
6. On-rig engine compile + live validation; then revisit the distortion caveat
   against real data.
