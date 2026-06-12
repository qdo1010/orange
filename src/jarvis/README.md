# JARVIS 3D pose in Lime — what was added and how

This module adds real-time **JARVIS-HybridNet 3D pose** to Lime, alongside the
existing YOLO/OBB path. The 4-camera rig produces live 3D keypoints that overlay
on the streamed video.

- Branch: **`lime-obb-jarvis`**
- Model export, TensorRT engine compilation, and the per-rig gotchas:
  see `/home/ratan/src/realtime_jarvis_model/README.md`.
- Original design + resource analysis: `docs/jarvis_integration.md`.

GUI-only: JARVIS is built into **`orange`**, never `orange_client` (we never run
it headless).

---

## How it works (1 paragraph)

Each camera's 2D stages (center-detect + effTrack) run **on that camera's own
GPU** (the frame is already there). The small per-camera heatmaps are
peer-copied to the **A6000**, which runs the 3D network (`hybrid3d` = V2VNet
reprojection + soft-argmax) and emits the 3D keypoints. A background worker owns
the pipeline so it never blocks the 180 fps capture; the GUI overlays the latest
pose every render frame. See the project README for the full data flow.

---

## Files added (`src/jarvis/`)

| File | Purpose |
|------|---------|
| `jarvis_pose.h` | `Cam2D` (per-cam center+effTrack, NPP debayer, per-GPU engine load, launch/finish split), `Hybrid3D` (central 3D net + `cudaMemcpyPeer` gather), `PoseCoordinator` (orchestration + RANSAC triangulation) |
| `jarvis_runner.{h,cpp}` | `JarvisPoseRunner`: `submit_by_serial()` per camera, background worker, `latest()` / `reproject_latest()`. `shared_runner()` = process-wide instance |
| `jarvis_trt.h` | TensorRT engine wrapper (vendored from red) |
| `jarvis_cuda.{h,cu}` | resize / crop / pad CUDA kernels (vendored from red) |
| `red_math.h` | Eigen projection / undistort / DLT triangulation (vendored from red) |
| `jarvis_calib.cpp` | manifest.json parse + calibration loader (OpenCV `FileStorage`) |
| `jarvis_pose_offline.cpp` | standalone synthetic smoke test (no Emergent SDK) |
| `jarvis_pose_images.cpp` | real-data test: decode frames, run pipeline, overlay 3D kp |
| `Makefile` | builds the test harnesses against TRT |

## Existing Lime files changed

| File | Change |
|------|--------|
| `video_capture.h` | `Pose3D_Jarvis` DetectMode; `enable_jarvis` flag on `CameraEachSelect` |
| `video_capture.cpp` | submit each enabled camera's frame to the runner (capture thread, `#ifndef HEADLESS`), paired by hardware PTP `frame_id` |
| `orange.cpp` | `jarvis_try_init()` (idempotent, both camera-open paths); `draw_jarvis_pose` overlay on streamed JARVIS cameras |
| `gui.h` | "JARVIS 3D Pose" checkbox + live status line + "JARVIS 3D Pose Settings" (Model dir / Calib dir / Central GPU); `draw_jarvis_pose()` helper |
| `project.cpp` | read `"enable_jarvis"` from camera JSON |
| `global.{h,cpp}` | `jarvis_model_dir` / `jarvis_calib_dir` / `jarvis_central_gpu` globals (GUI-editable, preset) |
| `CMakeLists.txt` | build `src/jarvis` into `orange` + `orange_client`'s shared objects; Eigen include |
| `quick_build/orange.sh` | compile `src/jarvis` (cuda + calib + runner) into `orange`; Eigen include |
| `run.sh` | `CUDA_MODULE_LOADING=LAZY` (less GPU mem, faster TRT init) |

---

## Progress log — what we did and how (chronological)

1. **Export the 3D stage.** `hybrid3d.onnx` never exported (4-vs-6 arg mismatch
   in the reprojection). Fixed script-side in red with a combined-`P`
   `RedReprojectionLayer`. Validated it compiles + runs (~2.3 ms in TRT 8.6).
2. **Distributed multi-GPU pipeline (Option B).** Built `src/jarvis`: per-camera
   2D on each camera's GPU, gather heatmaps, central 3D on the A6000. Overlapped
   the cameras (launch-all-then-sync) → **119 → ~50 ms/frame**.
3. **Validated on real rat data.** `jarvis_pose_images` overlay tool. Found two
   things: (a) calibration round-trips to **7–12 px** (red-native format works,
   no JARVIS conversion needed); (b) a confident **false** center-detect on an
   occluded view wrecked triangulation → added **RANSAC `robust_triangulate`**.
   Mean keypoint confidence **0.23 → 0.63** (occluded) / **0.79–0.82** (open).
4. **Wired into Lime's live capture loop.** DetectMode + `enable_jarvis` +
   `submit` hook + `jarvis_try_init` + `shared_runner`. Built clean (`orange`).
5. **GUI.** "JARVIS 3D Pose" checkbox, live status line, editable Model/Calib/GPU
   folders, and the reproject **overlay** on streamed cameras.
6. **GUI-only split.** Moved the hook inside `#ifndef HEADLESS` and reverted the
   client changes so `orange_client` stays JARVIS-free and still builds.
7. **TensorRT version fix.** Lime links **TRT 10.6**, not red's 8.6 → engines
   failed to deserialize. Recompiled with 10.6 (`compile_trt10_engines.sh`).
8. **Device-model hang.** TRT 10.6 `hybrid3d` *hangs* if built on a different GPU
   model than it runs on → build `hybrid3d` on the A6000, 2D on the A16s.
9. **Bayer debayer.** Lime feeds raw BayerRG8; the kernels read RGBA → illegal
   access/segfault. Added NPP `nppiCFAToRGBA` demosaic in `Cam2D`.
10. **Empty-table behavior confirmed correct** (no animal → no pose; one camera
    false-fires on static texture, rejected by the ≥2-camera requirement).
11. **Cross-device warning — real fix, not suppression.** Build a 2D engine **on
    each camera GPU** (`<stem>.gpu<N>.engine`); `Cam2D` loads the matching one.
    No cross-device mismatch → warning gone. (`CUDA_MODULE_LOADING=LAZY` fixes the
    separate lazy-loading warning.)
12. **Frame sync.** Pair cameras by the hardware PTP `frame_id` (robust to drops),
    not the local counter; fixed a sentinel so the "all cameras in" gate never
    fires on a partial set (incl. the 16-bit `frame_id` wrap every ~6 min).

---

## Build & run

```bash
cd ~/src/lime
bash quick_build/orange.sh     # builds targets/orange (with JARVIS)
./run.sh                        # launches with CUDA_MODULE_LOADING=LAZY
```

Then: select the `local/climb_jarvis` config, confirm the folders in the GUI's
"JARVIS 3D Pose Settings", open cameras, subscribe, stream a camera.

Engines must be compiled per-rig first (TRT 10.6, per-GPU):
```bash
~/src/realtime_jarvis_model/compile_trt10_engines.sh
```

### Test harnesses (no Emergent SDK)
```bash
cd src/jarvis && make
# synthetic mechanics test:
./jarvis_pose_offline <model_dir> <calib_dir> <s1,s2,s3,s4> 0,1,2,3 4
# real frames + overlay PNGs:
./jarvis_pose_images <model_dir> <calib_dir> <out_dir> s1:img1.png ... 0,1,2,3 4
```

---

## Open items

- Frame double-buffer (runner reads the capture pointer directly; snapshot at
  submit if tearing appears at 180 fps).
- INT8 effTrack to approach 60 fps.
- Optional: burn the overlay into the recorded .mp4 (currently live-GUI only).
- Retrain on rat data for production accuracy.
