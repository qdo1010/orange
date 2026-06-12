#pragma once
// jarvis_runner.h — lime-facing wrapper around the JARVIS 3D pose pipeline.
//
// Deliberately lime-agnostic (no Emergent / lime headers): lime feeds it the
// per-camera RGBA device pointers it already has and reads back 3D keypoints.
// Threading: submit() is called from each camera's capture thread when a frame
// is ready; when all cameras have submitted the same frame_id, a background
// worker runs the distributed pipeline (so capture threads never block). If the
// worker is still busy, newer frames are dropped (pose throttles below capture
// fps — expected and desired on the rig).
#if defined(__linux__) || defined(_WIN32)

#include "jarvis_pose.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace jarvis {

struct PoseResult {
    std::vector<float> points3D;     // J*3 world mm
    std::vector<float> confidences;  // J
    Eigen::Vector3d center3D = Eigen::Vector3d::Zero();
    uint64_t frame_id = 0;
    int cams_used = 0;
    double latency_ms = 0.0;
    bool valid = false;
};

class JarvisPoseRunner {
public:
    ~JarvisPoseRunner() { stop(); }

    // serials/cam_gpus are in camera order; calib files are
    // <calib_dir>/Cam<serial>.yaml (read fresh — rig re-calibrates daily).
    bool init(const std::string &model_dir, const std::string &calib_dir,
              const std::vector<std::string> &serials,
              const std::vector<int> &cam_gpus, int central_gpu);

    // Called per camera when its RGBA32 device frame (on cam_gpus[cam_idx]) is
    // ready. frame_id identifies a synced capture across cameras.
    void submit(int cam_idx, const uint8_t *rgba_dev, int w, int h, uint64_t frame_id);

    // Convenience for callers that know the camera serial but not the jarvis
    // index (e.g. lime's per-camera capture thread). No-op if serial unknown.
    void submit_by_serial(const std::string &serial, const uint8_t *rgba_dev,
                          int w, int h, uint64_t frame_id) {
        auto it = serial_to_idx_.find(serial);
        if (it != serial_to_idx_.end()) submit(it->second, rgba_dev, w, h, frame_id);
    }

    // Thread-safe snapshot of the most recent completed result.
    PoseResult latest() const;

    // Reproject the latest 3D keypoints to camera `serial`'s image (full
    // pinhole + distortion). uv_out = [u0,v0,u1,v1,...] in image pixels (origin
    // top-left); conf_out = per-keypoint confidence. Returns false if the serial
    // isn't a JARVIS camera or there's no valid result yet. Lets lime overlay
    // the pose on whichever camera(s) it streams.
    bool reproject_latest(const std::string &serial,
                          std::vector<float> &uv_out,
                          std::vector<float> &conf_out) const;

    bool ready() const { return loaded_; }
    const Config &config() const { return cfg_; }
    void stop();

private:
    void worker_loop();

    bool loaded_ = false;
    int n_ = 0;
    Config cfg_;
    PoseCoordinator coord_;
    std::vector<CameraParams> cams_;   // calib per camera (for reprojection)

    // Per-camera frame snapshots. lime's capture buffers get reused at 180 fps,
    // so we copy each camera's frame into JARVIS-owned memory at submit time
    // (capture thread, frame still valid) — the worker then processes a stable,
    // synced set regardless of streaming/display load. Without this the worker
    // reads stale/overwritten buffers and the views desync.
    bool snapshot_frame(int cam_idx, const uint8_t *frame, int w, int h);
    std::vector<uint8_t *> snap_;        // device buffer per cam (on its gpu)
    std::vector<size_t> snap_cap_;       // allocated bytes per cam
    std::vector<cudaStream_t> snap_stream_;

    // Staging for the in-progress frame set.
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<const uint8_t *> rgba_;
    std::vector<int> w_, h_;
    std::vector<uint64_t> have_;     // frame_id submitted per cam (0 = none)
    std::unordered_map<std::string, int> serial_to_idx_;
    uint64_t cur_frame_ = 0;
    bool job_ready_ = false;         // a complete frame set is staged
    bool busy_ = false;              // worker mid-run (drop newer frames)

    mutable std::mutex result_mtx_;
    PoseResult result_;

    std::thread worker_;
    std::atomic<bool> running_{false};
};

// Process-wide runner so lime's capture/display threads share one instance.
// Init once at startup, submit per frame, read latest() in the display thread.
inline JarvisPoseRunner &shared_runner() {
    static JarvisPoseRunner r;
    return r;
}

} // namespace jarvis
#endif // __linux__ || _WIN32
