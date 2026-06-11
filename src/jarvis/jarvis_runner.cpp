// jarvis_runner.cpp — see jarvis_runner.h.
#include "jarvis_runner.h"
#include <chrono>
#include <cstdio>
#include <filesystem>

namespace jarvis {

bool JarvisPoseRunner::init(const std::string &model_dir, const std::string &calib_dir,
                            const std::vector<std::string> &serials,
                            const std::vector<int> &cam_gpus, int central_gpu) {
    if (loaded_) return true;   // idempotent: safe to call from multiple sites
    n_ = (int)serials.size();
    if (n_ < 2 || (int)cam_gpus.size() != n_) {
        std::fprintf(stderr, "[jarvis-runner] need >=2 cams and matching gpu list\n");
        return false;
    }
    std::vector<std::string> calib_files;
    for (auto &s : serials)
        calib_files.push_back((std::filesystem::path(calib_dir) / ("Cam" + s + ".yaml")).string());
    std::vector<CameraParams> cams;
    if (!load_calibration(calib_files, cams)) return false;
    for (int c = 0; c < n_; ++c) cams[c].gpu_id = cam_gpus[c];
    // lime feeds raw BayerRG8 frames → debayer to RGBA inside the pipeline.
    if (!coord_.load(model_dir, cams, central_gpu, /*input_bayer=*/true)) return false;
    cfg_ = coord_.config();
    cams_ = cams;

    rgba_.assign(n_, nullptr);
    w_.assign(n_, 0); h_.assign(n_, 0);
    have_.assign(n_, 0);
    serial_to_idx_.clear();
    for (int c = 0; c < n_; ++c) serial_to_idx_[serials[c]] = c;
    loaded_ = true;
    running_ = true;
    worker_ = std::thread(&JarvisPoseRunner::worker_loop, this);
    std::fprintf(stderr, "[jarvis-runner] ready: %d cams, %d joints, central GPU %d\n",
                 n_, cfg_.num_joints, central_gpu);
    return true;
}

void JarvisPoseRunner::submit(int cam_idx, const uint8_t *rgba_dev, int w, int h, uint64_t frame_id) {
    if (!loaded_ || cam_idx < 0 || cam_idx >= n_ || !rgba_dev) return;
    std::lock_guard<std::mutex> lk(mtx_);
    if (busy_) return;                       // worker busy → drop (throttle)
    if (frame_id != cur_frame_) {            // new frame set: reset staging
        cur_frame_ = frame_id;
        std::fill(have_.begin(), have_.end(), 0);
    }
    rgba_[cam_idx] = rgba_dev; w_[cam_idx] = w; h_[cam_idx] = h;
    have_[cam_idx] = frame_id;
    for (int c = 0; c < n_; ++c) if (have_[c] != cur_frame_) return;  // not all in yet
    job_ready_ = true;                       // complete set → wake worker
    cv_.notify_one();
}

void JarvisPoseRunner::worker_loop() {
    while (running_) {
        std::vector<const uint8_t *> rgba; std::vector<int> w, h; uint64_t fid;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [&] { return job_ready_ || !running_; });
            if (!running_) break;
            rgba = rgba_; w = w_; h = h_; fid = cur_frame_;
            job_ready_ = false; busy_ = true;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = coord_.predict(rgba, w, h);
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        if (ok) {
            std::lock_guard<std::mutex> rl(result_mtx_);
            result_.points3D = coord_.points3D();
            result_.confidences = coord_.confidences();
            result_.center3D = coord_.last_center3d();
            result_.frame_id = fid;
            result_.cams_used = coord_.cams_used();
            result_.latency_ms = ms;
            result_.valid = true;
        }
        { std::lock_guard<std::mutex> lk(mtx_); busy_ = false; }
    }
}

PoseResult JarvisPoseRunner::latest() const {
    std::lock_guard<std::mutex> rl(result_mtx_);
    return result_;
}

bool JarvisPoseRunner::reproject_latest(const std::string &serial,
                                        std::vector<float> &uv_out,
                                        std::vector<float> &conf_out) const {
    if (!loaded_) return false;
    auto it = serial_to_idx_.find(serial);
    if (it == serial_to_idx_.end()) return false;
    const CameraParams &cp = cams_[it->second];
    PoseResult r;
    { std::lock_guard<std::mutex> rl(result_mtx_); r = result_; }
    if (!r.valid) return false;
    const int J = cfg_.num_joints;
    uv_out.resize(J * 2);
    conf_out.resize(J);
    for (int j = 0; j < J; ++j) {
        Eigen::Vector3d P(r.points3D[j*3+0], r.points3D[j*3+1], r.points3D[j*3+2]);
        Eigen::Vector2d uv = cp.telecentric
            ? red_math::projectPointTelecentric(P, cp.projection_mat, cp.k, cp.dist_coeffs)
            : red_math::projectPointR(P, cp.r, cp.tvec, cp.k, cp.dist_coeffs);
        uv_out[j*2+0] = (float)uv[0];
        uv_out[j*2+1] = (float)uv[1];
        conf_out[j] = r.confidences[j];
    }
    return true;
}

void JarvisPoseRunner::stop() {
    if (!running_.exchange(false)) return;
    { std::lock_guard<std::mutex> lk(mtx_); cv_.notify_all(); }
    if (worker_.joinable()) worker_.join();
}

} // namespace jarvis
