#pragma once
// jarvis_pose.h — real-time JARVIS HybridNet 3D pose, decomposed for lime's
// multi-GPU rig (Option B: per-camera 2D on each camera's own GPU, then gather
// the small heatmaps to one GPU for the 3D stage).
//
// Pipeline (4 cams, 6 joints for mouse_merge_6kp):
//   per-cam GPU:  RGBA frame --center_detect--> 2D center peak
//   central:      peaks --DLT triangulate--> center3D (world mm)
//   per-cam GPU:  reproject center3D -> crop --efftrack--> 2D heatmaps -> pad
//   gather:       cudaMemcpyPeer padded heatmaps -> central GPU
//   central GPU:  heatmaps + calib --hybrid3d--> 3D keypoints + confidences
//
// Engines are pinned to specific GPUs. The 2D engines (center_detect,
// hybridnet_efftrack) must be compiled at BATCH=1 (one camera per GPU):
//   HN_BATCH=1 red/scripts/compile_tensorrt_engines.sh <onnx_dir>
// hybrid3d's cam axis is baked from the ONNX (=num_cameras), no batch flag.
#if defined(__linux__) || defined(_WIN32)

#include "jarvis_trt.h"
#include "jarvis_cuda.h"
#include "red_math.h"

#include <npp.h>
#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

namespace jarvis {

// ── Calibration (Eigen). Filled from lime's calib_yaml via jarvis_calib.cpp. ──
// Distinct from lime's global ::CameraParams — keep this namespaced.
struct CameraParams {
    Eigen::Matrix3d k = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 5, 1> dist_coeffs = Eigen::Matrix<double, 5, 1>::Zero();
    Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
    Eigen::Vector3d rvec = Eigen::Vector3d::Zero();
    Eigen::Vector3d tvec = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 4> projection_mat = Eigen::Matrix<double, 3, 4>::Zero();
    bool telecentric = false;
    int image_width = 0, image_height = 0;
    int gpu_id = 0;  // GPU where this camera's frame lives / 2D runs
};

// ── Config (from the model dir's manifest.json). ──
struct Config {
    int center_image_size  = 320;
    int keypoint_bbox_size  = 704;
    int num_joints          = 6;
    int num_cameras         = 4;
    float roi_cube_size_mm  = 200.0f;
    float grid_spacing_mm   = 2.0f;
    std::array<float, 3> dataset_mean = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> dataset_std  = {0.229f, 0.224f, 0.225f};
    std::vector<std::string> keypoint_names;

    int heatmap_hw_padded() const { return keypoint_bbox_size / 2 + 2; } // 354
};

// Parse manifest.json (OpenCV-free, uses lime's bundled nlohmann json).
// Implemented in jarvis_calib.cpp. Returns false on missing/malformed file.
bool load_manifest(Config &cfg, const std::string &manifest_path);

// Load N per-camera calibration YAMLs from a folder (lime calib_yaml format:
// camera_matrix, distortion_coefficients, tc_ext, rc_ext). `files` are the
// per-camera paths in camera order. Implemented in jarvis_calib.cpp (OpenCV).
bool load_calibration(const std::vector<std::string> &files,
                      std::vector<CameraParams> &out);

constexpr float kCenterDetectThreshold = 50.0f; // matches JARVIS python

// ─────────────────────────────────────────────────────────────────────────
// Cam2D — center_detect + hybridnet_efftrack engines pinned to one GPU.
// One instance per camera. All methods set the device first.
// ─────────────────────────────────────────────────────────────────────────
class Cam2D {
public:
    // input_bayer: lime feeds raw BayerRG8 (1 byte/px) — debayer to RGBA first.
    // Test harnesses feed RGBA32 already (input_bayer=false).
    bool load(const std::string &model_dir, int gpu_id, const Config &cfg,
              bool input_bayer) {
        gpu_id_ = gpu_id;
        cfg_ = cfg;
        input_bayer_ = input_bayer;
        if (cudaSetDevice(gpu_id_) != cudaSuccess) return false;
        namespace fs = std::filesystem;
        std::string cen = (fs::path(model_dir) / "center_detect.engine").string();
        std::string eff = (fs::path(model_dir) / "hybridnet_efftrack.engine").string();
        if (!jarvis_hn_trt::load_engine(center_, cen, logger_)) return false;
        if (!jarvis_hn_trt::load_engine(efftrack_, eff, logger_)) return false;

        const int J = cfg_.num_joints;
        const int Heff = cfg_.keypoint_bbox_size / 2;        // 352
        const int Hpad = cfg_.heatmap_hw_padded();           // 354
        padded_elems_ = (size_t)J * Hpad * Hpad;
        if (!ok(cudaMalloc(&d_padded_, padded_elems_ * sizeof(float)), "malloc padded")) return false;

        // Per-cam scratch for the device preprocessing kernels (N=1).
        if (!ok(cudaMalloc(&d_rgba_, sizeof(uint8_t *)), "malloc d_rgba")) return false;
        if (!ok(cudaMalloc(&d_w_, sizeof(int)), "malloc d_w")) return false;
        if (!ok(cudaMalloc(&d_h_, sizeof(int)), "malloc d_h")) return false;
        if (!ok(cudaMalloc(&d_cx_, sizeof(int)), "malloc d_cx")) return false;
        if (!ok(cudaMalloc(&d_cy_, sizeof(int)), "malloc d_cy")) return false;

        center_out_high_.resize((size_t)(cfg_.center_image_size / 2) *
                                (cfg_.center_image_size / 2));
        for (int i = 0; i < 3; ++i) inv_std_[i] = 1.0f / cfg_.dataset_std[i];
        loaded_ = true;
        return true;
    }

    // Stage 1, split launch/finish so N cameras overlap across their GPUs.
    // launch_center enqueues everything on center_.stream and returns without
    // syncing; finish_center waits and peak-picks. Run launch on all cams,
    // THEN finish on all cams — that's what makes the 4 GPUs run concurrently.
    bool launch_center(const uint8_t *frame_dev, int w, int h) {
        if (!loaded_ || cudaSetDevice(gpu_id_) != cudaSuccess) return false;
        const int C = cfg_.center_image_size;     // 320
        const int Hcen = C / 2;                    // 160
        const uint8_t *rgba_dev = debayer_if_needed(frame_dev, w, h, center_.stream);
        if (!rgba_dev) return false;
        upload_frame_meta(rgba_dev, w, h, center_.stream);
        auto *in = input_binding(center_);
        auto *out = output_binding(center_, Hcen);
        if (!in || !out) { std::fprintf(stderr, "[jarvis] center bindings missing\n"); return false; }
        if (jarvis_hn_resize_normalize_device(
                (const uint8_t *const *)d_rgba_, d_w_, d_h_,
                (float *)in->d_ptr, 1, C, cfg_.dataset_mean.data(), inv_std_,
                center_.stream) != cudaSuccess) return false;
        if (!center_.context->enqueueV3(center_.stream)) return false;
        return ok(cudaMemcpyAsync(center_out_high_.data(), out->d_ptr,
                  center_out_high_.size() * sizeof(float),
                  cudaMemcpyDeviceToHost, center_.stream), "center D2H");
    }
    bool finish_center(int &peak_x, int &peak_y, float &peak_val) {
        if (cudaSetDevice(gpu_id_) != cudaSuccess) return false;
        if (!ok(cudaStreamSynchronize(center_.stream), "center sync")) return false;
        const int Hcen = cfg_.center_image_size / 2;
        peak_pick(center_out_high_.data(), Hcen, Hcen, peak_x, peak_y, peak_val);
        return true;
    }
    bool run_center(const uint8_t *rgba_dev, int w, int h,
                    int &px, int &py, float &v) {
        return launch_center(rgba_dev, w, h) && finish_center(px, py, v);
    }

    // Stage 3+4 split: crop + efftrack + pad enqueued on efftrack_.stream
    // (no sync). The padded (J,354,354) heatmap lands in d_padded_ on this GPU.
    // Caller must sync_efftrack() before reading/gathering d_padded_.
    bool launch_efftrack(const uint8_t *frame_dev, int w, int h, int cx, int cy) {
        if (!loaded_ || cudaSetDevice(gpu_id_) != cudaSuccess) return false;
        const int B = cfg_.keypoint_bbox_size;     // 704
        const int Heff = B / 2;                     // 352
        const int J = cfg_.num_joints;
        const uint8_t *rgba_dev = debayer_if_needed(frame_dev, w, h, efftrack_.stream);
        if (!rgba_dev) return false;
        upload_frame_meta(rgba_dev, w, h, efftrack_.stream);
        if (!ok(cudaMemcpyAsync(d_cx_, &cx, sizeof(int), cudaMemcpyHostToDevice, efftrack_.stream), "cx H2D")) return false;
        if (!ok(cudaMemcpyAsync(d_cy_, &cy, sizeof(int), cudaMemcpyHostToDevice, efftrack_.stream), "cy H2D")) return false;
        auto *in = input_binding(efftrack_);
        auto *out = output_binding(efftrack_, Heff);
        if (!in || !out) { std::fprintf(stderr, "[jarvis] efftrack bindings missing\n"); return false; }
        if (jarvis_hn_crop_normalize_device(
                (const uint8_t *const *)d_rgba_, d_w_, d_h_, d_cx_, d_cy_,
                (float *)in->d_ptr, 1, B, cfg_.dataset_mean.data(), inv_std_,
                efftrack_.stream) != cudaSuccess) return false;
        if (!efftrack_.context->enqueueV3(efftrack_.stream)) return false;
        return jarvis_hn_pad_heatmaps_device((const float *)out->d_ptr, d_padded_,
                1, J, Heff, Heff, efftrack_.stream) == cudaSuccess;
    }
    bool sync_efftrack() {
        if (cudaSetDevice(gpu_id_) != cudaSuccess) return false;
        return ok(cudaStreamSynchronize(efftrack_.stream), "efftrack sync");
    }
    bool run_efftrack(const uint8_t *rgba_dev, int w, int h, int cx, int cy) {
        return launch_efftrack(rgba_dev, w, h, cx, cy) && sync_efftrack();
    }

    const float *padded_heatmap() const { return d_padded_; }   // device ptr on gpu_id_
    size_t padded_bytes() const { return padded_elems_ * sizeof(float); }
    int gpu_id() const { return gpu_id_; }

private:
    static bool ok(cudaError_t e, const char *w) { return jarvis_hn_trt::cuda_ok(e, w); }

    // Async on the given stream; kernels reading these run on the same stream,
    // so stream ordering guarantees the uploads land first — no sync needed.
    void upload_frame_meta(const uint8_t *rgba_dev, int w, int h, cudaStream_t s) {
        cudaMemcpyAsync(d_rgba_, &rgba_dev, sizeof(uint8_t *), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_w_, &w, sizeof(int), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_h_, &h, sizeof(int), cudaMemcpyHostToDevice, s);
    }
    static jarvis_hn_trt::Binding *input_binding(jarvis_hn_trt::Engine &e) {
        for (auto &kv : e.bindings) if (kv.second.is_input) return &kv.second;
        return nullptr;
    }
    static jarvis_hn_trt::Binding *output_binding(jarvis_hn_trt::Engine &e, int hi_dim) {
        for (auto &kv : e.bindings)
            if (!kv.second.is_input && kv.second.dims.nbDims == 4 &&
                kv.second.dims.d[2] == hi_dim) return &kv.second;
        return nullptr;
    }
    static void peak_pick(const float *hm, int W, int H, int &bx, int &by, float &bv) {
        bx = 0; by = 0; bv = hm[0];
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float v = hm[y * W + x];
                if (v > bv) { bv = v; bx = x; by = y; }
            }
    }

    // lime feeds raw BayerRG8 (1 byte/px). Demosaic to RGBA (4 byte/px) so the
    // resize/crop kernels read it correctly. Returns the RGBA device buffer
    // (cam-local, on the given stream). When input is already RGBA, passthrough.
    const uint8_t *debayer_if_needed(const uint8_t *frame, int w, int h,
                                     cudaStream_t s) {
        if (!input_bayer_) return frame;
        if (!d_debayer_ || w != debayer_w_ || h != debayer_h_) {
            if (d_debayer_) cudaFree(d_debayer_);
            if (!ok(cudaMalloc(&d_debayer_, (size_t)w * h * 4), "malloc debayer")) return nullptr;
            debayer_w_ = w; debayer_h_ = h;
        }
        NppiSize sz{w, h};
        NppiRect roi{0, 0, w, h};
        nppSetStream(s);   // single worker thread → sequential, safe
        // BayerRG8 -> RGGB grid (matches lime image_processing.h). Alpha = 255.
        NppStatus st = nppiCFAToRGBA_8u_C1AC4R(
            frame, w * (int)sizeof(uint8_t), sz, roi,
            d_debayer_, w * (int)sizeof(uint8_t) * 4,
            NPPI_BAYER_RGGB, NPPI_INTER_UNDEFINED, 255);
        if (st != NPP_SUCCESS) {
            std::fprintf(stderr, "[jarvis] nppiCFAToRGBA failed: %d\n", (int)st);
            return nullptr;
        }
        return d_debayer_;
    }

    int gpu_id_ = 0;
    bool loaded_ = false;
    bool input_bayer_ = false;
    uint8_t *d_debayer_ = nullptr;
    int debayer_w_ = 0, debayer_h_ = 0;
    Config cfg_;
    jarvis_hn_trt::Logger logger_;
    jarvis_hn_trt::Engine center_, efftrack_;
    float *d_padded_ = nullptr;
    size_t padded_elems_ = 0;
    uint8_t **d_rgba_ = nullptr;
    int *d_w_ = nullptr, *d_h_ = nullptr, *d_cx_ = nullptr, *d_cy_ = nullptr;
    std::vector<float> center_out_high_;
    float inv_std_[3] = {1, 1, 1};
};

// ─────────────────────────────────────────────────────────────────────────
// Hybrid3D — the 3D stage on one central GPU. Gathers per-cam padded heatmaps
// (peer copy) into its heatmaps_padded binding, then runs the engine.
// ─────────────────────────────────────────────────────────────────────────
class Hybrid3D {
public:
    bool load(const std::string &model_dir, int gpu_id, const Config &cfg) {
        gpu_id_ = gpu_id;
        cfg_ = cfg;
        if (cudaSetDevice(gpu_id_) != cudaSuccess) return false;
        namespace fs = std::filesystem;
        std::string h3 = (fs::path(model_dir) / "hybrid3d.engine").string();
        if (!jarvis_hn_trt::load_engine(hybrid3d_, h3, logger_)) return false;
        const int J = cfg_.num_joints;
        const int Hpad = cfg_.heatmap_hw_padded();
        slot_bytes_ = (size_t)J * Hpad * Hpad * sizeof(float);
        camera_matrices_.assign((size_t)cfg_.num_cameras * 12, 0.f);
        centerHM_.assign((size_t)cfg_.num_cameras * 2, 0.f);
        points3D_.assign((size_t)J * 3, 0.f);
        confidences_.assign((size_t)J, 0.f);
        loaded_ = true;
        return true;
    }

    // Peer-copy camera c's padded heatmap (on src_gpu) into slot c of the
    // central heatmaps_padded binding. Async on the hybrid3d stream.
    bool gather_slot(int c, const float *src_dev, int src_gpu) {
        if (!loaded_ || cudaSetDevice(gpu_id_) != cudaSuccess) return false;
        auto *hp = hybrid3d_.get("heatmaps_padded");
        if (!hp) { std::fprintf(stderr, "[jarvis] heatmaps_padded binding missing\n"); return false; }
        char *dst = (char *)hp->d_ptr + (size_t)c * slot_bytes_;
        return jarvis_hn_trt::cuda_ok(
            cudaMemcpyPeerAsync(dst, gpu_id_, src_dev, src_gpu, slot_bytes_,
                                hybrid3d_.stream), "gather peer copy");
    }

    // Run the 3D stage. centerHM/center3D/cameraMatrices are filled by the
    // coordinator (assemble_aux). Fills points3D()/confidences().
    bool run(const Eigen::Vector3d &center3D) {
        if (!loaded_ || cudaSetDevice(gpu_id_) != cudaSuccess) return false;
        float c3[3] = {(float)center3D[0], (float)center3D[1], (float)center3D[2]};
        if (!h2d("centerHM", centerHM_.data(), centerHM_.size() * sizeof(float))) return false;
        if (!h2d("center3D", c3, sizeof(c3))) return false;
        if (!h2d("cameraMatrices", camera_matrices_.data(), camera_matrices_.size() * sizeof(float))) return false;
        if (!hybrid3d_.context->enqueueV3(hybrid3d_.stream)) return false;
        if (!d2h("points3D", points3D_.data(), points3D_.size() * sizeof(float))) return false;
        if (!d2h("confidences", confidences_.data(), confidences_.size() * sizeof(float))) return false;
        return jarvis_hn_trt::cuda_ok(cudaStreamSynchronize(hybrid3d_.stream), "h3d sync");
    }

    // cameraMatrices[c] = P^T (4x3 row-major) from each cam's projection_mat.
    void assemble_aux(const std::vector<CameraParams> &cams,
                      const std::vector<int> &cx, const std::vector<int> &cy) {
        const int N = cfg_.num_cameras;
        for (int c = 0; c < N; ++c) {
            const auto &P = cams[c].projection_mat;
            float *o = camera_matrices_.data() + c * 12;
            for (int r = 0; r < 4; ++r)
                for (int col = 0; col < 3; ++col)
                    o[r * 3 + col] = (float)P(col, r);
            centerHM_[c * 2 + 0] = (float)cx[c];
            centerHM_[c * 2 + 1] = (float)cy[c];
        }
    }

    const std::vector<float> &points3D() const { return points3D_; }
    const std::vector<float> &confidences() const { return confidences_; }
    int gpu_id() const { return gpu_id_; }

private:
    bool h2d(const char *name, const void *host, size_t bytes) {
        auto *b = hybrid3d_.get(name);
        if (!b || !b->is_input || b->bytes != bytes) {
            std::fprintf(stderr, "[jarvis] h3d input %s bad (have=%p bytes=%zu want=%zu)\n",
                         name, (void *)b, b ? b->bytes : 0, bytes);
            return false;
        }
        return jarvis_hn_trt::cuda_ok(
            cudaMemcpyAsync(b->d_ptr, host, bytes, cudaMemcpyHostToDevice, hybrid3d_.stream), "h3d H2D");
    }
    bool d2h(const char *name, void *host, size_t bytes) {
        auto *b = hybrid3d_.get(name);
        if (!b || b->is_input || b->bytes != bytes) {
            std::fprintf(stderr, "[jarvis] h3d output %s bad\n", name);
            return false;
        }
        return jarvis_hn_trt::cuda_ok(
            cudaMemcpyAsync(host, b->d_ptr, bytes, cudaMemcpyDeviceToHost, hybrid3d_.stream), "h3d D2H");
    }

    int gpu_id_ = 0;
    bool loaded_ = false;
    Config cfg_;
    jarvis_hn_trt::Logger logger_;
    jarvis_hn_trt::Engine hybrid3d_;
    size_t slot_bytes_ = 0;
    std::vector<float> camera_matrices_, centerHM_, points3D_, confidences_;
};

// ─────────────────────────────────────────────────────────────────────────
// Coordinator — owns N Cam2D (one per camera GPU) + one central Hybrid3D and
// runs the full distributed pass. Returns false if < 2 cams clear the center
// threshold (need 2 rays to triangulate).
// ─────────────────────────────────────────────────────────────────────────
class PoseCoordinator {
public:
    bool load(const std::string &model_dir, const std::vector<CameraParams> &cams,
              int central_gpu, bool input_bayer = false) {
        if (!load_manifest(cfg_, (std::filesystem::path(model_dir) / "manifest.json").string()))
            return false;
        // hybrid3d's camera axis is baked into the ONNX at the model's camera
        // count — enabling a different number of cameras can't work.
        if ((int)cams.size() != cfg_.num_cameras) {
            std::fprintf(stderr,
                "[jarvis] model expects exactly %d cameras but %d were enabled — "
                "enable JARVIS on exactly the %d calibrated rig cameras.\n",
                cfg_.num_cameras, (int)cams.size(), cfg_.num_cameras);
            return false;
        }
        cams_ = cams;
        cam2d_.resize(cams.size());
        for (size_t c = 0; c < cams.size(); ++c)
            if (!cam2d_[c].load(model_dir, cams[c].gpu_id, cfg_, input_bayer)) return false;
        return hybrid3d_.load(model_dir, central_gpu, cfg_);
    }

    // rgba_dev[c] is camera c's RGBA32 frame on cams_[c].gpu_id.
    bool predict(const std::vector<const uint8_t *> &rgba_dev,
                 const std::vector<int> &w, const std::vector<int> &h) {
        Eigen::Vector3d center3D;
        if (!compute_center3d(rgba_dev, w, h, center3D)) return false;
        return predict_with_center(rgba_dev, w, h, center3D);
    }

    // Phase 1 only: per-cam center detect + DLT triangulation → center3D.
    // Returns false if < 2 cams clear the threshold.
    // Diagnostics from the last compute_center3d (for tracking-quality debug).
    struct CenterDiag { int cam; double img_x, img_y; float val; bool passed; };
    const std::vector<CenterDiag> &center_diag() const { return center_diag_; }
    const Eigen::Vector3d &last_center3d() const { return last_center3d_; }

    bool compute_center3d(const std::vector<const uint8_t *> &rgba_dev,
                          const std::vector<int> &w, const std::vector<int> &h,
                          Eigen::Vector3d &center3D) {
        const int N = cfg_.num_cameras;
        // Launch all cams' center-detect first so the GPUs run concurrently,
        // then collect — wall-clock ≈ one center-detect, not N.
        for (int c = 0; c < N; ++c)
            if (!cam2d_[c].launch_center(rgba_dev[c], w[c], h[c])) return false;
        std::vector<Eigen::Vector2d> und;
        std::vector<Eigen::Matrix<double, 3, 4>> proj;
        std::vector<float> vals;
        center_diag_.clear();
        for (int c = 0; c < N; ++c) {
            int px, py; float v;
            if (!cam2d_[c].finish_center(px, py, v)) return false;
            const int Hcen = cfg_.center_image_size / 2;
            double nx = (px + 0.5) * w[c] / (double)Hcen;
            double ny = (py + 0.5) * h[c] / (double)Hcen;
            center_diag_.push_back({c, nx, ny, v, v >= kCenterDetectThreshold});
            if (v < kCenterDetectThreshold) continue;
            const auto &cp = cams_[c];
            und.push_back(cp.telecentric
                ? red_math::undistortPointTelecentric({nx, ny}, cp.k, cp.dist_coeffs)
                : red_math::undistortPoint({nx, ny}, cp.k, cp.dist_coeffs));
            proj.push_back(cp.projection_mat);
            vals.push_back(v);
        }
        cams_used_ = (int)und.size();
        if (cams_used_ < 2) { std::fprintf(stderr, "[jarvis] only %d cams cleared center\n", cams_used_); return false; }
        center3D = robust_triangulate(und, proj, vals, center_inlier_px_);
        last_center3d_ = center3D;
        return true;
    }

    // RANSAC over camera pairs: a false center-detect in one view (common when
    // the animal is occluded and the model misfires on a corner/reflection)
    // would wreck a plain DLT. Pick the pair whose triangulation has the most
    // inlier rays (undistorted reprojection error < thresh_px), then refine on
    // the inliers. Falls back to all-ray DLT for n==2.
    static Eigen::Vector3d robust_triangulate(
            const std::vector<Eigen::Vector2d> &und,
            const std::vector<Eigen::Matrix<double, 3, 4>> &proj,
            const std::vector<float> &vals, double thresh_px) {
        const int n = (int)und.size();
        if (n == 2) return red_math::triangulatePoints(und, proj);
        auto reproj_err = [&](const Eigen::Vector3d &X, int k) {
            Eigen::Vector4d Xh(X[0], X[1], X[2], 1.0);
            Eigen::Vector3d p = proj[k] * Xh;
            if (std::abs(p[2]) < 1e-9) return 1e30;
            return (Eigen::Vector2d(p[0]/p[2], p[1]/p[2]) - und[k]).norm();
        };
        std::vector<int> best;
        double best_valsum = -1;
        Eigen::Vector3d bestX = red_math::triangulatePoints(und, proj);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                Eigen::Vector3d X = red_math::triangulatePoints(
                    {und[i], und[j]}, {proj[i], proj[j]});
                std::vector<int> in; double valsum = 0;
                for (int k = 0; k < n; ++k)
                    if (reproj_err(X, k) < thresh_px) { in.push_back(k); valsum += vals[k]; }
                // Prefer more inliers; tie-break toward higher-confidence rays
                // (a false detect on an occluded view has a weaker peak).
                if ((int)in.size() > (int)best.size() ||
                    ((int)in.size() == (int)best.size() && valsum > best_valsum)) {
                    best = in; best_valsum = valsum; bestX = X;
                }
            }
        if ((int)best.size() >= 2 && (int)best.size() < n) {
            std::vector<Eigen::Vector2d> u2; std::vector<Eigen::Matrix<double,3,4>> p2;
            for (int k : best) { u2.push_back(und[k]); p2.push_back(proj[k]); }
            bestX = red_math::triangulatePoints(u2, p2);
            std::fprintf(stderr, "[jarvis] robust center: kept %d/%d rays\n", (int)best.size(), n);
        }
        return bestX;
    }

    // Phases 2+3 with a caller-supplied center3D (skips center-detect). Useful
    // as a fallback when center-detect misses (reuse last frame's center) and
    // for deterministic testing.
    bool predict_with_center(const std::vector<const uint8_t *> &rgba_dev,
                             const std::vector<int> &w, const std::vector<int> &h,
                             const Eigen::Vector3d &center3D) {
        const int N = cfg_.num_cameras;
        // Phase 2 — reproject center to each cam, launch ALL efftracks first
        // (4 GPUs compute concurrently), then sync + gather per cam.
        const int bbox_hw = cfg_.keypoint_bbox_size / 2;
        std::vector<int> cx(N), cy(N);
        for (int c = 0; c < N; ++c) {
            const auto &cp = cams_[c];
            Eigen::Vector2d uv = cp.telecentric
                ? red_math::projectPointTelecentric(center3D, cp.projection_mat, cp.k, cp.dist_coeffs)
                : red_math::projectPointR(center3D, cp.r, cp.tvec, cp.k, cp.dist_coeffs);
            int ix = (int)std::lround(uv[0]), iy = (int)std::lround(uv[1]);
            ix = std::max(bbox_hw, std::min(w[c] - bbox_hw, ix));
            iy = std::max(bbox_hw, std::min(h[c] - bbox_hw, iy));
            cx[c] = ix; cy[c] = iy;
            if (!cam2d_[c].launch_efftrack(rgba_dev[c], w[c], h[c], ix, iy)) return false;
        }
        for (int c = 0; c < N; ++c) {
            if (!cam2d_[c].sync_efftrack()) return false;  // cam c heatmap ready
            if (!hybrid3d_.gather_slot(c, cam2d_[c].padded_heatmap(), cam2d_[c].gpu_id())) return false;
        }
        // Phase 3 — central 3D.
        hybrid3d_.assemble_aux(cams_, cx, cy);
        return hybrid3d_.run(center3D);
    }

    const std::vector<float> &points3D() const { return hybrid3d_.points3D(); }
    const std::vector<float> &confidences() const { return hybrid3d_.confidences(); }
    const Config &config() const { return cfg_; }
    int cams_used() const { return cams_used_; }

private:
    Config cfg_;
    std::vector<CameraParams> cams_;
    std::vector<Cam2D> cam2d_;
    Hybrid3D hybrid3d_;
    int cams_used_ = 0;
    double center_inlier_px_ = 60.0;  // RANSAC reprojection-inlier threshold
    std::vector<CenterDiag> center_diag_;
    Eigen::Vector3d last_center3d_ = Eigen::Vector3d::Zero();
};

} // namespace jarvis
#endif // __linux__ || _WIN32
