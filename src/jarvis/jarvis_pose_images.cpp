// jarvis_pose_images.cpp — run the full JARVIS 3D pose pipeline on real frames
// (one image per camera) and report tracking quality: per-cam center peaks, 3D
// keypoints + confidences, and 3D→2D reprojection overlays saved to disk.
//
// Usage:
//   jarvis_pose_images <model_dir> <calib_dir> <out_dir> \
//       <serial1:img1> <serial2:img2> <serial3:img3> <serial4:img4> [cam_gpus] [central_gpu]

#include "jarvis_pose.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string &s, char d) {
    std::vector<std::string> v; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, d)) if (!t.empty()) v.push_back(t);
    return v;
}

int main(int argc, char **argv) {
    if (argc < 8) {
        std::fprintf(stderr, "usage: %s <model_dir> <calib_dir> <out_dir> "
            "<serial:img> x4 [cam_gpus] [central_gpu]\n", argv[0]);
        return 2;
    }
    std::string model_dir = argv[1], calib_dir = argv[2], out_dir = argv[3];
    std::vector<std::string> serials, imgs;
    int firstgpu_arg = -1;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        auto colon = a.find(':');
        if (colon != std::string::npos && a.find(".png") != std::string::npos) {
            serials.push_back(a.substr(0, colon));
            imgs.push_back(a.substr(colon + 1));
        } else { firstgpu_arg = i; break; }
    }
    const int N = (int)serials.size();
    std::vector<int> cam_gpus;
    int central_gpu = -1;
    if (firstgpu_arg > 0) {
        for (auto &g : split(argv[firstgpu_arg], ',')) cam_gpus.push_back(std::atoi(g.c_str()));
        if (firstgpu_arg + 1 < argc) central_gpu = std::atoi(argv[firstgpu_arg + 1]);
    }
    if ((int)cam_gpus.size() != N) { cam_gpus.clear(); for (int c = 0; c < N; ++c) cam_gpus.push_back(c); }
    int ngpu = 0; cudaGetDeviceCount(&ngpu);
    if (central_gpu < 0) central_gpu = ngpu - 1;

    // Calib (fresh from folder).
    std::vector<std::string> calib_files;
    for (auto &s : serials)
        calib_files.push_back((std::filesystem::path(calib_dir) / ("Cam" + s + ".yaml")).string());
    std::vector<jarvis::CameraParams> cams;
    if (!jarvis::load_calibration(calib_files, cams)) return 1;
    for (int c = 0; c < N; ++c) cams[c].gpu_id = cam_gpus[c];

    jarvis::PoseCoordinator coord;
    if (!coord.load(model_dir, cams, central_gpu)) { std::fprintf(stderr, "load failed\n"); return 1; }
    const auto &cfg = coord.config();

    // Load each frame, BGR->RGBA, upload to its camera's GPU.
    std::vector<cv::Mat> frames(N);
    std::vector<const uint8_t *> rgba(N);
    std::vector<int> ws(N), hs(N);
    for (int c = 0; c < N; ++c) {
        cv::Mat img = cv::imread(imgs[c], cv::IMREAD_COLOR);  // BGR, 3ch
        if (img.empty()) { std::fprintf(stderr, "cannot read %s\n", imgs[c].c_str()); return 1; }
        frames[c] = img;
        cv::Mat rgba_h; cv::cvtColor(img, rgba_h, cv::COLOR_BGR2RGBA);  // R,G,B,A
        ws[c] = img.cols; hs[c] = img.rows;
        cudaSetDevice(cam_gpus[c]);
        uint8_t *d = nullptr;
        size_t bytes = (size_t)img.cols * img.rows * 4;
        if (cudaMalloc(&d, bytes) != cudaSuccess) return 1;
        cudaMemcpy(d, rgba_h.data, bytes, cudaMemcpyHostToDevice);
        rgba[c] = d;
        std::printf("cam %d (Cam%s) %dx%d on GPU %d\n", c, serials[c].c_str(), img.cols, img.rows, cam_gpus[c]);
    }

    // Full pipeline with REAL center-detect + triangulation.
    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = coord.predict(rgba, ws, hs);
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::printf("\npredict: %s  (%d/%d cams cleared center, %.1f ms)\n",
                ok ? "OK" : "FAILED", coord.cams_used(), N, ms);
    if (!ok) {
        std::fprintf(stderr, "center-detect found <2 cams above threshold — the "
            "mouse model may not fire on this rat frame, or the animal is small.\n");
        return 1;
    }

    // Per-cam center-detect diagnostics: where did CenterDetect fire?
    std::printf("\n=== center-detect (per cam) ===\n");
    for (const auto &d : coord.center_diag())
        std::printf("  cam %d (Cam%s): peak img=(%.0f, %.0f) val=%.1f %s\n",
                    d.cam, serials[d.cam].c_str(), d.img_x, d.img_y, d.val,
                    d.passed ? "PASS" : "below-threshold");
    Eigen::Vector3d c3 = coord.last_center3d();
    std::printf("triangulated center3D = (%.1f, %.1f, %.1f) mm\n", c3[0], c3[1], c3[2]);
    // Calibration sanity: reproject center3D to each passing cam and compare to
    // its detected peak. Low error on good cams ⇒ calibration is self-consistent.
    std::printf("=== center3D reprojection error vs CenterDetect peak ===\n");
    for (const auto &d : coord.center_diag()) {
        if (!d.passed) continue;
        const auto &cp = cams[d.cam];
        Eigen::Vector2d uv = cp.telecentric
            ? red_math::projectPointTelecentric(c3, cp.projection_mat, cp.k, cp.dist_coeffs)
            : red_math::projectPointR(c3, cp.r, cp.tvec, cp.k, cp.dist_coeffs);
        double err = std::hypot(uv[0] - d.img_x, uv[1] - d.img_y);
        std::printf("  cam %d (Cam%s): reproj=(%.0f,%.0f) peak=(%.0f,%.0f) err=%.0f px\n",
                    d.cam, serials[d.cam].c_str(), uv[0], uv[1], d.img_x, d.img_y, err);
    }

    const auto &p3 = coord.points3D();
    const auto &cf = coord.confidences();
    std::printf("\n=== 3D keypoints (world mm) ===\n");
    double meanconf = 0;
    for (int j = 0; j < cfg.num_joints; ++j) {
        std::string nm = j < (int)cfg.keypoint_names.size() ? cfg.keypoint_names[j] : std::to_string(j);
        std::printf("  %-10s (%8.1f, %8.1f, %8.1f)  conf=%.3f\n",
                    nm.c_str(), p3[j*3+0], p3[j*3+1], p3[j*3+2], cf[j]);
        meanconf += cf[j];
    }
    std::printf("mean confidence: %.3f\n", meanconf / cfg.num_joints);

    // Skeleton edges (mouse_merge_6kp): Snout-EarL, Snout-EarR, EarL-Neck,
    // EarR-Neck, Neck-SpineL, SpineL-TailBase.
    int edges[][2] = {{0,1},{0,2},{1,3},{2,3},{3,4},{4,5}};
    std::filesystem::create_directories(out_dir);
    std::vector<cv::Scalar> colors = {{0,0,255},{0,165,255},{0,255,255},{0,255,0},{255,0,0},{255,0,255}};
    for (int c = 0; c < N; ++c) {
        cv::Mat vis = frames[c].clone();
        const auto &cp = cams[c];
        auto proj = [&](const Eigen::Vector3d &P) {
            Eigen::Vector2d uv = cp.telecentric
                ? red_math::projectPointTelecentric(P, cp.projection_mat, cp.k, cp.dist_coeffs)
                : red_math::projectPointR(P, cp.r, cp.tvec, cp.k, cp.dist_coeffs);
            return cv::Point((int)std::lround(uv[0]), (int)std::lround(uv[1]));
        };
        // CenterDetect peak (cyan cross) — where the center stage fired.
        for (const auto &d : coord.center_diag())
            if (d.cam == c && d.passed) {
                cv::Point pk((int)d.img_x, (int)d.img_y);
                cv::drawMarker(vis, pk, {255,255,0}, cv::MARKER_TILTED_CROSS, 60, 4);
            }
        // Triangulated center3D (white circle).
        cv::circle(vis, proj(c3), 22, {255,255,255}, 3);
        // Skeleton edges + keypoints.
        std::vector<cv::Point> kp(cfg.num_joints);
        for (int j = 0; j < cfg.num_joints; ++j)
            kp[j] = proj(Eigen::Vector3d(p3[j*3+0], p3[j*3+1], p3[j*3+2]));
        for (auto &e : edges)
            if (e[0] < cfg.num_joints && e[1] < cfg.num_joints)
                cv::line(vis, kp[e[0]], kp[e[1]], {200,200,200}, 2);
        for (int j = 0; j < cfg.num_joints; ++j) {
            cv::circle(vis, kp[j], 14, colors[j % colors.size()], -1);
            cv::putText(vis, std::to_string(j), kp[j] + cv::Point(16, 6),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, colors[j % colors.size()], 3);
        }
        std::string outp = (std::filesystem::path(out_dir) / ("annot_Cam" + serials[c] + ".png")).string();
        cv::imwrite(outp, vis);
        // Zoomed crop around the center3D reprojection so the animal is visible.
        cv::Point ctr = proj(c3);
        int half = 400;
        cv::Rect roi(std::max(0, ctr.x - half), std::max(0, ctr.y - half), 2*half, 2*half);
        roi &= cv::Rect(0, 0, vis.cols, vis.rows);
        if (roi.width > 10 && roi.height > 10) {
            std::string zp = (std::filesystem::path(out_dir) / ("zoom_Cam" + serials[c] + ".png")).string();
            cv::imwrite(zp, vis(roi));
        }
        std::printf("wrote %s (+zoom)\n", outp.c_str());
    }
    for (int c = 0; c < N; ++c) { cudaSetDevice(cam_gpus[c]); cudaFree((void *)rgba[c]); }
    return 0;
}
