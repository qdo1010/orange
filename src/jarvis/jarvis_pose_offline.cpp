// jarvis_pose_offline.cpp — standalone smoke test for the Option-B distributed
// JARVIS 3D pose pipeline. No Emergent SDK / no live cameras: it loads the real
// TRT engines onto multiple GPUs, synthesizes RGBA frames on each camera's GPU,
// and runs the full per-cam-2D → gather → central-3D path, printing 3D
// keypoints and per-stage timing. Proves the multi-GPU mechanics + engine I/O.
//
// Usage:
//   jarvis_pose_offline <model_dir> <calib_dir> <serial1,serial2,serial3,serial4> [cam_gpus] [central_gpu]
// e.g.
//   jarvis_pose_offline .../mouse_merge_6kp/onnx .../calibration \
//       2002486,2002487,2005325,2006050 0,1,2,3 4
//
// 2D engines (center_detect, hybridnet_efftrack) must be compiled at BATCH=1:
//   HN_BATCH=1 red/scripts/compile_tensorrt_engines.sh <model_dir>

#include "jarvis_pose.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> split(const std::string &s, char d) {
    std::vector<std::string> v; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, d)) if (!t.empty()) v.push_back(t);
    return v;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <model_dir> <calib_dir> <serials csv> [cam_gpus csv] [central_gpu]\n",
            argv[0]);
        return 2;
    }
    std::string model_dir = argv[1];
    std::string calib_dir = argv[2];
    auto serials = split(argv[3], ',');
    std::vector<int> cam_gpus;
    if (argc > 4) for (auto &g : split(argv[4], ',')) cam_gpus.push_back(std::atoi(g.c_str()));
    int central_gpu = (argc > 5) ? std::atoi(argv[5]) : -1;

    const int N = (int)serials.size();
    if ((int)cam_gpus.size() != N) {           // default: cams on GPUs 0..N-1
        cam_gpus.clear();
        for (int c = 0; c < N; ++c) cam_gpus.push_back(c);
    }
    int ngpu = 0; cudaGetDeviceCount(&ngpu);
    if (central_gpu < 0) central_gpu = ngpu - 1;   // default: last GPU (A6000)
    std::printf("GPUs available: %d | cam GPUs:", ngpu);
    for (int g : cam_gpus) std::printf(" %d", g);
    std::printf(" | central (3D) GPU: %d\n", central_gpu);

    // Calibration files in camera order.
    std::vector<std::string> calib_files;
    for (auto &s : serials)
        calib_files.push_back((std::filesystem::path(calib_dir) / ("Cam" + s + ".yaml")).string());
    std::vector<jarvis::CameraParams> cams;
    if (!jarvis::load_calibration(calib_files, cams)) { std::fprintf(stderr, "calib load failed\n"); return 1; }
    for (int c = 0; c < N; ++c) cams[c].gpu_id = cam_gpus[c];

    jarvis::PoseCoordinator coord;
    auto t_load = std::chrono::high_resolution_clock::now();
    if (!coord.load(model_dir, cams, central_gpu)) { std::fprintf(stderr, "coordinator load failed\n"); return 1; }
    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_load).count();
    const auto &cfg = coord.config();
    std::printf("loaded %d cams, %d joints, bbox=%d, roi=%.0fmm, grid=%.0fmm in %.0f ms\n",
                cfg.num_cameras, cfg.num_joints, cfg.keypoint_bbox_size,
                cfg.roi_cube_size_mm, cfg.grid_spacing_mm, load_ms);

    // Synthesize a mid-gray RGBA frame on each camera's GPU.
    std::vector<const uint8_t *> rgba(N);
    std::vector<int> ws(N), hs(N);
    for (int c = 0; c < N; ++c) {
        int w = cams[c].image_width  > 0 ? cams[c].image_width  : 3208;
        int h = cams[c].image_height > 0 ? cams[c].image_height : 2200;
        ws[c] = w; hs[c] = h;
        cudaSetDevice(cam_gpus[c]);
        uint8_t *d = nullptr;
        if (cudaMalloc(&d, (size_t)w * h * 4) != cudaSuccess) { std::fprintf(stderr, "frame malloc failed\n"); return 1; }
        cudaMemset(d, 128, (size_t)w * h * 4);
        rgba[c] = d;
    }

    // Synthetic center3D = mean camera optical center (-R^T t) — roughly the
    // rig center, so each camera's crop lands in-frame. (Real runs use
    // compute_center3d from the live frames.)
    Eigen::Vector3d center3D = Eigen::Vector3d::Zero();
    for (int c = 0; c < N; ++c) center3D += (-cams[c].r.transpose() * cams[c].tvec);
    center3D /= N;
    std::printf("synthetic center3D = [%.1f, %.1f, %.1f] mm\n", center3D[0], center3D[1], center3D[2]);

    // Warmup + timed runs of the distributed pass (phases 2+3 with given center).
    bool ok = coord.predict_with_center(rgba, ws, hs, center3D);
    if (!ok) { std::fprintf(stderr, "predict_with_center failed\n"); return 1; }

    const int ITERS = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i)
        if (!coord.predict_with_center(rgba, ws, hs, center3D)) { std::fprintf(stderr, "iter %d failed\n", i); return 1; }
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

    const auto &p3 = coord.points3D();
    const auto &cf = coord.confidences();
    std::printf("\n=== 3D keypoints (world mm) ===\n");
    for (int j = 0; j < cfg.num_joints; ++j) {
        std::string name = j < (int)cfg.keypoint_names.size() ? cfg.keypoint_names[j] : std::to_string(j);
        std::printf("  %-10s  (%8.1f, %8.1f, %8.1f)  conf=%.3f\n",
                    name.c_str(), p3[j*3+0], p3[j*3+1], p3[j*3+2], cf[j]);
    }
    std::printf("\ndistributed pass: %.2f ms/frame over %d iters "
                "(per-cam 2D on GPUs, gather + 3D on GPU %d)\n", ms, ITERS, central_gpu);
    std::printf("RESULT: OK — multi-GPU distributed pipeline ran end-to-end.\n");

    for (int c = 0; c < N; ++c) { cudaSetDevice(cam_gpus[c]); cudaFree((void *)rgba[c]); }
    return 0;
}
