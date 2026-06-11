// jarvis_calib.cpp — manifest + calibration loading for the JARVIS pose path.
// Isolates the OpenCV (cv::FileStorage) and nlohmann-json dependencies to this
// one translation unit so jarvis_pose.h stays OpenCV-free.
//
// Calibration changes daily (rig is re-calibrated each day), so callers should
// read the folder fresh at startup — never copy/bake it. Point lime's config
// at the live folder (e.g. /home/ratan/src/realtime_jarvis_model/calibration).

#include "jarvis_pose.h"

#include "../json.hpp"      // lime's bundled nlohmann::json
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <cstdio>
#include <fstream>

namespace jarvis {

bool load_manifest(Config &cfg, const std::string &manifest_path) {
    std::ifstream f(manifest_path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[jarvis] cannot open manifest: %s\n", manifest_path.c_str());
        return false;
    }
    nlohmann::json j;
    try { f >> j; } catch (const std::exception &e) {
        std::fprintf(stderr, "[jarvis] manifest parse error: %s\n", e.what());
        return false;
    }
    const auto &s = j.contains("training_config_summary") ? j["training_config_summary"] : j;
    auto get_i = [&](const char *k, int d) { return s.contains(k) ? s[k].get<int>() : d; };
    auto get_f = [&](const char *k, float d) { return s.contains(k) ? s[k].get<float>() : d; };
    cfg.center_image_size = get_i("center_image_size", cfg.center_image_size);
    cfg.keypoint_bbox_size = get_i("keypoint_bbox_size", cfg.keypoint_bbox_size);
    cfg.num_joints = get_i("num_joints", cfg.num_joints);
    cfg.num_cameras = get_i("num_cameras", cfg.num_cameras);
    cfg.roi_cube_size_mm = get_f("roi_cube_size_mm", cfg.roi_cube_size_mm);
    cfg.grid_spacing_mm = get_f("grid_spacing_mm", cfg.grid_spacing_mm);
    if (s.contains("dataset_mean") && s["dataset_mean"].size() == 3)
        for (int i = 0; i < 3; ++i) cfg.dataset_mean[i] = s["dataset_mean"][i].get<float>();
    if (s.contains("dataset_std") && s["dataset_std"].size() == 3)
        for (int i = 0; i < 3; ++i) cfg.dataset_std[i] = s["dataset_std"][i].get<float>();
    if (s.contains("keypoint_names")) {
        cfg.keypoint_names.clear();
        for (const auto &n : s["keypoint_names"]) cfg.keypoint_names.push_back(n.get<std::string>());
        // Config lists more names than joints (24kp leftover) — trust num_joints.
        if ((int)cfg.keypoint_names.size() > cfg.num_joints)
            cfg.keypoint_names.resize(cfg.num_joints);
    }
    return true;
}

static bool mat_to_eigen3(const cv::Mat &m, Eigen::Matrix3d &out) {
    if (m.rows != 3 || m.cols != 3) return false;
    cv::Mat md; m.convertTo(md, CV_64F);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) out(r, c) = md.at<double>(r, c);
    return true;
}

bool load_calibration(const std::vector<std::string> &files,
                      std::vector<CameraParams> &out) {
    out.clear();
    out.reserve(files.size());
    for (const auto &path : files) {
        if (!std::filesystem::exists(path)) {
            std::fprintf(stderr, "[jarvis] calib file missing: %s\n", path.c_str());
            return false;
        }
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::fprintf(stderr, "[jarvis] cannot open calib: %s\n", path.c_str());
            return false;
        }
        cv::Mat k, dist, r, t;
        fs["camera_matrix"] >> k;
        fs["distortion_coefficients"] >> dist;
        fs["rc_ext"] >> r;
        fs["tc_ext"] >> t;
        CameraParams cp;
        if (fs["image_width"].isInt()) cp.image_width = (int)fs["image_width"];
        if (fs["image_height"].isInt()) cp.image_height = (int)fs["image_height"];
        fs.release();
        if (k.empty() || r.empty() || t.empty()) {
            std::fprintf(stderr, "[jarvis] calib %s missing required matrices\n", path.c_str());
            return false;
        }
        if (!mat_to_eigen3(k, cp.k) || !mat_to_eigen3(r, cp.r)) return false;
        cv::Mat td; t.convertTo(td, CV_64F);
        for (int i = 0; i < 3; ++i) cp.tvec[i] = td.at<double>(i);
        cp.dist_coeffs.setZero();
        if (!dist.empty()) {
            cv::Mat dd; dist.convertTo(dd, CV_64F);
            for (int i = 0; i < std::min(5, (int)dd.total()); ++i)
                cp.dist_coeffs[i] = dd.at<double>(i);
        }
        cp.rvec = red_math::rotationMatrixToVector(cp.r);
        cp.projection_mat = red_math::projectionFromKRt(cp.k, cp.r, cp.tvec);
        out.push_back(cp);
    }
    return true;
}

} // namespace jarvis
