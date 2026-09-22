#ifndef POSE_NORMAL_COMMON_HPP
#define POSE_NORMAL_COMMON_HPP
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include <unistd.h>

class Logger : public nvinfer1::ILogger {
  public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity =
                        nvinfer1::ILogger::Severity::kINFO)
        : reportableSeverity(severity) {}

    void log(nvinfer1::ILogger::Severity severity,
             const char *msg) noexcept override {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

inline int get_size_by_dims(const nvinfer1::Dims &dims) {
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType &dataType) {
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        return 4;
    }
}

inline static float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}

inline bool IsPathExist(const std::string &path) {
    if (access(path.c_str(), 0) == F_OK) {
        return true;
    }
    return false;
}

inline bool IsFile(const std::string &path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

inline bool IsFolder(const std::string &path) {
    if (!IsPathExist(path)) {
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

struct Binding {
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    nvinfer1::Dims dims;
    std::string name;
};

struct Bbox {
    cv::Rect_<float> rect;   // axis-aligned box (image px). For OBB models this is
                             // the bounding rect of the rotated box.
    int label = 0;
    float prob = 0.0;
    std::vector<float> kps;
    std::vector<float> mask_coeffs;  // 32 mask prototype coefficients (seg models)

    // Native oriented box, filled only by YOLO-OBB engines (has_obb == true).
    // (cx, cy) center and (rw, rh) size in image px with rw >= rh.
    // theta_deg: direction of the rw (long) axis in image coordinates
    // (x right, y down), measured from +x toward +y, in [0, 180).
    bool has_obb = false;
    float cx = 0.f, cy = 0.f, rw = 0.f, rh = 0.f;
    float theta_deg = 0.f;
};

// Corners of an oriented Bbox as x0,y0,...,x3,y3 (clockwise on screen).
inline void bbox_obb_corners(const Bbox &b, float out[8]) {
    const float t = b.theta_deg * (float)M_PI / 180.f;
    const float c = std::cos(t), s = std::sin(t);
    const float ax = 0.5f * b.rw * c, ay = 0.5f * b.rw * s;   // half long axis
    const float bx = -0.5f * b.rh * s, by = 0.5f * b.rh * c;  // half short axis
    out[0] = b.cx - ax - bx; out[1] = b.cy - ay - by;
    out[2] = b.cx + ax - bx; out[3] = b.cy + ay - by;
    out[4] = b.cx + ax + bx; out[5] = b.cy + ay + by;
    out[6] = b.cx - ax + bx; out[7] = b.cy - ay + by;
}

struct PreParam {
    float ratio = 1.0f;
    float dw = 0.0f;
    float dh = 0.0f;
    float height = 0;
    float width = 0;
};
#endif // POSE_NORMAL_COMMON_HPP
