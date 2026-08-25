#ifndef GLOBAL_H
#define GLOBAL_H

#include <atomic>
#include <mutex>
#include <cstdint>

// Calibration state enumeration
enum CalibState {
    CalibIdle,
    CalibNextPose,
    CalibPoseReached,
    CalibSavePictures
};

inline const char * const *enum_names_calib_state() {
    static const char * const names[5] = {
        "Idle",
        "NextPose",
        "PoseReached",
        "SavePictures",
        nullptr
      };
    return names;
}

// Atomic calibration state (now using CalibState instead of int)
extern std::atomic<CalibState> calib_state;

// Latest YOLO detections in image pixels, written by the display thread and
// read by the enet thread, which forwards them to indigo (cbot) on channel 1.
// x/y stay at the -1000 sentinel when the object is not detected.
struct DetectedPose {
    float x = -1000.0f;
    float y = -1000.0f;
    float prob = 0.0f;
    bool valid = false;
};

struct DetectedPoses {
    std::mutex mtx;
    DetectedPose ball;
    DetectedPose mouse;
    uint64_t seq = 0; // bumped on every yolo frame; lets the sender skip stale data
};

extern DetectedPoses g_detected_poses;

#endif // GLOBAL_H
