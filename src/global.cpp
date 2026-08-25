#include "global.h"

std::atomic<CalibState> calib_state{CalibIdle};

DetectedPoses g_detected_poses;