#ifndef ORANGE_VIDEO_CAPTURE
#define ORANGE_VIDEO_CAPTURE
#include "camera.h"
#include "network_base.h"
#include <atomic>
#include <mutex>
#include <vector>

class LabJackTrigger;

enum PictureState {
    State_Frame_Idle,
    State_Copy_New_Frame,
    State_Frame_Copy_Done,
    State_Frame_Detection_Ready
};

// Pending remote focus + frame grab request (set by ENet thread, consumed by
// camera thread in start_ptp_sync wait loop).
struct SetFocusRequest {
    std::atomic<int> generation{0};
    int focus_value{0};
    std::string camera_serial;
    // Reply: camera thread writes JPEG here, ENet thread sends it
    std::mutex reply_mu;
    std::vector<uint8_t> reply_jpeg;
    bool reply_ready{false};
    ENetPeer *reply_peer{nullptr};
};

struct CameraControl {
    bool open = false;
    bool subscribe = false;
    bool stop_record = false;
    bool record_video = false;
    bool sync_camera = false;
    bool trigger_mode = false;
    bool lj_trigger_mode = false;
    LabJackTrigger *lj_trigger = nullptr;
    // Number of IR frames the camera bursts per LJ edge. 1 = strict 1:1 sync
    // (IR rate matches microscope, exposure inside flyback). 2/4 = 1:2/1:4
    // (IR rate = N×microscope rate, frames 2..N land mid-scan).
    int lj_frames_per_edge = 1;
    // First-frame edge label set by master at STARTRECORDING. All cameras
    // init their per-thread last_lj_edge = lj_start_edge - 1 so the first
    // wait_for_next_edge returns exactly lj_start_edge for everyone — kills
    // the cross-camera 1-edge labeling race at recording start.
    uint64_t lj_start_edge = 0;
    // Microseconds to sleep between detecting the LJ falling edge and
    // calling TriggerSoftware. With PtpMode=Off the actual firmware
    // latency from TriggerSoftware to shutter open is ~1-2 ms, so an
    // offset of 0 fires the LED inside the current flyback (0-2.16 ms).
    // A non-zero offset would push the LED into the scan period AND
    // stretch the per-frame iteration past one scanner cycle, causing
    // client cameras to drop every other (or every third) edge because
    // GetFrame doesn't return in time to wait for the next edge.
    int lj_trigger_offset_us = 0;
    std::atomic<int> focus_test_generation{0};
    SetFocusRequest setfocus;
};

enum DetectMode {
    Detect_OFF,
    Detect2D_GLThread,
    Detect2D_Standoff,
    Detect3D_Standoff
};
constexpr const char *DetectModeNames[] = {"OFF", "2DGLThread", "2DStandoff",
                                           "3DStandoff"};
struct CameraEachSelect {
    bool stream_on = true;
    bool record = true;
    int downsample = 1;
    std::atomic<PictureState> frame_save_state;
    std::string frame_save_format;
    std::string frame_save_name;
    int pictures_counter = 0;
    bool selected_to_save = false;
    std::string picture_save_folder;
    std::string yolo_model;
    DetectMode detect_mode = Detect_OFF;
    int idx2d = 0;
    int idx3d = 0;
    int total_standoff_detector = 0;
    std::atomic<PictureState> frame_detect_state;
    
    // OBB Detection Configuration
    bool enable_obb = false;
    std::string obb_csv_path = "";
    float obb_threshold = 30.0f;
    int obb_bg_frames = 10;

    // Tracks which focus_test_generation was last processed post-recording
    int focus_test_gen_processed{0};

    CameraEachSelect()
        : frame_save_state(State_Frame_Idle),
          frame_detect_state(State_Frame_Idle) {}
};

struct CameraState {
    int camera_return = 0;
    unsigned short id_prev = 0;
    unsigned short dropped_frames = 0;
    unsigned int frames_recd = 0;
    unsigned long long frame_count = 0;
    unsigned int encoder_drops = 0;
};

struct PTPState {
    int ptp_offset;
    int ptp_offset_sum = 0;
    int ptp_offset_prev = 0;
    unsigned int ptp_time_low;
    unsigned int ptp_time_high;
    unsigned int ptp_time_plus_delta_to_start_low;
    unsigned int ptp_time_plus_delta_to_start_high;
    unsigned long long ptp_time_delta_sum = 0;
    unsigned long long ptp_time_delta;
    unsigned long long ptp_time;
    unsigned long long ptp_time_prev;
    unsigned long long ptp_time_countdown;
    unsigned long long frame_ts;
    unsigned long long frame_ts_prev;
    unsigned long long frame_ts_delta;
    unsigned long long frame_ts_delta_sum = 0;
    unsigned long long ptp_time_plus_delta_to_start;
    char ptp_status[100];
    unsigned long ptp_status_sz_ret;
    unsigned int ptp_time_plus_delta_to_start_uint;
};

void report_statistics(CameraParams *camera_params, CameraState *camera_state,
                       double time_diff);
void show_ptp_offset(PTPState *ptp_state, CameraEmergent *ecam);
void show_lj_edge_status(LabJackTrigger *lj_trigger, const char *cam_serial);
class MjpegServer; // forward declaration
void start_ptp_sync(PTPState *ptp_state, PTPParams *ptp_params,
                    CameraParams *camera_params, CameraEmergent *ecam,
                    unsigned int delay_in_second,
                    CameraControl *camera_control = nullptr,
                    CameraEachSelect *camera_select = nullptr,
                    MjpegServer *mjpeg_server = nullptr);
void grab_frames_after_countdown(PTPState *ptp_state, CameraEmergent *ecam);
bool try_start_timer();
bool try_stop_timer();
void acquire_frames(CameraEmergent *ecam, CameraParams *camera_params,
                    CameraEachSelect *camera_select,
                    CameraControl *camera_control,
                    unsigned char *display_buffer, std::string encoder_setup,
                    std::string folder_name, PTPParams *ptp_params,
                    INDIGOSignalBuilder *indigo_signal_builder);
#endif
