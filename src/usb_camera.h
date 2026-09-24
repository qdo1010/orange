#ifndef ORANGE_USB_CAMERA
#define ORANGE_USB_CAMERA
// USB (UVC / V4L2) camera support. USB cameras are listed alongside the
// Emergent GigE cameras by appending entries to the GigEVisionDeviceInfo
// array (modelName == USB_CAMERA_MODEL_NAME, currentIp holds the /dev/videoN
// path). Their CameraParams carry is_usb = true, which makes the Emergent
// lifecycle functions in camera.cpp no-ops, and start_camera_streaming runs
// acquire_frames_usb instead of acquire_frames.
//
// Frames are converted to 8-bit for display / NVENC mp4 like the other
// cameras. 16-bit sources (Y16) are additionally recorded losslessly to
// Cam<serial>_raw16.bin with a per-frame metadata csv and a json header.
#include "camera.h"
#include "video_capture.h"

#define USB_CAMERA_MODEL_NAME "USB-UVC"

// Appends detected USB cameras to device_info. Returns the number added.
int scan_usb_cameras(GigEVisionDeviceInfo *device_info, int max_cameras);

bool is_usb_device_info(const GigEVisionDeviceInfo *device_info);

// Sets up camera_params for a USB camera and negotiates the format with the
// device. With from_config, width/height/frame_rate/gpu_id already loaded
// from the camera's json are requested; otherwise the device's current
// format is used. The negotiated values are written back.
bool init_usb_camera_params(CameraParams *camera_params,
                            GigEVisionDeviceInfo *device_info, int camera_idx,
                            int num_cameras, bool from_config);

// Stops the device's stream (started by init_usb_camera_params). Called by
// close_camera. Streaming runs continuously in between because the FT602
// camera stops sending frames after a stream stop/restart.
void usb_camera_release(const std::string &device);

int count_usb_cameras(CameraParams *cameras_params, int num_cameras);

// Index of the first Emergent camera, or -1 if all cameras are USB.
int first_emergent_camera(CameraParams *cameras_params, int num_cameras);

// PTP time from the first Emergent camera, used to schedule synced start and
// stop. Falls back to host CLOCK_REALTIME when every camera is USB.
unsigned long long reference_ptp_time(CameraEmergent *ecams,
                                      CameraParams *cameras_params,
                                      int num_cameras);

void acquire_frames_usb(CameraEmergent *ecam, CameraParams *camera_params,
                        CameraEachSelect *camera_select,
                        CameraControl *camera_control,
                        unsigned char *display_buffer,
                        std::string encoder_setup, std::string folder_name,
                        PTPParams *ptp_params,
                        INDIGOSignalBuilder *indigo_signal_builder);
#endif
