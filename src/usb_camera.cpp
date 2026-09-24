#include "usb_camera.h"
#include "FrameSaver.h"
#include "global.h"
#include "gpu_video_encoder.h"
#include "mjpeg_stream.h"
#include "utils.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cuda_runtime.h>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <linux/videodev2.h>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#ifndef HEADLESS
#include "FrameDetector.h"
#include "opengldisplay.h"
#endif

// Number of 8-bit (and 16-bit raw) host frame slots. Downstream consumers
// (encoder queue, display, raw writer) hold pointers into these slots, same
// as the Emergent path's evt_buffer_size ring.
static constexpr int USB_RING_SIZE = 100;
static constexpr int USB_V4L2_BUFFERS = 8;

static int xioctl(int fd, unsigned long request, void *arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

static bool is_supported_pixfmt(uint32_t fourcc) {
    return fourcc == V4L2_PIX_FMT_Y16 || fourcc == V4L2_PIX_FMT_GREY ||
           fourcc == V4L2_PIX_FMT_YUYV;
}

static std::string fourcc_to_string(uint32_t fourcc) {
    std::string s;
    for (int i = 0; i < 4; i++)
        s += (char)((fourcc >> (8 * i)) & 0xFF);
    return s;
}

static uint64_t realtime_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static std::string read_sysfs_line(const std::string &path) {
    std::ifstream f(path);
    std::string line;
    if (f.is_open())
        std::getline(f, line);
    return line;
}

// Picks a supported pixel format (the current one if possible) and reports
// the current geometry and frame rate.
static bool probe_usb_format(int fd, uint32_t *fourcc, unsigned int *width,
                             unsigned int *height, unsigned int *fps) {
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_G_FMT, &fmt) != 0)
        return false;

    if (!is_supported_pixfmt(fmt.fmt.pix.pixelformat)) {
        bool found = false;
        v4l2_fmtdesc desc{};
        desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        for (desc.index = 0; xioctl(fd, VIDIOC_ENUM_FMT, &desc) == 0;
             desc.index++) {
            if (is_supported_pixfmt(desc.pixelformat)) {
                fmt.fmt.pix.pixelformat = desc.pixelformat;
                found = true;
                break;
            }
        }
        if (!found || xioctl(fd, VIDIOC_TRY_FMT, &fmt) != 0)
            return false;
    }
    *fourcc = fmt.fmt.pix.pixelformat;
    *width = fmt.fmt.pix.width;
    *height = fmt.fmt.pix.height;

    *fps = 30;
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_G_PARM, &parm) == 0 &&
        parm.parm.capture.timeperframe.numerator > 0) {
        *fps = parm.parm.capture.timeperframe.denominator /
               parm.parm.capture.timeperframe.numerator;
    }
    return true;
}

// Applies camera_params' pixel format / size / frame rate to the device and
// writes back what the driver actually accepted.
static bool apply_usb_format(int fd, CameraParams *camera_params,
                             unsigned int *bytes_per_line) {
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.pixelformat = camera_params->usb_pixfmt;
    fmt.fmt.pix.width = camera_params->width;
    fmt.fmt.pix.height = camera_params->height;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) != 0) {
        printf("USB cam %s: VIDIOC_S_FMT failed: %s\n",
               camera_params->camera_serial.c_str(), strerror(errno));
        return false;
    }
    if (fmt.fmt.pix.pixelformat != camera_params->usb_pixfmt) {
        printf("USB cam %s: driver refused pixel format %s\n",
               camera_params->camera_serial.c_str(),
               fourcc_to_string(camera_params->usb_pixfmt).c_str());
        return false;
    }
    camera_params->width = fmt.fmt.pix.width;
    camera_params->height = fmt.fmt.pix.height;
    if (bytes_per_line)
        *bytes_per_line = fmt.fmt.pix.bytesperline;

    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = camera_params->frame_rate;
    if (xioctl(fd, VIDIOC_S_PARM, &parm) == 0 &&
        parm.parm.capture.timeperframe.numerator > 0) {
        camera_params->frame_rate =
            parm.parm.capture.timeperframe.denominator /
            parm.parm.capture.timeperframe.numerator;
    }
    return true;
}

bool is_usb_device_info(const GigEVisionDeviceInfo *device_info) {
    return strcmp(device_info->modelName, USB_CAMERA_MODEL_NAME) == 0;
}

int scan_usb_cameras(GigEVisionDeviceInfo *device_info, int max_cameras) {
    int found = 0;
    for (int n = 0; n < 64 && found < max_cameras; n++) {
        std::string dev = "/dev/video" + std::to_string(n);
        if (access(dev.c_str(), F_OK) != 0)
            continue;
        int fd = open(dev.c_str(), O_RDWR | O_NONBLOCK);
        if (fd < 0) {
            printf("USB camera: cannot open %s: %s\n", dev.c_str(),
                   strerror(errno));
            continue;
        }

        v4l2_capability cap{};
        uint32_t fourcc = 0;
        unsigned int width = 0, height = 0, fps = 0;
        bool ok = xioctl(fd, VIDIOC_QUERYCAP, &cap) == 0;
        if (ok) {
            uint32_t caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS)
                                ? cap.device_caps
                                : cap.capabilities;
            // Metadata nodes (e.g. the second node uvcvideo creates) have no
            // VIDEO_CAPTURE capability and are skipped here.
            ok = (caps & V4L2_CAP_VIDEO_CAPTURE) && (caps & V4L2_CAP_STREAMING);
        }
        ok = ok && probe_usb_format(fd, &fourcc, &width, &height, &fps);
        close(fd);
        if (!ok)
            continue;

        // Serial from the USB device (stable across replugs, unlike the
        // videoN index), trimmed to fit GigEVisionDeviceInfo::serialNumber.
        std::string usb_serial = read_sysfs_line(
            "/sys/class/video4linux/video" + std::to_string(n) +
            "/device/../serial");
        usb_serial.erase(std::remove_if(usb_serial.begin(), usb_serial.end(),
                                        [](char c) { return !isalnum(c); }),
                         usb_serial.end());
        if (usb_serial.size() > 8)
            usb_serial = usb_serial.substr(usb_serial.size() - 8);
        std::string serial = usb_serial.empty() ? "USBvideo" + std::to_string(n)
                                                : "USB" + usb_serial;
        for (int k = 0; k < found; k++) {
            if (serial == device_info[k].serialNumber) {
                serial += "_" + std::to_string(n);
                break;
            }
        }

        GigEVisionDeviceInfo &info = device_info[found];
        memset(&info, 0, sizeof(info));
        snprintf(info.serialNumber, sizeof(info.serialNumber), "%s",
                 serial.c_str());
        snprintf(info.currentIp, sizeof(info.currentIp), "%s", dev.c_str());
        snprintf(info.modelName, sizeof(info.modelName), "%s",
                 USB_CAMERA_MODEL_NAME);
        snprintf(info.manufacturerName, sizeof(info.manufacturerName), "%s",
                 (const char *)cap.driver);
        snprintf(info.manufacturerSpecifiedInfo,
                 sizeof(info.manufacturerSpecifiedInfo), "%s",
                 (const char *)cap.card);

        printf("USB camera: %s \"%s\" %ux%u %s @ %u fps -> serial %s\n",
               dev.c_str(), (const char *)cap.card, width, height,
               fourcc_to_string(fourcc).c_str(), fps, info.serialNumber);
        found++;
    }
    return found;
}

static bool usb_camera_start(CameraParams *camera_params);

bool init_usb_camera_params(CameraParams *camera_params,
                            GigEVisionDeviceInfo *device_info, int camera_idx,
                            int num_cameras, bool from_config) {
    camera_params->is_usb = true;
    camera_params->usb_device = device_info->currentIp;
    camera_params->camera_id = camera_idx;
    camera_params->num_cameras = num_cameras;
    // Frames reach display/encoder as 8-bit mono host buffers.
    camera_params->pixel_format = "Mono8";
    camera_params->color = false;
    camera_params->gpu_direct = false;
    camera_params->need_reorder = false;

    int fd = open(camera_params->usb_device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        printf("USB cam %s: cannot open %s: %s\n",
               camera_params->camera_serial.c_str(),
               camera_params->usb_device.c_str(), strerror(errno));
        return false;
    }
    uint32_t fourcc;
    unsigned int width, height, fps;
    if (!probe_usb_format(fd, &fourcc, &width, &height, &fps)) {
        close(fd);
        return false;
    }
    camera_params->usb_pixfmt = fourcc;
    if (!from_config) {
        camera_params->width = width;
        camera_params->height = height;
        camera_params->frame_rate = fps;
        camera_params->gpu_id = 0;
        camera_params->gop = 1;
        camera_params->color_temp = "";
    }
    close(fd);
    // Streaming starts here, at camera open, and runs until close_camera so
    // the GL textures (allocated before acquisition) match the negotiated
    // size, and the device is never stopped between sessions.
    bool ok = usb_camera_start(camera_params);

    // No adjustable Emergent-style properties; pin the slider ranges.
    camera_params->gain = camera_params->exposure = 0;
    camera_params->focus = camera_params->iris = 0;
    camera_params->offsetx = camera_params->offsety = 0;
    camera_params->width_min = camera_params->width_max = camera_params->width;
    camera_params->height_min = camera_params->height_max =
        camera_params->height;
    camera_params->frame_rate_min = camera_params->frame_rate_max =
        camera_params->frame_rate;
    camera_params->sens_temp = 0;

    printf("USB cam %s: %s %ux%u %s @ %u fps\n",
           camera_params->camera_serial.c_str(),
           camera_params->usb_device.c_str(), camera_params->width,
           camera_params->height, fourcc_to_string(fourcc).c_str(),
           camera_params->frame_rate);
    return ok;
}

int count_usb_cameras(CameraParams *cameras_params, int num_cameras) {
    int n = 0;
    for (int i = 0; i < num_cameras; i++)
        if (cameras_params[i].is_usb)
            n++;
    return n;
}

int first_emergent_camera(CameraParams *cameras_params, int num_cameras) {
    for (int i = 0; i < num_cameras; i++)
        if (!cameras_params[i].is_usb)
            return i;
    return -1;
}

unsigned long long reference_ptp_time(CameraEmergent *ecams,
                                      CameraParams *cameras_params,
                                      int num_cameras) {
    int idx = first_emergent_camera(cameras_params, num_cameras);
    if (idx < 0)
        return realtime_ns();
    return get_current_PTP_time(&ecams[idx].camera);
}

// Converts one V4L2 frame to 8-bit mono. 16-bit input is scaled by a peak
// that rises instantly and decays slowly, so brightness stays stable across
// frames regardless of how many bits the source actually uses.
static void convert_to_mono8(const uint8_t *src, unsigned int bytes_per_line,
                             uint32_t fourcc, unsigned int width,
                             unsigned int height, uint8_t *dst,
                             uint32_t *display_peak) {
    if (fourcc == V4L2_PIX_FMT_GREY) {
        for (unsigned int y = 0; y < height; y++)
            memcpy(dst + y * width, src + y * bytes_per_line, width);
    } else if (fourcc == V4L2_PIX_FMT_YUYV) {
        for (unsigned int y = 0; y < height; y++) {
            const uint8_t *row = src + y * bytes_per_line;
            for (unsigned int x = 0; x < width; x++)
                dst[y * width + x] = row[2 * x];
        }
    } else { // V4L2_PIX_FMT_Y16, little endian
        uint32_t frame_max = 0;
        for (unsigned int y = 0; y < height; y++) {
            const uint16_t *row = (const uint16_t *)(src + y * bytes_per_line);
            for (unsigned int x = 0; x < width; x += 4)
                frame_max = std::max<uint32_t>(frame_max, row[x]);
        }
        frame_max = std::max<uint32_t>(frame_max, 255);
        uint32_t decayed = (*display_peak * 63) / 64;
        *display_peak = std::max(frame_max, decayed);
        uint32_t scale = (255u << 16) / *display_peak;
        for (unsigned int y = 0; y < height; y++) {
            const uint16_t *row = (const uint16_t *)(src + y * bytes_per_line);
            uint8_t *out = dst + y * width;
            for (unsigned int x = 0; x < width; x++)
                out[x] = (uint8_t)std::min<uint32_t>(255, (row[x] * scale) >> 16);
        }
    }
}

// Lossless 16-bit recording on its own thread so disk stalls never block
// frame capture. Frames are referenced by ring slot; a frame is dropped (and
// counted) rather than overwriting a slot the writer has not consumed yet.
class RawFrameWriter {
  public:
    RawFrameWriter(const std::string &prefix, CameraParams *camera_params,
                   std::vector<uint16_t> *ring)
        : ring_(ring), frame_pixels_(camera_params->width *
                                     camera_params->height) {
        bin_ = fopen((prefix + "_raw16.bin").c_str(), "wb");
        meta_ = fopen((prefix + "_raw16_meta.csv").c_str(), "w");
        if (bin_)
            setvbuf(bin_, nullptr, _IOFBF, 8 << 20);
        if (meta_)
            fprintf(meta_, "frame_id,v4l2_sequence,timestamp_mono_ns,"
                           "timestamp_sys_ns\n");
        FILE *header = fopen((prefix + "_raw16.json").c_str(), "w");
        if (header) {
            fprintf(header,
                    "{\n  \"width\": %u,\n  \"height\": %u,\n"
                    "  \"dtype\": \"uint16\",\n  \"byte_order\": \"little\",\n"
                    "  \"frame_rate\": %u,\n  \"device\": \"%s\",\n"
                    "  \"serial\": \"%s\"\n}\n",
                    camera_params->width, camera_params->height,
                    camera_params->frame_rate,
                    camera_params->usb_device.c_str(),
                    camera_params->camera_serial.c_str());
            fclose(header);
        }
        if (!bin_ || !meta_)
            printf("USB cam %s: cannot create raw16 files at %s\n",
                   camera_params->camera_serial.c_str(), prefix.c_str());
        thread_ = std::thread([this] { run(); });
    }

    ~RawFrameWriter() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_one();
        thread_.join();
        if (bin_)
            fclose(bin_);
        if (meta_)
            fclose(meta_);
    }

    // True if the slot may be overwritten with a new frame for the writer.
    bool can_accept() const {
        return outstanding_.load() < USB_RING_SIZE - 1;
    }

    void push(int slot, uint64_t frame_id, uint32_t sequence,
              uint64_t ts_mono_ns, uint64_t ts_sys_ns) {
        outstanding_++;
        {
            std::lock_guard<std::mutex> lk(mu_);
            pending_.push_back({slot, frame_id, sequence, ts_mono_ns,
                                ts_sys_ns});
        }
        cv_.notify_one();
    }

    uint64_t drops = 0;

  private:
    struct Item {
        int slot;
        uint64_t frame_id;
        uint32_t sequence;
        uint64_t ts_mono_ns;
        uint64_t ts_sys_ns;
    };

    void run() {
        while (true) {
            Item item;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !pending_.empty(); });
                if (pending_.empty())
                    return;
                item = pending_.front();
                pending_.pop_front();
            }
            if (bin_)
                fwrite(ring_[item.slot].data(), sizeof(uint16_t),
                       frame_pixels_, bin_);
            if (meta_)
                fprintf(meta_, "%lu,%u,%lu,%lu\n", item.frame_id,
                        item.sequence, item.ts_mono_ns, item.ts_sys_ns);
            outstanding_--;
        }
    }

    std::vector<uint16_t> *ring_;
    size_t frame_pixels_;
    FILE *bin_ = nullptr;
    FILE *meta_ = nullptr;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> pending_;
    bool stop_ = false;
    std::atomic<int> outstanding_{0};
};

// One V4L2 stream per device, running from camera open to camera close.
// The FT602-based camera stops delivering frames after the stream is stopped
// and restarted (only a replug recovers it), so acquisition sessions attach
// to this stream instead of starting their own. While no session is
// attached, the drain thread dequeues and discards frames so the device
// keeps streaming.
struct UsbStream {
    int fd = -1;
    std::vector<std::pair<void *, size_t>> buffers;
    unsigned int bytes_per_line = 0;
    bool streaming = false;
    // Held by the drain thread for each dequeue. Attach/stop are signalled
    // through the atomics (the drain thread re-locks immediately, so waiting
    // on the mutex alone would starve the other side).
    std::mutex fd_mu;
    std::atomic<bool> attached{false};
    std::atomic<bool> stop{false};
    std::thread drain;
};

static bool usb_stream_open(UsbStream *s, CameraParams *camera_params) {
    const char *serial = camera_params->camera_serial.c_str();
    s->fd = open(camera_params->usb_device.c_str(), O_RDWR | O_NONBLOCK);
    if (s->fd < 0) {
        printf("USB cam %s: cannot open %s: %s\n", serial,
               camera_params->usb_device.c_str(), strerror(errno));
        return false;
    }
    if (!apply_usb_format(s->fd, camera_params, &s->bytes_per_line))
        return false;

    v4l2_requestbuffers req{};
    req.count = USB_V4L2_BUFFERS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(s->fd, VIDIOC_REQBUFS, &req) != 0 || req.count < 2) {
        printf("USB cam %s: VIDIOC_REQBUFS failed: %s\n", serial,
               strerror(errno));
        return false;
    }
    for (unsigned int i = 0; i < req.count; i++) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(s->fd, VIDIOC_QUERYBUF, &buf) != 0)
            return false;
        void *p = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
                       MAP_SHARED, s->fd, buf.m.offset);
        if (p == MAP_FAILED)
            return false;
        s->buffers.push_back({p, buf.length});
        if (xioctl(s->fd, VIDIOC_QBUF, &buf) != 0)
            return false;
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(s->fd, VIDIOC_STREAMON, &type) != 0) {
        printf("USB cam %s: VIDIOC_STREAMON failed: %s\n", serial,
               strerror(errno));
        return false;
    }
    s->streaming = true;
    return true;
}

static void usb_stream_close(UsbStream *s) {
    if (s->fd < 0)
        return;
    if (s->streaming) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(s->fd, VIDIOC_STREAMOFF, &type);
    }
    for (auto &b : s->buffers)
        munmap(b.first, b.second);
    s->buffers.clear();
    v4l2_requestbuffers req{};
    req.count = 0;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(s->fd, VIDIOC_REQBUFS, &req);
    close(s->fd);
    s->fd = -1;
}

static void usb_stream_drain(UsbStream *s) {
    while (!s->stop) {
        if (s->attached) {
            usleep(5000);
            continue;
        }
        std::lock_guard<std::mutex> lk(s->fd_mu);
        if (s->attached || s->stop)
            continue;
        pollfd pfd{s->fd, POLLIN, 0};
        if (poll(&pfd, 1, 50) > 0) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            if (xioctl(s->fd, VIDIOC_DQBUF, &buf) == 0)
                xioctl(s->fd, VIDIOC_QBUF, &buf);
        }
    }
}

static std::mutex streams_mu;
static std::map<std::string, UsbStream *> streams;

static bool usb_camera_start(CameraParams *camera_params) {
    std::lock_guard<std::mutex> lk(streams_mu);
    auto it = streams.find(camera_params->usb_device);
    if (it != streams.end()) {
        // Already streaming (camera reopened without a close); the format
        // cannot change while streaming, so report the running one.
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(it->second->fd, VIDIOC_G_FMT, &fmt) == 0) {
            camera_params->width = fmt.fmt.pix.width;
            camera_params->height = fmt.fmt.pix.height;
        }
        return true;
    }
    UsbStream *s = new UsbStream;
    if (!usb_stream_open(s, camera_params)) {
        usb_stream_close(s);
        delete s;
        return false;
    }
    s->drain = std::thread(usb_stream_drain, s);
    streams[camera_params->usb_device] = s;
    return true;
}

void usb_camera_release(const std::string &device) {
    UsbStream *s;
    {
        std::lock_guard<std::mutex> lk(streams_mu);
        auto it = streams.find(device);
        if (it == streams.end())
            return;
        s = it->second;
        streams.erase(it);
    }
    s->stop = true;
    s->drain.join();
    usb_stream_close(s);
    delete s;
}

static UsbStream *usb_stream_attach(const std::string &device) {
    UsbStream *s;
    {
        std::lock_guard<std::mutex> lk(streams_mu);
        auto it = streams.find(device);
        if (it == streams.end())
            return nullptr;
        s = it->second;
    }
    s->attached = true;
    // Wait out a dequeue the drain thread may have in progress.
    std::lock_guard<std::mutex> lk(s->fd_mu);
    return s;
}

static void usb_stream_detach(UsbStream *s) { s->attached = false; }

// Emergent cameras align their first frame to a PTP gate time. USB cameras
// have no PTP clock, so they start when the local Emergent cameras report
// the gate was reached, and record host timestamps for post-hoc alignment.
static void wait_for_synced_start(CameraParams *camera_params,
                                  CameraControl *camera_control,
                                  PTPParams *ptp_params) {
    bool synced = ptp_params->network_sync || camera_control->sync_camera;
    if (!synced)
        return;
    while (ptp_params->network_sync && !ptp_params->network_set_start_ptp &&
           camera_control->subscribe)
        usleep(100);
    int num_emergent = camera_params->num_cameras - ptp_params->num_usb_cameras;
    if (num_emergent > 0) {
        while (!ptp_params->ptp_start_reached && camera_control->subscribe)
            usleep(100);
    } else if (ptp_params->network_sync) {
        // No local PTP camera: the master used host time for the gate.
        uint64_t deadline = realtime_ns() + 10000000000ull;
        while (realtime_ns() < ptp_params->ptp_global_time &&
               realtime_ns() < deadline && camera_control->subscribe)
            usleep(100);
        ptp_params->ptp_start_reached = true;
    }
}

void acquire_frames_usb(CameraEmergent *ecam, CameraParams *camera_params,
                        CameraEachSelect *camera_select,
                        CameraControl *camera_control,
                        unsigned char *display_buffer,
                        std::string encoder_setup, std::string folder_name,
                        PTPParams *ptp_params,
                        INDIGOSignalBuilder *indigo_signal_builder) {
    (void)ecam;
    cudaSetDevice(camera_params->gpu_id);
    const char *serial = camera_params->camera_serial.c_str();
    CameraState camera_state;
    const bool recording = camera_control->record_video && camera_select->record;

    UsbStream *stream = usb_stream_attach(camera_params->usb_device);
    bool device_ok = stream != nullptr;
    if (!device_ok)
        printf("USB cam %s: device is not streaming (see errors at camera "
               "open); no frames this session\n",
               serial);

    const unsigned int width = camera_params->width;
    const unsigned int height = camera_params->height;
    const size_t frame_pixels = (size_t)width * height;
    const bool is_16bit = camera_params->usb_pixfmt == V4L2_PIX_FMT_Y16;

    // Pinned host memory speeds up the display/encoder host-to-device copies.
    uint8_t *ring8 = nullptr;
    bool ring8_pinned =
        cudaHostAlloc((void **)&ring8, frame_pixels * USB_RING_SIZE,
                      cudaHostAllocDefault) == cudaSuccess;
    if (!ring8_pinned)
        ring8 = (uint8_t *)malloc(frame_pixels * USB_RING_SIZE);
    std::vector<std::vector<uint16_t>> ring16;
    RawFrameWriter *raw_writer = nullptr;
    if (recording && is_16bit && device_ok) {
        ring16.assign(USB_RING_SIZE, std::vector<uint16_t>(frame_pixels));
        raw_writer = new RawFrameWriter(folder_name + "/Cam" +
                                            camera_params->camera_serial,
                                        camera_params, ring16.data());
    }

    int mjpeg_port = 8080 + camera_params->camera_id;
    MjpegServer mjpeg_server(mjpeg_port);
    mjpeg_server.start();

#ifndef HEADLESS
    FrameDetector *frame_detector = nullptr;
    if (camera_select->detect_mode == Detect3D_Standoff ||
        camera_select->detect_mode == Detect2D_Standoff) {
        frame_detector = new FrameDetector(camera_params, camera_select);
        frame_detector->start();
        while (detector_counter.load() !=
               camera_select->total_standoff_detector) {
            usleep(10);
        }
        camera_select->frame_detect_state.store(State_Copy_New_Frame);
    }

    COpenGLDisplay *openGLDisplay = nullptr;
    if (camera_select->stream_on) {
        openGLDisplay =
            new COpenGLDisplay("", camera_params, camera_select, display_buffer,
                               indigo_signal_builder);
        openGLDisplay->StartThread();
    }
#endif

    FrameSaver frame_saver(camera_params, camera_select);
    frame_saver.start();

    GPUVideoEncoder *gpu_encoder = nullptr;
    bool encoder_ready_signal = false;
    if (recording) {
        gpu_encoder = new GPUVideoEncoder("", camera_params, encoder_setup,
                                          folder_name, &encoder_ready_signal);
        gpu_encoder->StartThread();
        while (!encoder_ready_signal) {
            usleep(10);
        }
    }

    wait_for_synced_start(camera_params, camera_control, ptp_params);
    try_start_timer();
    auto t_start = std::chrono::steady_clock::now();

    uint32_t display_peak = 255;
    uint32_t prev_sequence = 0;
    uint64_t v4l2_errors = 0;
    bool stop_seen = false;
    uint64_t stop_seen_ns = 0;

    while (camera_control->subscribe) {
        if (device_ok) {
            pollfd pfd{stream->fd, POLLIN, 0};
            int pr = poll(&pfd, 1, 200);
            if (pr > 0) {
                v4l2_buffer buf{};
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_MMAP;
                if (xioctl(stream->fd, VIDIOC_DQBUF, &buf) != 0) {
                    if (errno != EAGAIN) {
                        printf("USB cam %s: VIDIOC_DQBUF failed: %s\n", serial,
                               strerror(errno));
                        device_ok = false;
                    }
                } else {
                    uint64_t sys_ns = realtime_ns();
                    uint64_t ts_ns =
                        (uint64_t)buf.timestamp.tv_sec * 1000000000ull +
                        (uint64_t)buf.timestamp.tv_usec * 1000ull;
                    size_t expected = (size_t)stream->bytes_per_line * height;
                    if ((buf.flags & V4L2_BUF_FLAG_ERROR) ||
                        buf.bytesused < expected) {
                        v4l2_errors++;
                        camera_state.dropped_frames++;
                        xioctl(stream->fd, VIDIOC_QBUF, &buf);
                    } else {
                        if (camera_state.frame_count != 0 &&
                            buf.sequence != prev_sequence + 1)
                            camera_state.dropped_frames +=
                                buf.sequence - prev_sequence - 1;
                        else
                            camera_state.frames_recd++;
                        prev_sequence = buf.sequence;

                        int slot = camera_state.frame_count % USB_RING_SIZE;
                        uint8_t *frame8 = ring8 + slot * frame_pixels;
                        const uint8_t *src =
                            (const uint8_t *)stream->buffers[buf.index].first;
                        convert_to_mono8(src, stream->bytes_per_line,
                                         camera_params->usb_pixfmt, width,
                                         height, frame8, &display_peak);
                        if (raw_writer) {
                            if (raw_writer->can_accept()) {
                                uint16_t *dst = ring16[slot].data();
                                for (unsigned int y = 0; y < height; y++)
                                    memcpy(dst + y * width,
                                           src + y * stream->bytes_per_line,
                                           width * sizeof(uint16_t));
                                raw_writer->push(slot, camera_state.frame_count,
                                                 buf.sequence, ts_ns, sys_ns);
                            } else if (raw_writer->drops++ % 100 == 0) {
                                printf("WARNING: cam %s raw16 writer behind, "
                                       "%lu frames dropped\n",
                                       serial, raw_writer->drops);
                            }
                        }
                        xioctl(stream->fd, VIDIOC_QBUF, &buf);

                        if (gpu_encoder &&
                            !gpu_encoder->PushToDisplay(
                                frame8, frame_pixels, width, height,
                                GVSP_PIX_MONO8, ts_ns, camera_state.frame_count,
                                sys_ns, 0, 0)) {
                            camera_state.encoder_drops++;
                            if (camera_state.encoder_drops % 100 == 1)
                                printf("WARNING: cam %s encoder queue full "
                                       "(total encoder drops: %u)\n",
                                       serial, camera_state.encoder_drops);
                        }
#ifndef HEADLESS
                        if (openGLDisplay)
                            openGLDisplay->PushToDisplay(
                                frame8, frame_pixels, width, height,
                                GVSP_PIX_MONO8, ts_ns,
                                camera_state.frame_count);
                        if (frame_detector &&
                            camera_select->frame_detect_state.load() ==
                                State_Copy_New_Frame)
                            frame_detector->notify_frame_ready(frame8, 0);
#endif
                        if (camera_select->frame_save_state.load() ==
                            State_Copy_New_Frame)
                            frame_saver.notify_frame_ready(frame8);

                        if (camera_state.frame_count % 15 == 0 && !recording) {
                            cv::Mat img(height, width, CV_8UC1, frame8);
                            std::vector<uint8_t> jpeg;
                            cv::imencode(".jpg", img, jpeg,
                                         {cv::IMWRITE_JPEG_QUALITY, 75});
                            mjpeg_server.push(jpeg);
                        }
                        camera_state.frame_count++;
                    }
                }
            } else if (pr < 0 && errno != EINTR) {
                printf("USB cam %s: poll failed: %s\n", serial,
                       strerror(errno));
                device_ok = false;
            }
        } else {
            usleep(10000);
        }

        // Networked stop: mirror the Emergent stop barrier so their threads
        // are not left waiting on this camera. Keep recording until the local
        // Emergent cameras reach the stop time (or 3 s, as in acquire_frames).
        if (ptp_params->network_sync && ptp_params->network_set_stop_ptp) {
            if (!stop_seen) {
                stop_seen = true;
                stop_seen_ns = realtime_ns();
            }
            int num_emergent =
                camera_params->num_cameras - ptp_params->num_usb_cameras;
            bool emergent_done =
                num_emergent > 0 &&
                ptp_params->ptp_stop_counter >= (uint64_t)num_emergent;
            bool by_timeout = realtime_ns() - stop_seen_ns > 3000000000ull;
            if (emergent_done || by_timeout) {
                sync_fetch_and_add(&ptp_params->ptp_stop_counter, 1);
                while (ptp_params->ptp_stop_counter !=
                       camera_params->num_cameras) {
                    usleep(10);
                }
                ptp_params->ptp_stop_reached = true;
                camera_control->subscribe = false;
                break;
            }
        }
    }

    try_stop_timer();
    double time_diff = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t_start)
                           .count();
    if (stream)
        usb_stream_detach(stream);

#ifndef HEADLESS
    if (openGLDisplay) {
        openGLDisplay->StopThread();
        delete openGLDisplay;
    }
    if (frame_detector) {
        frame_detector->stop();
        delete frame_detector;
    }
#endif
    frame_saver.stop();
    if (gpu_encoder) {
        gpu_encoder->StopThread();
        delete gpu_encoder;
    }
    if (raw_writer) {
        printf("USB cam %s: raw16 writer drops: %lu\n", serial,
               raw_writer->drops);
        delete raw_writer;
    }
    mjpeg_server.stop();
    if (v4l2_errors)
        printf("USB cam %s: %lu corrupt/short V4L2 frames\n", serial,
               v4l2_errors);
    report_statistics(camera_params, &camera_state, time_diff);

    if (ring8_pinned)
        cudaFreeHost(ring8);
    else
        free(ring8);
}
