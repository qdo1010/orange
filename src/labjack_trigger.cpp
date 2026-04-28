#include "labjack_trigger.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <vector>

#ifdef HEADLESS

// orange_client doesn't open a local T7. Trigger events arrive from the
// master over ENet and feed in via inject_edge(). The waiting/notifying
// machinery (mutex + cv + edge_counter_) is identical in both modes.

LabJackTrigger::LabJackTrigger() = default;
LabJackTrigger::~LabJackTrigger() { stop(); }
bool LabJackTrigger::start(double, double) {
    if (running_.load()) return true;
    stop_flag_.store(false); // allow restart after a previous stop()
    running_.store(true);
    return true;
}
void LabJackTrigger::stop() {
    if (!running_.load()) return;
    stop_flag_.store(true);
    cv_.notify_all();
    running_.store(false);
}
void LabJackTrigger::reader_loop_(double, double) {}

uint64_t LabJackTrigger::wait_for_next_edge(uint64_t last_seen, int timeout_ms,
                                            uint64_t *ts_out) {
    std::unique_lock<std::mutex> lock(mu_);
    bool ok = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
        return stop_flag_.load() || edge_counter_ > last_seen;
    });
    if (stop_flag_.load() || !ok) {
        if (ts_out) *ts_out = 0;
        return 0;
    }
    if (ts_out) *ts_out = edge_timestamp_ns_;
    return edge_counter_;
}

uint64_t LabJackTrigger::last_edge_timestamp_ns() {
    std::lock_guard<std::mutex> lock(mu_);
    return edge_timestamp_ns_;
}

uint64_t LabJackTrigger::edge_counter() {
    std::lock_guard<std::mutex> lock(mu_);
    return edge_counter_;
}

bool LabJackTrigger::dequeue_pending_edge(uint64_t *, uint64_t *) {
    return false; // clients don't broadcast
}

void LabJackTrigger::inject_edge(uint64_t idx, uint64_t ts_ns) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (idx > edge_counter_) edge_counter_ = idx;
        edge_timestamp_ns_ = ts_ns;
    }
    cv_.notify_all();
}

#else

#include <LabJackM.h>

namespace {
// Small chunks → low edge-detection latency. At SCAN_RATE=10000 Hz this is
// ~1 ms per LJM_eStreamRead, which keeps detection latency well under the
// ~25 ms ScanImage frame period.
constexpr int SCANS_PER_READ = 10;
constexpr int NUM_CHANNELS = 1;
constexpr size_t PENDING_QUEUE_CAP = 256;

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

void log_ljm_err(int err, const char *where) {
    char msg[LJM_MAX_NAME_SIZE];
    LJM_ErrorToString(err, msg);
    fprintf(stderr, "[LabJackTrigger] %s failed: %d (%s)\n", where, err, msg);
}
} // namespace

LabJackTrigger::LabJackTrigger() = default;

LabJackTrigger::~LabJackTrigger() { stop(); }

bool LabJackTrigger::start(double scan_rate_hz, double threshold_v) {
    if (running_.load()) return true;

    int err = LJM_Open(LJM_dtT7, LJM_ctANY, "ANY", &handle_);
    if (err) {
        log_ljm_err(err, "LJM_Open");
        return false;
    }

    const char *cfg_names[] = {"AIN2_RANGE", "AIN2_RESOLUTION_INDEX",
                               "STREAM_SETTLING_US",
                               "STREAM_RESOLUTION_INDEX"};
    double cfg_vals[] = {10.0, 0.0, 0.0, 0.0};
    int errAddr = 0;
    err = LJM_eWriteNames(handle_, 4, cfg_names, cfg_vals, &errAddr);
    if (err) {
        log_ljm_err(err, "LJM_eWriteNames");
        LJM_Close(handle_);
        handle_ = -1;
        return false;
    }

    int scan_list[NUM_CHANNELS];
    err = LJM_NameToAddress("AIN2", &scan_list[0], NULL);
    if (err) {
        log_ljm_err(err, "LJM_NameToAddress");
        LJM_Close(handle_);
        handle_ = -1;
        return false;
    }

    double rate = scan_rate_hz;
    err = LJM_eStreamStart(handle_, SCANS_PER_READ, NUM_CHANNELS, scan_list,
                           &rate);
    if (err) {
        log_ljm_err(err, "LJM_eStreamStart");
        LJM_Close(handle_);
        handle_ = -1;
        return false;
    }

    fprintf(stderr,
            "[LabJackTrigger] streaming AIN2 at %.2f Hz "
            "(TTL threshold %.2fV)\n",
            rate, threshold_v);

    stop_flag_.store(false);
    running_.store(true);
    reader_thread_ =
        std::thread(&LabJackTrigger::reader_loop_, this, rate, threshold_v);
    return true;
}

void LabJackTrigger::stop() {
    if (!running_.load()) return;
    stop_flag_.store(true);
    cv_.notify_all();
    if (reader_thread_.joinable()) reader_thread_.join();
    if (handle_ >= 0) {
        LJM_eStreamStop(handle_);
        LJM_Close(handle_);
        handle_ = -1;
    }
    running_.store(false);
    fprintf(stderr, "[LabJackTrigger] stopped (%lu edges seen)\n",
            (unsigned long)edge_counter_);
}

uint64_t LabJackTrigger::wait_for_next_edge(uint64_t last_seen, int timeout_ms,
                                            uint64_t *ts_out) {
    std::unique_lock<std::mutex> lock(mu_);
    bool ok = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
        return stop_flag_.load() || edge_counter_ > last_seen;
    });
    if (stop_flag_.load() || !ok) {
        if (ts_out) *ts_out = 0;
        return 0;
    }
    if (ts_out) *ts_out = edge_timestamp_ns_;
    return edge_counter_;
}

uint64_t LabJackTrigger::last_edge_timestamp_ns() {
    std::lock_guard<std::mutex> lock(mu_);
    return edge_timestamp_ns_;
}

uint64_t LabJackTrigger::edge_counter() {
    std::lock_guard<std::mutex> lock(mu_);
    return edge_counter_;
}

bool LabJackTrigger::dequeue_pending_edge(uint64_t *idx, uint64_t *ts_ns) {
    std::lock_guard<std::mutex> lock(mu_);
    if (pending_.empty()) return false;
    auto front = pending_.front();
    pending_.pop_front();
    if (idx) *idx = front.first;
    if (ts_ns) *ts_ns = front.second;
    return true;
}

void LabJackTrigger::inject_edge(uint64_t idx, uint64_t ts_ns) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (idx > edge_counter_) edge_counter_ = idx;
        edge_timestamp_ns_ = ts_ns;
    }
    cv_.notify_all();
}

void LabJackTrigger::reader_loop_(double scan_rate_hz, double threshold_v) {
    (void)scan_rate_hz;
    int buf_len = SCANS_PER_READ * NUM_CHANNELS;
    std::vector<double> buf(buf_len);
    int prev_above = -1;
    int dev_backlog = 0, ljm_backlog = 0;

    while (!stop_flag_.load()) {
        int err =
            LJM_eStreamRead(handle_, buf.data(), &dev_backlog, &ljm_backlog);
        if (err) {
            log_ljm_err(err, "LJM_eStreamRead");
            break;
        }
        for (int i = 0; i < buf_len; i++) {
            int above = buf[i] > threshold_v ? 1 : 0;
            if (prev_above == 1 && above == 0) {
                uint64_t ts = now_ns();
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    edge_counter_++;
                    edge_timestamp_ns_ = ts;
                    if (pending_.size() < PENDING_QUEUE_CAP) {
                        pending_.emplace_back(edge_counter_, ts);
                    }
                }
                cv_.notify_all();
            }
            if (above != prev_above) prev_above = above;
        }
    }
}

#endif // HEADLESS
