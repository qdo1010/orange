#ifndef ORANGE_LABJACK_TRIGGER
#define ORANGE_LABJACK_TRIGGER

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

// Reads a TTL clock signal (e.g. ScanImage frame clock) on the LabJack T7's
// AIN2 in stream mode and exposes a thread-safe "wait for next falling edge"
// primitive. Multiple camera threads can wait concurrently and will all unblock
// on the same edge.
class LabJackTrigger {
  public:
    LabJackTrigger();
    ~LabJackTrigger();

    // Open T7, configure AIN2, start streaming + reader thread.
    bool start(double scan_rate_hz = 10000.0, double threshold_v = 2.5);

    // Stop reader thread, close T7. Wakes any waiters.
    void stop();

    // Block until edge_counter > last_seen, or until timeout/stop.
    // Returns the new edge counter value, or 0 on timeout/stop.
    // If ts_out is non-null, also returns the wall-clock timestamp of the
    // returned edge (read under the same lock, so index and ts are
    // consistent).
    uint64_t wait_for_next_edge(uint64_t last_seen, int timeout_ms = 1000,
                                uint64_t *ts_out = nullptr);

    // CLOCK_REALTIME ns at the moment the most recent falling edge was seen.
    uint64_t last_edge_timestamp_ns();

    // Total falling edges seen since start().
    uint64_t edge_counter();

    // True if start() succeeded and stop() hasn't been called yet.
    bool running() const { return running_.load(); }

    // Master-side: pop one pending edge for ENet broadcast. Returns false
    // when the queue is empty.
    bool dequeue_pending_edge(uint64_t *idx, uint64_t *ts_ns);

    // Inject an edge from an external source (used by clients receiving
    // LJEDGE messages over ENet, and by the HEADLESS stub which has no
    // local T7). Updates the edge counter, records the timestamp, wakes
    // any threads waiting in wait_for_next_edge. Does NOT enqueue to
    // pending_ — clients don't re-broadcast.
    void inject_edge(uint64_t idx, uint64_t ts_ns);

    // Master only: begin writing every analog sample to a binary file.
    // Format (24-byte header):
    //   uint64_t start_clock_realtime_ns
    //   double   scan_rate_hz
    //   uint32_t num_channels
    //   uint32_t reserved
    // followed by interleaved float64 samples (num_channels per scan, in
    // scan_list order: currently AIN2, then AIN0).
    // No-op when not streaming, or in HEADLESS clients.
    void start_logging(const std::string &path);
    void stop_logging();

  private:
    int handle_ = -1;
    std::thread reader_thread_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> running_{false};

    std::mutex mu_;
    std::condition_variable cv_;
    uint64_t edge_counter_ = 0;
    uint64_t edge_timestamp_ns_ = 0;
    std::deque<std::pair<uint64_t, uint64_t>> pending_; // (idx, ts_ns)

    // AIN2 raw-sample logging (master only)
    std::mutex log_mu_;
    std::ofstream log_file_;
    std::string log_path_;
    bool log_active_ = false;
    double scan_rate_hz_ = 0.0;

    void reader_loop_(double scan_rate_hz, double threshold_v);
};

#endif
