#!/usr/bin/env python3
"""Real edge-to-exposure latency.

We have three clocks per frame:
  - timestamp                  : camera-internal PTP (exposure start)
  - timestamp_sys              : host CLOCK_REALTIME at frame arrival
  - lj_edge_timestamp_ns       : host CLOCK_REALTIME when LJ saw the edge

Camera PTP and host CLOCK_REALTIME are NOT in the same epoch — there's a
constant offset K (camera_PTP - host_REALTIME). We can't measure K directly
without an external anchor, but we don't need to: the JITTER in
(timestamp - lj_edge_timestamp_ns) is independent of K, and that's the
number that actually matters for "can we land the LED in flyback?".
"""
import csv, glob, os, sys, statistics

session = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("/home/ratan/orange_data/exp/unsorted/202*"))[-1]
print("session:", session)

for csv_path in sorted(glob.glob(os.path.join(session, "Cam*_meta.csv"))):
    cam = os.path.basename(csv_path).replace("_meta.csv", "")
    rows = list(csv.DictReader(open(csv_path)))
    if len(rows) < 50:
        continue

    ts_cam = [int(r["timestamp"]) for r in rows]
    ts_sys = [int(r["timestamp_sys"]) for r in rows]
    lj_ts  = [int(r["lj_edge_timestamp_ns"]) for r in rows]
    lj_idx = [int(r["lj_edge_index"]) for r in rows]

    cam_period_ms  = [(ts_cam[i] - ts_cam[i-1])/1e6 for i in range(1, len(ts_cam))]
    lj_period_ms   = [(lj_ts[i]  - lj_ts[i-1])/1e6  for i in range(1, len(lj_ts))]

    # apparent offset = real_latency + K  (K constant per cam, drops out of jitter)
    apparent_off_ms = [(ts_cam[i] - lj_ts[i])/1e6 for i in range(len(rows))]
    host_arrive_ms  = [(ts_sys[i] - lj_ts[i])/1e6 for i in range(len(rows))]

    # 1:1 edge-to-frame check
    skips = sum(1 for i in range(1, len(lj_idx)) if lj_idx[i] - lj_idx[i-1] != 1)

    print(f"\n=== {cam}  ({len(rows)} frames, {skips} edge-skips) ===")
    print(f"  cam-PTP frame period:   mean {statistics.mean(cam_period_ms):.4f} ms  "
          f"std {statistics.stdev(cam_period_ms):.4f}  "
          f"min {min(cam_period_ms):.3f}  max {max(cam_period_ms):.3f}")
    print(f"  LJ-edge period:         mean {statistics.mean(lj_period_ms):.4f} ms  "
          f"std {statistics.stdev(lj_period_ms):.4f}  "
          f"min {min(lj_period_ms):.3f}  max {max(lj_period_ms):.3f}")
    print(f"  apparent (cam_ts - lj_ts) = real_latency + K_const:")
    print(f"    mean {statistics.mean(apparent_off_ms):.4f} ms  "
          f"std {statistics.stdev(apparent_off_ms):.4f}  "
          f"min {min(apparent_off_ms):.4f}  max {max(apparent_off_ms):.4f}  "
          f"→ jitter (max-min) {max(apparent_off_ms)-min(apparent_off_ms):.3f} ms")
    print(f"  host-arrival latency (ts_sys - lj_ts):")
    print(f"    mean {statistics.mean(host_arrive_ms):.4f} ms  "
          f"std {statistics.stdev(host_arrive_ms):.4f}  "
          f"min {min(host_arrive_ms):.3f}  max {max(host_arrive_ms):.3f}")

    # readout + transfer + queue ≈ host_arrive - real_latency
    # = (ts_sys - lj_ts) - ((ts_cam - lj_ts) + K_const offset to host)
    # = ts_sys - ts_cam - K
    # we don't know K, but ts_sys - ts_cam should be consistent per cam
    sys_minus_cam_s = [(ts_sys[i] - ts_cam[i])/1e9 for i in range(len(rows))]
    print(f"  ts_sys - timestamp (= readout+transfer+queue + K, in seconds):")
    print(f"    mean {statistics.mean(sys_minus_cam_s):.6f} s  "
          f"std*1000 {statistics.stdev(sys_minus_cam_s)*1000:.3f} ms  "
          f"→ jitter {(max(sys_minus_cam_s)-min(sys_minus_cam_s))*1000:.3f} ms")
