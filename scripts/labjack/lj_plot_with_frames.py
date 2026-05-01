#!/usr/bin/env python3
"""Plot AIN2 trace with camera frames and LJ edges overlaid.

Shows the actual recording window (skips the 3s PTP gate at the start
where no frames exist). All three sources of timing info are in master's
CLOCK_REALTIME, so they share a clock and can be plotted on the same axis:

  - AIN2 samples (the analog stream, blue line)
  - lj_edge_timestamp_ns from meta CSV (red dashed) — when master's
    LabJack reader detected each falling edge
  - timestamp_sys from meta CSV (green) — when each camera frame arrived
    at the host (i.e. after the camera exposed + read out + transferred)

The IR LED fires ~1-2 ms AFTER the red line (camera-internal trigger
latency) and stays on for the exposure duration. The frame's
timestamp_sys (green) is later again, after readout+transfer.

Usage:
  python3 /tmp/lj_plot_with_frames.py                # window starting at first frame
  python3 /tmp/lj_plot_with_frames.py <window_ms>    # different window length
"""
import os, struct, glob, sys, csv
import numpy as np
import matplotlib.pyplot as plt


WIN_MS = int(sys.argv[1]) if len(sys.argv) > 1 else 200

session_dir = sorted(glob.glob("/home/ratan/orange_data/exp/unsorted/202*"))[-1]
print("session:", session_dir)

with open(os.path.join(session_dir, "labjack_analog.bin"), "rb") as f:
    start_ns, rate = struct.unpack("Qd", f.read(16))
    samples = np.frombuffer(f.read(), dtype=np.float64).copy()

# Find the master cam's meta CSV (one whose timestamp_sys is close to start_ns)
meta_files = sorted(glob.glob(os.path.join(session_dir, "Cam*_meta.csv")))
master_csv = None
for p in meta_files:
    with open(p) as f:
        first = next(csv.DictReader(f))
        # master cam's timestamp_sys is in the same clock as start_ns;
        # frame 0 is ~3 s after start_ns due to PTP gate. Allow up to 60 s.
        if abs(int(first["timestamp_sys"]) - start_ns) < int(60e9):
            master_csv = p
            break
if master_csv is None:
    sys.exit("no master cam meta CSV found")
print("master cam:", os.path.basename(master_csv))

fids, ts_sys, lj_idx, lj_ts = [], [], [], []
with open(master_csv) as f:
    for row in csv.DictReader(f):
        fids.append(int(row["frame_id"]))
        ts_sys.append(int(row["timestamp_sys"]))
        lj_idx.append(int(row["lj_edge_index"]))
        lj_ts.append(int(row["lj_edge_timestamp_ns"]))

# Convert to ms relative to analog file start
frame_t = [(t - start_ns) / 1e6 for t in ts_sys]
edge_t = [(t - start_ns) / 1e6 for t in lj_ts]

# Window: 5 ms before first frame, then WIN_MS more
t0 = frame_t[0] - 5
t1 = t0 + WIN_MS
t_axis_ms = np.arange(len(samples)) / rate * 1000

mask = (t_axis_ms >= t0) & (t_axis_ms <= t1)
t_view = t_axis_ms[mask]
s_view = samples[mask]

frames_in_view = [(fi, ft) for fi, ft in zip(fids, frame_t) if t0 <= ft <= t1]
edges_in_view = [(li, et) for li, et in zip(lj_idx, edge_t)
                 if t0 <= et <= t1 and li > 0]

print(f"window: {t0:.1f} → {t1:.1f} ms")
print(f"  {len(frames_in_view)} camera-frame markers, "
      f"{len(edges_in_view)} LJ-edge markers")
if frames_in_view and edges_in_view:
    f0_t = frames_in_view[0][1]
    e_for_f0 = next((et for li, et in edges_in_view
                     if li == lj_idx[fids.index(frames_in_view[0][0])]), None)
    if e_for_f0 is not None:
        print(f"  frame 0 timestamp_sys = {f0_t:.3f} ms,  "
              f"its LJ edge_ts = {e_for_f0:.3f} ms,  "
              f"diff = {f0_t - e_for_f0:.3f} ms (camera processing latency)")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t_view, s_view, lw=0.7, color="C0", label="AIN2 (V)")
ax.axhline(2.5, color="orange", lw=0.5, ls=":", label="2.5 V threshold")

# LJ falling edges (red dashed) — when master detected the edge
# Camera fires TriggerSoftware ~1 ms after this red line.
for li, et in edges_in_view:
    ax.axvline(et, color="red", lw=0.6, alpha=0.5, ls="--")
    ax.text(et, -0.7, f"e{li}", ha="center", fontsize=7, color="red")

# Estimated IR LED ON window (orange shaded) — ~1 ms after edge, lasts
# the exposure duration. Shown as a 1.5 ms-wide block at this scale just
# for visibility (real exposure is 25 µs).
EST_LATENCY_MS = 1.5     # rough estimate: cv wakeup + GigE TriggerSoftware
EST_LED_WIDTH_MS = 1.5   # exaggerated for visibility; actual = exposure_us/1000
for li, et in edges_in_view:
    led_t = et + EST_LATENCY_MS
    ax.axvspan(led_t, led_t + EST_LED_WIDTH_MS, color="orange",
               alpha=0.35)
ax.axvspan(0, 0, color="orange", alpha=0.35,
           label="est. IR LED on (edge + ~1 ms)")

# Camera frame timestamp_sys (green) — when frame arrived at master
# AFTER readout+transfer (NOT when IR fired)
for fi, ft in frames_in_view:
    ax.axvline(ft, color="green", lw=0.7, alpha=0.7)
    ax.text(ft, 5.5, f"f{fi}", ha="center", fontsize=7, color="darkgreen",
            fontweight="bold")

ax.set_xlim(t0, t1)
ax.set_ylim(-1, 6)
ax.set_xlabel("time since analog-log start (ms)  —  master CLOCK_REALTIME")
ax.set_ylabel("AIN2 (V)")
ax.grid(True, alpha=0.3)
ax.set_title(
    f"{os.path.basename(session_dir)}  /  {os.path.basename(master_csv)}\n"
    f"window {t0:.0f}→{t1:.0f} ms\n"
    f"red dash = LJ edge (start of flyback)   "
    f"orange = estimated IR LED on (edge + ~1 ms)   "
    f"green = frame ARRIVED at host (after readout+transfer, NOT LED on)"
)
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
out_png = "/tmp/lj_plot_with_frames.png"
plt.savefig(out_png, dpi=120)
print("saved:", out_png)
plt.show()
