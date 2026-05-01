#!/usr/bin/env python3
"""Plot edge -> LED latency over time. Tests the 'two clocks drifting'
hypothesis: if the latency drifts monotonically, the scanner and camera
internal clocks are running at different rates."""
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lj_load import load_lj


path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    sorted(glob.glob("/home/ratan/orange_data/exp/unsorted/202*"))[-1],
    "labjack_analog.bin")
print("loading:", path)

start_ns, rate, ch = load_lj(path)
ain2 = ch["AIN2"]
ain0 = ch["AIN0"]
t_axis_ms = np.arange(len(ain2)) / rate * 1000

# Detect edges
fl2 = np.where(np.diff((ain2 > 2.5).astype(np.int8)) < 0)[0]
ri0 = np.where(np.diff((ain0 > 2.5).astype(np.int8)) > 0)[0]
fl2_ms = fl2 / rate * 1000
ri0_ms = ri0 / rate * 1000

# For each AIN2 falling edge, find the next AIN0 rising within one cycle
pairs = []
j = 0
for fe in fl2_ms:
    while j < len(ri0_ms) and ri0_ms[j] <= fe:
        j += 1
    if j < len(ri0_ms) and ri0_ms[j] - fe < 30:
        pairs.append((fe, ri0_ms[j], ri0_ms[j] - fe))

if not pairs:
    sys.exit("no edge -> LED pairs found")

edges = np.array([p[0] for p in pairs])
leds  = np.array([p[1] for p in pairs])
lats  = np.array([p[2] for p in pairs])

# Fit a linear trend: lat = m*t + b
slope, intercept = np.polyfit(edges, lats, 1)  # slope in ms / ms_of_recording
slope_us_per_s = slope * 1000
print(f"  N pairs: {len(pairs)}")
print(f"  latency: mean {lats.mean():.3f} ms  std {lats.std():.3f}  range {lats.min():.3f}–{lats.max():.3f}")
print(f"  linear drift fit: slope = {slope_us_per_s:.2f} µs / second of recording")
print(f"                    intercept = {intercept:.3f} ms")
# Compare period of AIN2 vs AIN0
p2 = np.diff(fl2_ms)
p0 = np.diff(ri0_ms)
print(f"  AIN2 period  mean {p2.mean():.5f} ms  std {p2.std():.5f}  -> {1000/p2.mean():.4f} Hz")
print(f"  AIN0 period  mean {p0.mean():.5f} ms  std {p0.std():.5f}  -> {1000/p0.mean():.4f} Hz")
print(f"  rate mismatch: {(1000/p0.mean()) - (1000/p2.mean()):+.4f} Hz "
      f"= {(p2.mean() - p0.mean())*1000:+.2f} µs/cycle")

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig.suptitle(os.path.basename(os.path.dirname(path)) + "  /  edge→LED latency over time")

axes[0].scatter(edges/1000, lats, s=8, alpha=0.6, label="measured")
axes[0].plot(edges/1000, slope*edges + intercept, color="red", lw=1,
             label=f"linear fit ({slope_us_per_s:+.1f} µs/s drift)")
axes[0].set_ylabel("edge → next LED rising  (ms)")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best", fontsize=9)
axes[0].set_title("If this slopes monotonically, the scanner and camera clocks aren't locked")

# Cumulative cycle-to-cycle drift
edge_periods = p2  # ms
led_periods = p0   # ms
n = min(len(edge_periods), len(led_periods))
cum_drift_ms = np.cumsum(led_periods[:n] - edge_periods[:n])
axes[1].plot(np.arange(n) / 40, cum_drift_ms, color="C2")
axes[1].set_ylabel("cumulative drift\n(LED – scanner)  (ms)")
axes[1].set_xlabel("time (s)")
axes[1].grid(True, alpha=0.3)
axes[1].set_title("Cumulative drift = where the LED 'wraps' through the scan window")
axes[1].axhline(2.16, color="orange", lw=0.8, ls=":", label="flyback width (2.16 ms)")
axes[1].axhline(0, color="gray", lw=0.5)
axes[1].legend(loc="best", fontsize=9)

plt.tight_layout()
out = "/tmp/lj_latency_drift.png"
plt.savefig(out, dpi=120)
print("saved:", out)
plt.show()
