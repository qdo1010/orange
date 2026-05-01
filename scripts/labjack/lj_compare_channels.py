#!/usr/bin/env python3
"""Stack AIN2 (frame clock) and AIN0 (LED witness) with a shared time axis.

Usage:
  python3 /tmp/lj_compare_channels.py                # latest, 200 ms window
  python3 /tmp/lj_compare_channels.py 300            # 300 ms window
  python3 /tmp/lj_compare_channels.py 200 4500       # window starting 4500 ms in
  python3 /tmp/lj_compare_channels.py /path/to/bin   # explicit file (200 ms default)
"""
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lj_load import load_lj


def find_latest():
    s = sorted(glob.glob("/home/ratan/orange_data/exp/unsorted/202*"))
    if not s:
        sys.exit("no sessions found")
    return os.path.join(s[-1], "labjack_analog.bin")


# Parse args
path = None
win_ms = 200
start_ms = None
for a in sys.argv[1:]:
    if a.endswith(".bin") or "/" in a:
        path = a
    elif start_ms is None and win_ms == 200 and a.isdigit() and int(a) > 100:
        # First numeric arg is window length
        win_ms = int(a)
    elif start_ms is None:
        try:
            start_ms = int(a)
        except ValueError:
            pass
if path is None:
    path = find_latest()
print("loading:", path)

start_ns, rate, ch = load_lj(path)
ain2 = ch["AIN2"]
ain0 = ch.get("AIN0")
if ain0 is None:
    sys.exit("file has no AIN0 channel — old format?")
n = len(ain2)
total_ms = n / rate * 1000

# Auto-pick window start: first AIN0 rising edge - 5 ms (so we land where LED activity begins)
if start_ms is None:
    above0 = ain0 > 2.5
    rising0 = np.where(np.diff(above0.astype(np.int8)) > 0)[0]
    if len(rising0):
        start_ms = max(0, rising0[0] / rate * 1000 - 5)
    else:
        start_ms = 0

t0 = float(start_ms)
t1 = min(t0 + win_ms, total_ms)
print(f"  total {total_ms:.1f} ms,  window {t0:.1f} -> {t1:.1f} ms ({t1-t0:.1f} ms wide)")

t_axis = np.arange(n) / rate * 1000
m = (t_axis >= t0) & (t_axis <= t1)

# Edges in window
above2 = ain2 > 2.5
falling2 = np.where(np.diff(above2.astype(np.int8)) < 0)[0]
falling2_ms = falling2 / rate * 1000
fl_in = [t for t in falling2_ms if t0 <= t <= t1]

above0 = ain0 > 2.5
rising0 = np.where(np.diff(above0.astype(np.int8)) > 0)[0]
rising0_ms = rising0 / rate * 1000
ri_in = [t for t in rising0_ms if t0 <= t <= t1]

# For each AIN2 falling edge in window, find next AIN0 rising
pairs = []
for fe in fl_in:
    nx = [r for r in ri_in if r > fe and r - fe < 25]
    if nx:
        pairs.append((fe, nx[0], nx[0] - fe))

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle(f"{os.path.basename(os.path.dirname(path))}  /  {os.path.basename(path)}\n"
             f"window {t0:.0f}–{t1:.0f} ms   "
             f"({len(fl_in)} AIN2 edges, {len(ri_in)} AIN0 pulses)",
             fontsize=11)

# Top: AIN2 with falling-edge markers
axes[0].plot(t_axis[m], ain2[m], lw=0.7, color="C0", label="AIN2 frame clock")
axes[0].axhline(2.5, color="orange", lw=0.5, ls=":")
for fe in fl_in:
    axes[0].axvline(fe, color="red", lw=0.5, alpha=0.6, ls="--")
axes[0].set_ylim(-0.5, 6)
axes[0].set_ylabel("AIN2 (V)\n(frame clock)")
axes[0].grid(True, alpha=0.3)
axes[0].set_title(f"AIN2 — ScanImage frame clock  "
                  f"(red dash = falling edge: end of scan / start of flyback)")

# Bottom: AIN0 with rising-edge markers + connecting lines to corresponding AIN2 edge
axes[1].plot(t_axis[m], ain0[m], lw=0.7, color="C2", label="AIN0 LED witness")
axes[1].axhline(2.5, color="orange", lw=0.5, ls=":")
for fe in fl_in:
    axes[1].axvline(fe, color="red", lw=0.4, alpha=0.4, ls="--")
for ri in ri_in:
    axes[1].axvline(ri, color="purple", lw=0.5, alpha=0.6)
# Annotate latency arrows for each pair
for fe, ri, lag in pairs:
    axes[1].annotate("", xy=(ri, 5.6), xytext=(fe, 5.6),
                     arrowprops=dict(arrowstyle="<->", color="black", lw=0.7))
    axes[1].text((fe + ri) / 2, 5.8, f"{lag:.1f}ms",
                 ha="center", fontsize=7)
axes[1].set_ylim(-0.5, 6.5)
axes[1].set_ylabel("AIN0 (V)\n(LED witness)")
axes[1].set_xlabel("time since recording start (ms)")
axes[1].grid(True, alpha=0.3)
axes[1].set_title(f"AIN0 — daisy-chain TRIG_OUT  "
                  f"(purple = LED-on rising;  arrow = edge → LED latency)")

plt.tight_layout()
out = "/tmp/lj_compare_channels.png"
plt.savefig(out, dpi=120)
print("saved:", out)
print(f"  {len(pairs)} edge→LED pairs, mean latency "
      f"{np.mean([p[2] for p in pairs]):.2f} ms" if pairs else "  no pairs in window")
plt.show()
