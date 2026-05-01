#!/usr/bin/env python3
"""Plot a labjack_analog.bin with both AIN2 (frame clock) and AIN0 (LED witness).

Top panel: full trace, both channels stacked
Bottom panel: zoomed first 100 ms of both channels

Usage:
  python3 /tmp/lj_replot2.py                  # latest unsorted session
  python3 /tmp/lj_replot2.py <path_to_bin>    # explicit
"""
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lj_load import load_lj


def find_latest():
    sessions = sorted(glob.glob("/home/ratan/orange_data/exp/unsorted/202*"))
    if not sessions:
        sys.exit("no sessions found")
    p = os.path.join(sessions[-1], "labjack_analog.bin")
    if not os.path.exists(p):
        sys.exit(f"no labjack_analog.bin in {sessions[-1]}")
    return p


path = sys.argv[1] if len(sys.argv) > 1 else find_latest()
print("loading:", path)
start_ns, rate, ch = load_lj(path)
ain2 = ch["AIN2"]
ain0 = ch.get("AIN0")
n = len(ain2)
t = np.arange(n) / rate
print(f"  {n} samples @ {rate:.1f} Hz = {t[-1]:.2f} s")
print(f"  AIN2: min {ain2.min():+.3f} max {ain2.max():+.3f} mean {ain2.mean():+.3f}")
if ain0 is not None:
    print(f"  AIN0: min {ain0.min():+.3f} max {ain0.max():+.3f} mean {ain0.mean():+.3f}")
else:
    print("  AIN0: not present (old single-channel file)")

# Edge stats on AIN2
above = ain2 > 2.5
falling = np.where(np.diff(above.astype(np.int8)) < 0)[0]
print(f"  AIN2 falling edges: {len(falling)}")

n_panels = 4 if ain0 is not None else 2
fig, axes = plt.subplots(n_panels, 1, figsize=(13, 2.5 * n_panels))
fig.suptitle(os.path.basename(os.path.dirname(path)) + "  /  " +
             os.path.basename(path))

# Full traces
axes[0].plot(t, ain2, lw=0.5, color="C0")
axes[0].axhline(2.5, color="orange", lw=0.5, ls=":")
axes[0].set_ylabel("AIN2 (V)\nframe clock")
axes[0].set_ylim(-1, 6)
axes[0].grid(True, alpha=0.3)
axes[0].set_title(f"full trace ({t[-1]:.1f}s, {len(falling)} edges)")

if ain0 is not None:
    axes[1].plot(t, ain0, lw=0.5, color="C2")
    axes[1].set_ylabel("AIN0 (V)\nLED witness")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("full trace")

# Zoom first 100 ms
win = int(0.1 * rate)
axes[-2 if ain0 is not None else -1].plot(t[:win], ain2[:win], lw=0.8, color="C0")
axes[-2 if ain0 is not None else -1].axhline(2.5, color="orange", lw=0.5, ls=":")
axes[-2 if ain0 is not None else -1].set_ylabel("AIN2 (V)")
axes[-2 if ain0 is not None else -1].set_ylim(-1, 6)
axes[-2 if ain0 is not None else -1].grid(True, alpha=0.3)
axes[-2 if ain0 is not None else -1].set_title("first 100 ms")

if ain0 is not None:
    axes[-1].plot(t[:win], ain0[:win], lw=0.8, color="C2")
    axes[-1].set_ylabel("AIN0 (V)")
    axes[-1].grid(True, alpha=0.3)
    axes[-1].set_title("first 100 ms")
    axes[-1].set_xlabel("time since recording start (s)")
else:
    axes[-1].set_xlabel("time since recording start (s)")

plt.tight_layout()
out = "/tmp/lj_replot2.png"
plt.savefig(out, dpi=120)
print("saved:", out)
plt.show()
