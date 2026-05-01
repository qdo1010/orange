#!/usr/bin/env python3
"""Plot a labjack_analog.bin from a recording session.

Usage:
  python3 /tmp/lj_replot.py                  # picks latest unsorted session
  python3 /tmp/lj_replot.py <path_to_bin>    # explicit file
"""
import os
import struct
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


def find_latest():
    root = "/home/ratan/orange_data/exp/unsorted"
    sessions = sorted(glob.glob(os.path.join(root, "202*")))
    if not sessions:
        sys.exit("no sessions in " + root)
    bin_path = os.path.join(sessions[-1], "labjack_analog.bin")
    if not os.path.exists(bin_path):
        sys.exit("no labjack_analog.bin in latest session: " + sessions[-1])
    return bin_path


def load(path):
    with open(path, "rb") as f:
        start_ns, rate = struct.unpack("Qd", f.read(16))
        samples = np.frombuffer(f.read(), dtype=np.float64).copy()
    t = np.arange(len(samples)) / rate
    return t, samples, start_ns, rate


def detect_edges(samples, threshold=2.5):
    above = samples > threshold
    diff = np.diff(above.astype(np.int8))
    falling = np.where(diff < 0)[0]   # idx where sample i+1 went LOW
    rising = np.where(diff > 0)[0]
    return falling, rising


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_latest()
    print(f"loading: {path}")
    t, s, start_ns, rate = load(path)
    print(f"  {len(s)} samples @ {rate} Hz = {t[-1]:.2f} s")
    print(f"  V min={s.min():.3f}  max={s.max():.3f}  mean={s.mean():.3f}")

    falling, rising = detect_edges(s)
    print(f"  detected {len(falling)} falling edges, {len(rising)} rising")
    if len(falling) >= 2:
        periods_s = np.diff(falling) / rate
        print(f"  period: mean={1000*periods_s.mean():.3f} ms  "
              f"std={1000*periods_s.std():.3f} ms  "
              f"→ rate={1.0/periods_s.mean():.2f} Hz")
    if len(falling) and len(rising):
        # LOW pulse widths (falling -> next rising)
        lows = []
        for f in falling:
            n = rising[rising > f]
            if len(n):
                lows.append((n[0] - f) / rate * 1000)
        if lows:
            lows = np.array(lows)
            print(f"  LOW (flyback) width: mean={lows.mean():.3f} ms  "
                  f"min={lows.min():.3f}  max={lows.max():.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    fig.suptitle(os.path.basename(os.path.dirname(path)) +
                 "  /  labjack_analog.bin")

    # Top: full trace + edge markers
    axes[0].plot(t, s, lw=0.5, label="AIN2 (V)")
    axes[0].axhline(2.5, color="orange", lw=0.5, ls=":", label="threshold")
    if len(falling):
        axes[0].scatter(falling / rate, np.full(len(falling), 0.5),
                        s=4, c="red", label=f"falling ({len(falling)})")
    axes[0].set_xlabel("time since recording start (s)")
    axes[0].set_ylabel("AIN2 (V)")
    axes[0].set_ylim(-1, 6)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"full {t[-1]:.1f}s trace")

    # Bottom: zoomed view of first 100 ms (4 cycles at 40 Hz)
    win = int(0.1 * rate)
    axes[1].plot(t[:win], s[:win], lw=0.8)
    axes[1].axhline(2.5, color="orange", lw=0.5, ls=":")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("AIN2 (V)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("first 100 ms")
    axes[1].set_ylim(-1, 6)

    plt.tight_layout()
    out_png = "/tmp/lj_replot.png"
    plt.savefig(out_png, dpi=120)
    print("saved:", out_png)
    plt.show()


if __name__ == "__main__":
    main()
