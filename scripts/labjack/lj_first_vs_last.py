#!/usr/bin/env python3
"""Compare first 3s and last 3s of an AIN2 recording.

Two stacked panels, same y-scale, with LJ edge markers (red dash) and
camera frame timestamps (green) overlaid where available.
"""
import os, struct, glob, csv, sys
import numpy as np
import matplotlib.pyplot as plt

WIN_S = 3.0

session = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("/home/ratan/orange_data/exp/unsorted/202*"))[-1]
print("session:", session)

bin_path = os.path.join(session, "labjack_analog.bin")
with open(bin_path, "rb") as f:
    start_ns, rate = struct.unpack("Qd", f.read(16))
    samples = np.frombuffer(f.read(), dtype=np.float64).copy()
total_s = len(samples) / rate
print(f"  {len(samples)} samples @ {rate:.1f} Hz = {total_s:.2f} s")

t_axis = np.arange(len(samples)) / rate

# Master cam (frame timestamps in same epoch as start_ns)
master_csv = None
for p in sorted(glob.glob(os.path.join(session, "Cam*_meta.csv"))):
    with open(p) as f:
        r = next(csv.DictReader(f), None)
        if r and abs(int(r["timestamp_sys"]) - start_ns) < int(60e9):
            master_csv = p
            break

frame_t, edge_idx, edge_t = [], [], []
if master_csv:
    print("master cam:", os.path.basename(master_csv))
    with open(master_csv) as f:
        for r in csv.DictReader(f):
            frame_t.append((int(r["timestamp_sys"]) - start_ns) / 1e9)
            ei = int(r["lj_edge_index"])
            et = (int(r["lj_edge_timestamp_ns"]) - start_ns) / 1e9
            if ei > 0:
                edge_idx.append(ei)
                edge_t.append(et)

# Detect LJ edges directly from analog (covers the start gap before first frame)
above = samples > 2.5
falling_idx = np.where(np.diff(above.astype(np.int8)) < 0)[0]
falling_t = falling_idx / rate

windows = [
    (0.0, WIN_S, "first 3 s"),
    (max(0.0, total_s - WIN_S), total_s, f"last 3 s (ends at {total_s:.2f}s)"),
]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharey=True)
fig.suptitle(f"{os.path.basename(session)}  /  AIN2  first vs last {WIN_S:.0f}s")

for ax, (t0, t1, label) in zip(axes, windows):
    m = (t_axis >= t0) & (t_axis <= t1)
    ax.plot(t_axis[m], samples[m], lw=0.5, color="C0")
    ax.axhline(2.5, color="orange", lw=0.5, ls=":")

    # Edges from analog (every falling edge in the window)
    fe = [t for t in falling_t if t0 <= t <= t1]
    for t in fe:
        ax.axvline(t, color="red", lw=0.4, alpha=0.4, ls="--")

    # Camera frame arrivals (host clock)
    fr = [t for t in frame_t if t0 <= t <= t1]
    for t in fr:
        ax.axvline(t, color="green", lw=0.5, alpha=0.5)

    ax.set_xlim(t0, t1)
    ax.set_ylim(-1, 6)
    ax.set_ylabel("AIN2 (V)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{label}   [{len(fe)} edges, {len(fr)} frames]")

axes[-1].set_xlabel("time since recording start (s)")
plt.tight_layout()
out = "/tmp/lj_first_vs_last.png"
plt.savefig(out, dpi=120)
print("saved:", out)
plt.show()
