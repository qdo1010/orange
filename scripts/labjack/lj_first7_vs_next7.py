#!/usr/bin/env python3
"""Compare AIN2 around frames 0-6 vs frames 7-13.

Both panels span the same time width so the cycles can be visually
compared side-by-side. Red dash = LJ falling edge, green = frame arrival.
"""
import os, struct, glob, csv, sys
import numpy as np
import matplotlib.pyplot as plt

PAD_MS = 5.0  # padding before first frame in each group

session = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("/home/ratan/orange_data/exp/unsorted/202*"))[-1]
print("session:", session)

bin_path = os.path.join(session, "labjack_analog.bin")
with open(bin_path, "rb") as f:
    start_ns, rate = struct.unpack("Qd", f.read(16))
    samples = np.frombuffer(f.read(), dtype=np.float64).copy()
t_axis_ms = np.arange(len(samples)) / rate * 1000

# Find master cam
master_csv = None
for p in sorted(glob.glob(os.path.join(session, "Cam*_meta.csv"))):
    with open(p) as f:
        r = next(csv.DictReader(f), None)
        if r and abs(int(r["timestamp_sys"]) - start_ns) < int(60e9):
            master_csv = p
            break
if master_csv is None:
    sys.exit("no master cam meta CSV found")
print("master cam:", os.path.basename(master_csv))

fids, frame_t_ms, lj_idx, lj_t_ms = [], [], [], []
with open(master_csv) as f:
    for row in csv.DictReader(f):
        fids.append(int(row["frame_id"]))
        frame_t_ms.append((int(row["timestamp_sys"]) - start_ns) / 1e6)
        lj_idx.append(int(row["lj_edge_index"]))
        lj_t_ms.append((int(row["lj_edge_timestamp_ns"]) - start_ns) / 1e6)

if len(fids) < 14:
    sys.exit(f"only {len(fids)} frames; need 14")

# Detect all falling edges directly so we can render them inside each window
above = samples > 2.5
falling_idx = np.where(np.diff(above.astype(np.int8)) < 0)[0]
falling_t_ms = falling_idx / rate * 1000

# IR LED on-time estimate.
# We have NO direct measurement of trigger-to-shutter latency yet.
# Best evidence so far: 18 ms is host-arrival (incl. readout+transfer), not
# trigger-to-shutter; TriggerDelay=0 in eCapture; predicted firmware path is
# ~1-2 ms.  Drawn here as a wider band for visibility; real LED-on is 25 us.
EST_LATENCY_MS = 1.5    # guess until we measure
DISPLAY_WIDTH_MS = 1.5  # real exposure 25 us; widened so we can see it

groups = [
    (list(range(0, 7)),  "frames 0-6"),
    (list(range(7, 14)), "frames 7-13"),
]

# Use the same time-width on both axes for fair comparison
spans = []
for idxs, _ in groups:
    spans.append(frame_t_ms[idxs[-1]] - frame_t_ms[idxs[0]])
win_ms = max(spans) + 2 * PAD_MS
print(f"window width: {win_ms:.1f} ms")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharey=True)
fig.suptitle(f"{os.path.basename(session)}  /  {os.path.basename(master_csv)}\n"
             f"first 7 frames vs next 7 frames")

for ax, (idxs, label) in zip(axes, groups):
    t0 = frame_t_ms[idxs[0]] - PAD_MS
    t1 = t0 + win_ms

    m = (t_axis_ms >= t0) & (t_axis_ms <= t1)
    ax.plot(t_axis_ms[m], samples[m], lw=0.6, color="C0")
    ax.axhline(2.5, color="orange", lw=0.5, ls=":")

    # All falling edges in window + estimated IR LED-on band per edge
    n_edges = 0
    for et in falling_t_ms:
        if t0 <= et <= t1:
            ax.axvline(et, color="red", lw=0.5, alpha=0.5, ls="--")
            led_t = et + EST_LATENCY_MS
            ax.axvspan(led_t, led_t + DISPLAY_WIDTH_MS,
                       color="orange", alpha=0.35)
            n_edges += 1
    # Legend proxy for the orange band
    ax.axvspan(t0, t0, color="orange", alpha=0.35,
               label=f"est. IR on (edge + {EST_LATENCY_MS:g} ms, width inflated)")

    # Frame arrival markers + labels (only the ones in this group)
    for fi in idxs:
        ft = frame_t_ms[fi]
        ax.axvline(ft, color="green", lw=0.7, alpha=0.7)
        ax.text(ft, 5.4, f"f{fi}", ha="center", fontsize=8, color="darkgreen",
                fontweight="bold")

    ax.set_xlim(t0, t1)
    ax.set_ylim(-1, 6)
    ax.set_ylabel("AIN2 (V)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{label}   "
                 f"[t = {t0:.1f}-{t1:.1f} ms,  {n_edges} edges]")
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("time since recording start (ms)")
plt.tight_layout()
out = "/tmp/lj_first7_vs_next7.png"
plt.savefig(out, dpi=120)
print("saved:", out)
plt.show()
