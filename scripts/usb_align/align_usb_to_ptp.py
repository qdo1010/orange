#!/usr/bin/env python3
"""Align USB (V4L2) camera frames to PTP-synced Emergent camera frames.

USB cameras cannot be triggered or PTP-locked; they free-run on their own
clock and stamp each frame with host CLOCK_REALTIME (timestamp_sys). Emergent
metadata carries both the PTP timestamp and timestamp_sys, so the host clock
is the bridge:

  1. Fit host_time = a * ptp_time + b over all Emergent frames. Arrival
     latency only ever adds delay, so the fit is pulled down to the low
     envelope (the minimum-latency frames) rather than the mean.
  2. Map each USB frame's timestamp_sys into the PTP domain, minus a fixed
     USB latency (--usb-offset-ms, default 0 = frame *arrival* time; measure
     it, see docs/usb_camera_alignment.md).
  3. For each Emergent camera, report the nearest frame and the range of
     frames overlapping the USB frame's interval [t - period, t].

Usage:
  python3 align_usb_to_ptp.py <recording_folder>
  python3 align_usb_to_ptp.py <recording_folder> --usb-offset-ms 12.5
  python3 align_usb_to_ptp.py <recording_folder> --usb-serial USB00000001

Writes <recording_folder>/Cam<usb_serial>_aligned.csv and prints a summary.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np


def load_meta(path, columns):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {c: np.array([], dtype=np.int64) for c in columns}
    # The raw16 metadata suffixes its timestamp columns with _ns.
    key = {c: c if c in rows[0] else c + "_ns" for c in columns}
    return {c: np.array([int(r[key[c]]) for r in rows], dtype=np.int64)
            for c in columns}


def drop_stale_usb_frames(usb):
    """Drop frames that sat queued in the driver before recording started.

    The stream runs from camera open, so the first dequeues at record start
    return old buffers: their timestamp_sys is the dequeue time, not the
    arrival time. timestamp_mono (driver stamp) exposes them: timestamp_sys -
    timestamp_mono is constant for live frames and much larger for stale ones.
    """
    if "timestamp_mono" not in usb or not len(usb["frame_id"]):
        return usb, 0
    lag = usb["timestamp_sys"] - usb["timestamp_mono"]
    live = lag - np.median(lag) < 5_000_000  # 5 ms
    return {c: v[live] for c, v in usb.items()}, int((~live).sum())


def find_cameras(folder, usb_serial):
    usb, emergent = {}, {}
    for path in sorted(glob.glob(os.path.join(folder, "Cam*_meta.csv"))):
        name = os.path.basename(path)
        if name.endswith("_raw16_meta.csv"):
            continue
        serial = name[len("Cam"):-len("_meta.csv")]
        if serial.startswith("USB"):
            # Prefer the raw16 metadata: it has the V4L2 sequence number.
            raw = os.path.join(folder, f"Cam{serial}_raw16_meta.csv")
            usb[serial] = raw if os.path.exists(raw) else path
        else:
            emergent[serial] = path
    if usb_serial:
        if usb_serial not in usb:
            sys.exit(f"no metadata for {usb_serial} in {folder}")
        usb = {usb_serial: usb[usb_serial]}
    if not usb:
        sys.exit(f"no USB camera metadata (CamUSB*_meta.csv) in {folder}")
    if not emergent:
        sys.exit(f"no Emergent camera metadata in {folder}")
    if len(usb) > 1:
        sys.exit(f"several USB cameras {sorted(usb)}; pick one with "
                 "--usb-serial")
    return next(iter(usb.items())), emergent


def fit_host_vs_ptp(ptp, host, envelope_pct=10.0, iterations=5):
    """Least-squares host = a*ptp + b, refit on the lowest-residual frames."""
    x = (ptp - ptp[0]).astype(np.float64)
    y = (host - host[0]).astype(np.float64)
    keep = np.ones(len(x), dtype=bool)
    for _ in range(iterations):
        a, b = np.polyfit(x[keep], y[keep], 1)
        resid = y - (a * x + b)
        keep = resid <= np.percentile(resid, envelope_pct)
    resid = y - (a * x + b)
    return a, b, ptp[0], host[0], resid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--usb-serial", default=None)
    ap.add_argument("--usb-offset-ms", type=float, default=0.0,
                    help="fixed delay from USB frame acquisition to its "
                         "timestamp_sys; subtracted before matching")
    ap.add_argument("--usb-period-ms", type=float, default=None,
                    help="USB frame period; default: median of timestamps")
    args = ap.parse_args()

    (usb_serial, usb_path), emergent_paths = find_cameras(args.folder,
                                                         args.usb_serial)
    usb_cols = ["frame_id", "timestamp_sys"]
    with open(usb_path) as f:
        header = f.readline().strip().split(",")
    if "v4l2_sequence" in header:
        usb_cols.append("v4l2_sequence")
    if "timestamp_mono_ns" in header:
        usb_cols.append("timestamp_mono")
    usb, n_stale = drop_stale_usb_frames(load_meta(usb_path, usb_cols))

    emergent = {s: load_meta(p, ["frame_id", "timestamp", "timestamp_sys"])
                for s, p in emergent_paths.items()}
    emergent = {s: m for s, m in emergent.items() if len(m["frame_id"])}
    if not emergent:
        sys.exit("Emergent metadata files are empty")

    # A camera whose host pipeline falls behind has a growing
    # timestamp_sys - timestamp; its PTP stamps are still valid for matching,
    # but its host times would skew the fit, so leave it out of the fit.
    # Healthy cameras still have brief latency spikes, hence the median.
    def lag_spread_ms(m):
        lag = (m["timestamp_sys"] - m["timestamp"]).astype(np.float64)
        return (np.median(lag) - lag.min()) / 1e6
    fit_cams = [s for s, m in emergent.items() if lag_spread_ms(m) < 5.0]
    if not fit_cams:
        sys.exit("every Emergent camera fell behind; no host-clock reference")

    # One fit over the Emergent frames: their PTP clocks are common.
    all_ptp = np.concatenate([emergent[s]["timestamp"] for s in fit_cams])
    all_sys = np.concatenate([emergent[s]["timestamp_sys"] for s in fit_cams])
    order = np.argsort(all_ptp)
    a, b, ptp0, sys0, resid = fit_host_vs_ptp(all_ptp[order], all_sys[order])

    # Work relative to (ptp0, sys0) in int64 first: absolute ns (~1.8e18) in
    # float64 would lose ~256 ns of precision.
    offset_ns = args.usb_offset_ms * 1e6
    usb_rel = (usb["timestamp_sys"] - sys0).astype(np.float64)
    usb_ptp = (usb_rel - b - offset_ns) / a  # ns relative to ptp0

    usb_dt = np.diff(usb["timestamp_sys"]) / 1e6
    period_ms = (args.usb_period_ms if args.usb_period_ms
                 else float(np.median(usb_dt)) if len(usb_dt) else 0.0)
    period_ns = period_ms * 1e6

    out_cols = ["usb_frame_id", "usb_ptp_est_ns"]
    if "v4l2_sequence" in usb:
        out_cols.insert(1, "usb_v4l2_sequence")
    per_cam = {}
    for serial, m in emergent.items():
        t = (m["timestamp"] - ptp0).astype(np.float64)
        nearest = np.clip(np.searchsorted(t, usb_ptp), 1, len(t) - 1)
        nearest -= (usb_ptp - t[nearest - 1]) < (t[nearest] - usb_ptp)
        first = np.searchsorted(t, usb_ptp - period_ns, side="left")
        last = np.searchsorted(t, usb_ptp, side="right") - 1
        inside = (usb_ptp >= t[0]) & (usb_ptp <= t[-1])
        per_cam[serial] = dict(
            nearest=m["frame_id"][nearest],
            dt_ms=(usb_ptp - t[nearest]) / 1e6,
            first=np.where(first <= last, m["frame_id"][np.minimum(first, len(t) - 1)], -1),
            last=np.where(first <= last, m["frame_id"][np.maximum(last, 0)], -1),
            inside=inside)
        out_cols += [f"{serial}_nearest", f"{serial}_dt_ms",
                     f"{serial}_first", f"{serial}_last"]

    out_path = os.path.join(args.folder, f"Cam{usb_serial}_aligned.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(out_cols)
        for i in range(len(usb["frame_id"])):
            row = [int(usb["frame_id"][i])]
            if "v4l2_sequence" in usb:
                row.append(int(usb["v4l2_sequence"][i]))
            row.append(int(ptp0) + int(round(usb_ptp[i])))
            for serial in emergent:
                c = per_cam[serial]
                if c["inside"][i]:
                    row += [int(c["nearest"][i]), f"{c['dt_ms'][i]:.3f}",
                            int(c["first"][i]), int(c["last"][i])]
                else:  # USB frame outside this camera's recording
                    row += [-1, "", -1, -1]
            w.writerow(row)

    # Summary
    ppm = (a - 1.0) * 1e6
    print(f"USB camera {usb_serial}: {len(usb['frame_id'])} frames "
          f"({os.path.basename(usb_path)}), {n_stale} stale queued frames "
          "dropped")
    if len(usb_dt):
        print(f"  period median {period_ms:.3f} ms ({1000 / period_ms:.3f} fps)"
              f", max gap {usb_dt.max():.1f} ms")
    if "v4l2_sequence" in usb and len(usb["v4l2_sequence"]) > 1:
        gaps = np.diff(usb["v4l2_sequence"]) - 1
        print(f"  V4L2 sequence gaps: {int(gaps[gaps > 0].sum())} missing "
              "frames")
    skipped = sorted(set(emergent) - set(fit_cams))
    if skipped:
        print(f"left out of the fit (host pipeline fell behind): "
              f"{', '.join('Cam' + s for s in skipped)}")
    print(f"host-vs-PTP fit over {len(all_ptp)} Emergent frames: "
          f"drift {ppm:+.2f} ppm, host arrival latency above envelope: "
          f"median {np.median(resid) / 1e6:.3f} ms, "
          f"p99 {np.percentile(resid, 99) / 1e6:.3f} ms")
    print(f"USB offset applied: {args.usb_offset_ms} ms")
    for serial, m in emergent.items():
        c = per_cam[serial]
        n_in = int(c["inside"].sum())
        e_dt = np.median(np.diff(m["timestamp"])) / 1e6
        print(f"  Cam{serial}: {len(m['frame_id'])} frames "
              f"({1000 / e_dt:.2f} fps); USB frames within its span: "
              f"{n_in}/{len(c['inside'])}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
