#!/usr/bin/env python3
"""Standalone: stream AIN0 from the LabJack T7 for a short interval.

Don't run while orange/lime is recording — the T7 only allows one stream
session at a time. Stop orange, run this, fire the IR however you can
(e.g. `discharge_lights.sh`, or eCapture live preview on a camera),
then look at /tmp/lj_ain0.bin and the printed stats.

Usage:
  python3 /tmp/lj_read_ain0.py            # 10 s default
  python3 /tmp/lj_read_ain0.py 30         # 30 s

Output:
  /tmp/lj_ain0.bin   uint64 start_ns + double rate_hz + float64 samples
"""
import sys, time, struct, ctypes
import numpy as np
from labjack import ljm

DUR_S = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
RATE_HZ = 10000.0
SCANS_PER_READ = 100

handle = ljm.openS("T7", "ANY", "ANY")
info = ljm.getHandleInfo(handle)
print(f"opened T7 serial {info[2]}")

ljm.eWriteNames(handle,
                4,
                ["AIN0_RANGE", "AIN0_RESOLUTION_INDEX",
                 "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"],
                [10.0, 0.0, 0.0, 0.0])

scan_list = ljm.namesToAddresses(1, ["AIN0"])[0]
actual_rate = ljm.eStreamStart(handle, SCANS_PER_READ, 1, scan_list, RATE_HZ)
print(f"streaming AIN0 @ {actual_rate:.1f} Hz for {DUR_S:.1f} s")

start_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
all_samples = []
t_end = time.monotonic() + DUR_S
last_print = time.monotonic()

try:
    while time.monotonic() < t_end:
        chunk, dev_bl, ljm_bl = ljm.eStreamRead(handle)
        all_samples.extend(chunk)
        now = time.monotonic()
        if now - last_print >= 1.0:
            arr = np.array(chunk)
            print(f"  +{now - (t_end - DUR_S):5.1f}s  "
                  f"min={arr.min():+.3f}  max={arr.max():+.3f}  "
                  f"mean={arr.mean():+.3f}  "
                  f"backlog dev={dev_bl} ljm={ljm_bl}")
            last_print = now
finally:
    ljm.eStreamStop(handle)
    ljm.close(handle)

samples = np.array(all_samples, dtype=np.float64)
print(f"\ncollected {len(samples)} samples")
print(f"  overall: min={samples.min():+.3f}  max={samples.max():+.3f}  "
      f"mean={samples.mean():+.3f}")

# Pulse detection — anything spending time above midpoint between min/max
mid = (samples.min() + samples.max()) / 2
above = samples > mid
risings = np.where(np.diff(above.astype(np.int8)) > 0)[0]
fallings = np.where(np.diff(above.astype(np.int8)) < 0)[0]
print(f"  threshold (midpoint) {mid:.3f} V")
print(f"  detected {len(risings)} rising edges, {len(fallings)} falling")
if len(risings) >= 2:
    period_s = np.diff(risings) / actual_rate
    print(f"  rising-edge period: mean {period_s.mean()*1000:.3f} ms  "
          f"std {period_s.std()*1000:.3f}  "
          f"-> rate {1/period_s.mean():.2f} Hz")

with open("/tmp/lj_ain0.bin", "wb") as f:
    f.write(struct.pack("Qd", start_ns, actual_rate))
    f.write(samples.tobytes())
print("saved /tmp/lj_ain0.bin")
