#!/usr/bin/env python3
"""Live plot of AIN2 from /tmp/lj_live binary stream.

Reads float64 samples from stdin (piped from lj_live), shows a rolling
window of the most recent samples in matplotlib with TTL pulse-width
readout overlay (LOW / HIGH / PERIOD stats).
"""
import sys
import struct
import threading
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SCAN_RATE = 10000.0
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(SCAN_RATE * WINDOW_SECONDS)
CHUNK_BYTES = 8 * 1000
TTL_THRESHOLD = 2.5

buf = collections.deque(maxlen=WINDOW_SAMPLES)
buf.extend([0.0] * WINDOW_SAMPLES)
lock = threading.Lock()
stop = threading.Event()

def reader():
    while not stop.is_set():
        data = sys.stdin.buffer.read(CHUNK_BYTES)
        if not data:
            break
        n = len(data) // 8
        vals = struct.unpack(f"{n}d", data[:n*8])
        with lock:
            buf.extend(vals)
    stop.set()

t_axis = np.linspace(-WINDOW_SECONDS, 0.0, WINDOW_SAMPLES)
fig, ax = plt.subplots(figsize=(10, 4))
(line,) = ax.plot(t_axis, np.zeros(WINDOW_SAMPLES), lw=0.8)
ax.set_xlim(-WINDOW_SECONDS, 0.0)
ax.set_ylim(-10.5, 10.5)
ax.set_xlabel("time (s, relative to now)")
ax.set_ylabel("AIN2 (V)")
ax.set_title(f"LabJack T7 AIN2 — live, {SCAN_RATE:g} Hz, {WINDOW_SECONDS:g}s window")
ax.grid(True, alpha=0.3)

vmin_text = ax.text(0.01, 0.95, "", transform=ax.transAxes, va="top", family="monospace")
edge_text = ax.text(0.01, 0.05, "", transform=ax.transAxes, va="bottom", family="monospace", color="C1")

def pulse_widths(y, dt, thresh=TTL_THRESHOLD):
    above = y > thresh
    diff = np.diff(above.astype(np.int8))
    rising = np.where(diff > 0)[0]
    falling = np.where(diff < 0)[0]
    lows, highs, periods = [], [], []
    for f in falling:
        nxt = rising[rising > f]
        if len(nxt):
            lows.append((nxt[0] - f) * dt * 1000.0)
    for r in rising:
        nxt = falling[falling > r]
        if len(nxt):
            highs.append((nxt[0] - r) * dt * 1000.0)
    if len(falling) >= 2:
        periods = (np.diff(falling) * dt * 1000.0).tolist()
    return lows, highs, periods

def update(_frame):
    with lock:
        y = np.fromiter(buf, dtype=np.float64, count=len(buf))
    line.set_ydata(y)
    if len(y):
        vmin_text.set_text(
            f"min={y.min():+.4f} V  max={y.max():+.4f} V  "
            f"mean={y.mean():+.4f} V  pkpk={(y.max()-y.min()):.4f} V"
        )
        lows, highs, periods = pulse_widths(y, 1.0/SCAN_RATE)
        if lows or highs:
            def stats(arr, name):
                if not arr:
                    return f"{name}: -"
                a = np.array(arr)
                return f"{name}: n={len(a)} mean={a.mean():.3f}ms min={a.min():.3f} max={a.max():.3f}"
            edge_text.set_text(
                stats(lows, "LOW (flyback)") + "\n" +
                stats(highs, "HIGH (scan)  ") + "\n" +
                stats(periods, "PERIOD       ")
            )
        else:
            edge_text.set_text("no edges in window")
    pk = max(abs(y.min()), abs(y.max()), 0.1)
    if pk < 1.0:
        ax.set_ylim(-pk*1.2, pk*1.2)
    elif pk < 10.0:
        ax.set_ylim(-pk*1.1, pk*1.1)
    else:
        ax.set_ylim(-10.5, 10.5)
    return line, vmin_text, edge_text

t = threading.Thread(target=reader, daemon=True)
t.start()
ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
try:
    plt.show()
finally:
    stop.set()
