# LabJack analog-stream analysis scripts

Post-processing tools for the `labjack_analog.bin` written by lime
during a recording session (24-byte header + interleaved float64
samples for AIN2 + AIN0 at 10 kHz). See
`docs/scanimage_labjack_sync.md` for the file format.

## Common loader

`lj_load.py` — `load_lj(path)` returns `(start_ns, rate, channels)`,
where `channels` is a dict `{"AIN2": ndarray, "AIN0": ndarray_or_None}`.
Handles both the new 24-byte-header / multi-channel format and the
legacy 16-byte-header / AIN2-only format.

`find_lj_bin(session_dir)` resolves the bin path inside a session
folder, preferring `labjack_analog.bin` and falling back to the legacy
`labjack_ain2.bin`.

## Plot scripts (matplotlib popup + PNG to /tmp)

| script | purpose |
|---|---|
| `lj_replot.py` | Single-channel (AIN2) full trace + 100 ms zoom. |
| `lj_replot2.py` | Both channels (AIN2 + AIN0) stacked, full trace + 100 ms zoom. |
| `lj_compare_channels.py` | Short window with AIN2 falling-edge markers, AIN0 LED pulses, and edge→LED latency arrows. |
| `lj_first_vs_last.py` | First 3 s vs last 3 s of the recording, both panels same y-scale. |
| `lj_first7_vs_next7.py` | Frames 0–6 vs frames 7–13 windowed view. |
| `lj_latency_drift.py` | Edge → LED latency over time + cumulative drift. Tests if scanner and camera clocks are running at different rates. |
| `lj_real_latency.py` | Detailed per-camera CSV stats (period, jitter, latency). |
| `lj_plot.py` | Original AIN2-only plot. |
| `lj_plot_with_frames.py` | AIN2 trace overlaid with camera-meta-CSV edge markers and frame timestamps. |

## Standalone tools

| script | purpose |
|---|---|
| `lj_read_ain0.py` | Streams AIN0 directly via labjack-ljm Python (NO orange running) — useful for hardware-poking before recordings. |
| `lj_pulse_10hz.py` | Fires a test pulse train on a T7 DIO via labjack-ljm. |

## Default behavior

Most scripts auto-pick the latest session in
`/home/ratan/orange_data/exp/unsorted/`. Pass an explicit session path
or `.bin` path to override. PNG outputs go to `/tmp/<script_name>.png`.
