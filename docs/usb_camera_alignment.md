# USB camera (mini2p) in Orange, and aligning it to PTP cameras

## What is supported

USB/UVC cameras (V4L2) are listed next to the Emergent cameras and can be
streamed and recorded like them. Code: `src/usb_camera.{h,cpp}`.

- Detected at startup from `/dev/video*`, shown as `USB<serial> | /dev/videoN`.
- Config (optional): `~/orange_data/config/local/<set>/USB<serial>.json`, same
  keys as an Emergent config. Only `name`, `width`, `height`, `frame_rate`,
  `gpu_id` matter; `pixel_format`/`color`/`gpu_direct` are forced to 8-bit mono
  in host memory. The rig's camera uses `config/local/mini2p/USB00000001.json`.
- Live view is on by default for USB cameras (stream checkbox in Camera
  Control Setting). The preview is 8-bit, auto-scaled to a slowly decaying peak.
- Recording writes, per session folder:
  - `CamUSB<serial>.mp4` + `_meta.csv` (8-bit, same as Emergent cameras)
  - `CamUSB<serial>_raw16.bin` lossless little-endian uint16 frames, shape
    from `_raw16.json`; `_raw16_meta.csv`: `frame_id,v4l2_sequence,
    timestamp_mono_ns,timestamp_sys_ns`
- Recording folders are chowned back to the `sudo` user when a recording stops.

## The rig's camera (FTDI FT602 video bridge, 0403:602a)

- Y16 250x1000, advertises 30 fps, **really delivers 28.1 fps** (35.59 ms).
  The mp4 is tagged 30 fps and plays ~6% fast; timestamps are correct.
- **Must be on a USB 3 port.** At USB 2 (480M) it sends no data at all.
- **Stopping and restarting its stream often leaves it silent until a
  physical replug** (not autosuspend; ruled out). Orange therefore starts the
  stream at *Open camera* and keeps it running until *Close camera*; streaming
  and recording sessions attach to it. After Close camera (or quitting
  Orange), replug before the next open. Probing it with ffmpeg etc. also
  counts as a stop/restart.
- Observed pixel values 8-226 in a 16-bit container (dim signal, or the
  source only fills the low byte) - check with real signal.

## Behaviour in a multi-camera PTP session (e.g. 180 fps)

The USB camera cannot be triggered or PTP-locked. It free-runs on its own clock.

- Start: its thread waits until the Emergent cameras reach the PTP gate
  (`ptp_start_reached`), then records; first frame within ~1 USB frame period.
  At the PTP start barrier it counts as already arrived
  (`PTPParams::num_usb_cameras`); it joins the networked stop barrier itself.
- No fixed frame ratio (180 / 28.1 = 6.4, and the clocks drift).
- Every USB frame has host `timestamp_sys`; every Emergent frame has PTP
  `timestamp` and host `timestamp_sys`. The host clock is the bridge.

## Alignment

```
python3 scripts/usb_align/align_usb_to_ptp.py <session_folder> [--usb-offset-ms X]
```

1. Fits `host = a*ptp + b` over all Emergent frames, pulled to the low
   envelope (minimum arrival latency).
2. Maps each USB `timestamp_sys` into PTP time, minus `--usb-offset-ms`.
3. Writes `CamUSB<serial>_aligned.csv`: per USB frame, its estimated PTP time
   and for each Emergent camera the nearest frame (`_nearest`, `_dt_ms`) and
   the frames inside the USB frame interval `[t - period, t]`
   (`_first` .. `_last`). `-1` = outside that camera's recording.

Validated on synthetic data (2x180 fps + 28.1 fps USB, 5 ppm drift, known
12 ms USB delay): drift recovered (+5.02 ppm), per-frame error = injected USB
jitter (sd 0.49 ms). **Not yet run on a real multi-camera recording.**

### Accuracy limits

- USB arrival jitter: a few ms (measure: spread of `timestamp_sys` diffs).
- Unknown fixed offset: `timestamp_sys` is when the frame *finished
  arriving*, not when it was acquired. For a 2P raster frame (~35 ms scan)
  this is roughly one frame period plus transfer time, so without
  `--usb-offset-ms` USB frames map ~1 frame late. Must be measured.

## Plan

1. **First real session (local).** Emergent camera(s) + USB camera, local
   recording (play button). Check: all threads start/stop, no hang at the
   PTP barriers, USB frame count ~ 28.1 x duration, no V4L2 sequence gaps,
   Emergent drops unchanged vs. without the USB camera.
2. **Networked session.** Same via *Clients start camera threads* / Start
   Recording / Stop Recording. Check the stop barrier (no hang, all cameras
   stop within ~3 s).
3. **Run the alignment script** on both; check drift is small and stable,
   latency p99 reasonable, all USB frames inside the Emergent span.
4. **Measure the USB offset** with a shared event, then record it here and
   pass it as `--usb-offset-ms`:
   - preferred: mini2p frame-start/frame-clock output into a spare LabJack
     AIN (AIN2/AIN0 are taken by the ScanImage frame clock and the IR LED
     witness, see `scanimage_labjack_sync.md`). Gives every USB frame a
     hardware time; then the host-clock path is only a cross-check.
   - otherwise: a light flash visible to both the mini2p and an Emergent
     camera; offset = USB PTP estimate - Emergent PTP time of the flash.
5. **If the frame clock is wired:** extend the script to match USB frames to
   LabJack edges (by order, using `v4l2_sequence` to account for drops) and
   use those times instead of `timestamp_sys`.
6. Optional: tag the mp4 with the measured rate (28.1 fps) instead of 30.

## Open questions

- Does the mini2p expose a frame-start / frame-clock signal?
- Do the 16-bit values use the full range with real signal?
- Does the vendor software stop/restart the stream, or reset the device
  (vendor USB interface / UVC extension unit `25a6dcb8-...`)?
