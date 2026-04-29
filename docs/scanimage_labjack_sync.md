# ScanImage ↔ LabJack ↔ lime synchronization

This integration lets the IR camera array (`orange` / `orange_client`) record
in lockstep with a ScanImage 2-photon microscope. A LabJack T7 reads
ScanImage's exported frame-clock TTL, and lime fires camera triggers on each
falling edge so every IR frame is phase-aligned to the microscope.

## Measured rig timing (2026-04-28)

Frame clock on this rig at 40 Hz, measured live with ScanImage running and
the frame clock on LabJack T7 AIN2 (`/tmp/lj_plot.py`):

| Phase | Duration | Notes |
|---|---|---|
| Period | **25 ms** | 40 Hz |
| HIGH (active scan) | **22.83 ms** | y-galvo scanning, laser on sample |
| LOW (flyback) | **2.16 ms** | y-galvo retrace, laser between rasters |
| Duty cycle | 91 % HIGH / 9 % LOW | |

Use **2.16 ms** as the flyback budget for exposure decisions. (~2× wider
than the 1 ms placeholder I had inferred from the MINI2P paper.) If
ScanImage settings change — different FOV, zoom, scan rate, MEMS frequency
— re-measure with `/tmp/lj_plot.py` before recording.

## ⚠️ TODO before first real recording

1. **Set camera `exposure` to fit inside the 2.16 ms flyback.** The mini2p
   configs currently inherit `"exposure": 250` (µs) — already safe (12 % of
   the flyback window, ~1.9 ms of margin after exposure ends). You can
   crank up to ~1500 µs (≈70 % of flyback) for more SNR, or stay at 250
   µs for a generous safety margin. To bulk-edit:
   ```bash
   # on each machine (master + vlan-dosa0 + vlan-dosa1)
   sed -i 's/"exposure": [0-9]*/"exposure": <NEW_US>/' \
     /home/ratan/orange_data/config/network/mini2p_*/*.json
   ```
2. **Confirm the 2P PMT optics include an IR-blocking filter** before using
   1:2 / 1:4 modes (see "IR rate: 1:N modes" below). If the filter isn't
   confirmed, stay on 1:1.
3. **Click "Search for LabJack" BEFORE clicking "Open Cameras".** The LJ
   mode flag is latched at the OPENCAMERA broadcast moment, so a Search
   click after Open Cameras has no effect this session — you'd record a
   PTP-only run with `lj_edge_index=0` throughout. The Search button
   intentionally disables once cameras are open as a hint.

## At a glance

```
                 ┌─────────────────┐
                 │   ScanImage     │  exports a 5V TTL frame clock
                 │  (2P scope)     │  HIGH = scan, LOW = y-galvo flyback
                 └────────┬────────┘
                          │  BNC
                          ▼
        ┌──────────────────────────────────────────────────────────┐
        │ MASTER (dosa-live)                                       │
        │ ┌────────────┐                  ┌──────────────────┐     │
        │ │ LabJack T7 │── stream AIN2 ──▶│ LabJackTrigger    │     │
        │ │  USB       │   10 kHz/10-spr  │  reader thread   │     │
        │ └────────────┘                  │  - falling-edge  │     │
        │                                 │    detection     │     │
        │                                 │  - cv + queue    │     │
        │                                 └────┬─────────────┘     │
        │           ┌────────────────────────┐ │ ┌────────────┐    │
        │           │ master local cameras   │◀┘ │ enet_thread│    │
        │           │ (5 GigE Emergent)      │   │ broadcasts │    │
        │           │ wait_for_next_edge →   │   │ LJEDGE msgs│    │
        │           │ TriggerSoftware        │   └─────┬──────┘    │
        │           └────────────────────────┘         │           │
        └──────────────────────────────────────────────┼───────────┘
                                                       │ ENet
                                  ┌────────────────────┴────────────────────┐
                                  │                                          │
            ┌─────────────────────▼──────────┐         ┌─────────────────────▼──────────┐
            │ vlan-dosa0 (orange_client)     │         │ vlan-dosa1 (orange_client)     │
            │  inject_edge() →               │         │  inject_edge() →               │
            │  wait_for_next_edge →          │         │  wait_for_next_edge →          │
            │  TriggerSoftware → 6 cameras   │         │  TriggerSoftware → 6 cameras   │
            └────────────────────────────────┘         └────────────────────────────────┘
```

## What runs where

| Machine | Binary | Role |
|---|---|---|
| dosa-live (this PC) | `targets/orange` | master GUI; owns the LabJack T7 over USB; drives ENet broadcasts |
| vlan-dosa0 | `targets/orange_client` | headless client, ~6 cameras |
| vlan-dosa1 | `targets/orange_client` | headless client, ~6 cameras |

Master links `libLabJackM`. Clients use a HEADLESS-stubbed `LabJackTrigger`
(no LJM dep) — they receive edges via ENet and inject them into their local
copy of the trigger object so the per-camera wait/notify logic is identical
on both sides.

## Trigger composition: PTP + LabJack

Two orthogonal mechanisms run **at the same time**:

- **PTP (IEEE 1588)** — synchronizes the *internal clocks* of the 16-ish
  cameras across the LAN. Configured by `ptp_camera_sync()`
  (`src/camera.cpp:598`). Without PTP, each camera timestamps frames against
  its own local clock and per-camera timestamps drift. With PTP, all cameras
  agree on time to sub-µs.
- **LabJack-gated triggering** — controls *when* `TriggerSoftware` fires.
  `get_one_frame()` (`src/video_capture.cpp:372`) waits on the next falling
  edge from the `LabJackTrigger` and only then fires the trigger.

These are independent. The cameras are configured for `MultiFrame` +
`AcquisitionFrameCount=N` + `TriggerMode=On` + `TriggerSource=Software` +
`PtpMode=TwoStep`. PTP sets up a shared start-time gate; LJ provides the
per-frame trigger pulses.

## IR rate: 1:N modes

The "IR rate" dropdown in the master GUI (under **LabJack T7 / ScanImage
sync**) selects how many camera frames the camera bursts per LJ falling edge:

| Mode | N | IR rate at 40 Hz scope | Where IR exposures land |
|---|---|---|---|
| 1:1 | 1 | 40 Hz  | All inside flyback (~1 ms window). No 2P contamination. |
| 1:2 | 2 | 80 Hz  | Frame 1 in flyback, frame 2 mid-scan |
| 1:4 | 4 | 160 Hz | Frame 1 in flyback, frames 2–4 mid-scan |

**The geometric reason 1:N can't keep all frames in flyback**: at 40 Hz the
microscope cycle is 25 ms, of which only ~1 ms is flyback. A 1:4 burst spaces
4 frames evenly across 25 ms (one every 6.25 ms) — there's no way to fit
N>1 exposures inside a 1 ms window because the camera's per-frame readout
takes several ms at full resolution.

**Implication**: at 1:N (N>1), the IR LED — driven by the camera GPO during
exposure — fires while the 2P laser is actively scanning the sample. Whether
that contaminates the 2P signal depends on whether your PMT optics include an
IR-blocking filter. **Verify before running 1:N modes.** If your PMT doesn't
filter IR, stick to 1:1.

The first frame of each burst is always at the falling edge (≈ start of
flyback), so 1:1 mode preserves the strict flyback-gated property.

## Per-frame metadata

Each camera writes `Cam<serial>_meta.csv` next to its `.mp4`:

```
frame_id,timestamp,timestamp_sys,lj_edge_index,lj_edge_timestamp_ns
0,1745701234500000000,1745701234500000123,1,1745701234500000050
1,1745701234525000000,1745701234525000200,2,1745701234525000075
...
```

- `frame_id`: monotonic from the camera
- `timestamp`: camera-internal (PTP-synced across all cameras)
- `timestamp_sys`: host wall-clock at the moment `EVT_CameraGetFrame()` returned
- `lj_edge_index`: which LJ falling edge triggered this frame (1-based, monotonic across the session). At 1:N, all N frames in a burst share the same `lj_edge_index`.
- `lj_edge_timestamp_ns`: host wall-clock at the moment the edge was detected (master's clock for clients too — they receive it via ENet)

Both `lj_edge_*` columns are 0 for frames acquired with LJ trigger disabled.

Use `lj_edge_index` for post-hoc alignment to ScanImage frames — frame N of
ScanImage corresponds to all rows in the IR meta CSV with `lj_edge_index == N`.

### Analysis: how to pair IR frames in this metadata

There are two distinct pairing problems. Use a different key for each.

| Pairing question | Best key | Accuracy |
|---|---|---|
| Which IR frames across the 16 cameras correspond to the same physical instant? | **`timestamp`** (camera-internal PTP) | sub-µs, never drifts |
| Which microscope frame K does an IR frame belong to? | **`lj_edge_index`** | exact for ~14/16 cams; rare ±1 drift on slow cams |

PTP and the LJ trigger encode different information and aren't
substitutes:
- PTP synchronizes the **camera clocks** to each other across the LAN.
- LJ synchronizes **when each camera fires** to ScanImage's flyback,
  AND tags each IR frame with the microscope frame number it
  corresponds to.

You can't drop the LJ trigger — it's what makes the cameras fire on
microscope events in the first place. ScanImage's clock and the IR
cameras' PTP clock are independent; the LJ edge is the bridge between
them. `lj_edge_index` in the metadata is essentially "free" — the
trigger has to label the frame anyway.

### Master `lj_edge_index` is the trustworthy source of truth

The drift can't happen on master cams. On master, the LJ edge counter is
updated by the local reader thread, which only fires once per ScanImage
falling edge (~25 ms apart). Master's camera threads have a generous
window to wake up and read the same value before the next edge update
arrives. Empirically: master cams never drift among themselves.

The drift only affects **clients (dosa0, dosa1)** because there, the
counter is updated by the ENet receive thread calling `inject_edge`.
ENet can deliver packets in microsecond-spaced bursts (network buffering),
shrinking the contention window between two `inject_edge` calls and
giving some camera threads time to wake up and read the second update.

**So treat master's `lj_edge_index` as authoritative**, and pair
dosa0/dosa1 frames to master by PTP timestamp:

```python
# Pick any master cam; they all agree.
ref = master_cam_df

for K in microscope_frames_of_interest:
    master_rows = ref[ref.lj_edge_index == K]
    if master_rows.empty:
        continue  # master itself didn't capture K (rare — real frame drop)
    ref_ptp = master_rows.iloc[0].timestamp

    # Match each client cam by PTP nearest, ignoring its lj_edge_index.
    paired = []
    for client_cam_df in client_cam_dfs:
        idx = (client_cam_df.timestamp - ref_ptp).abs().idxmin()
        dt = abs(client_cam_df.loc[idx].timestamp - ref_ptp)
        if dt < 1_000_000:  # 1 ms tolerance — sub-µs is typical
            paired.append(client_cam_df.loc[idx])
    # paired has 1 row per client cam, all physically synced to the master
    # frame for microscope frame K, regardless of accumulated drift.
```

This is the simplest correct workflow. The more general PTP-fallback
recipe below is still valid (it handles the very rare case where master
itself drops a frame, by PTP-matching across all cams symmetrically).

### Recipe: get all 16 cameras' frames for a given microscope frame K

This is the pattern that gets **16/16 cameras every time** despite the
rare ±1 `lj_edge_index` drift:

```python
# Pseudocode — adapt to your analysis stack.

# Step 1: index every IR frame by camera, with both keys.
rows = []
for cam_serial in all_16_serials:
    df = pandas.read_csv(f"Cam{cam_serial}_meta.csv")
    df["serial"] = cam_serial
    rows.append(df)
all_rows = pandas.concat(rows)

# Step 2: pull rows where the LJ edge index matches.
#         Most cams will be in here.
target_K = 1708
direct = all_rows[all_rows.lj_edge_index == target_K]

# Step 3: any cameras that didn't show up?
present = set(direct.serial)
missing = set(all_16_serials) - present
if missing:
    # Step 4: use PTP timestamp to recover the frames from cams whose
    # lj_edge_index drifted by ±1. Pick a reference PTP from one of
    # the cams we DID find for K, then for each missing cam pull the
    # frame whose PTP timestamp is within (say) ±1 ms of that.
    ref_ts = direct.iloc[0].timestamp
    for serial in missing:
        cam_df = all_rows[all_rows.serial == serial]
        nearest = (cam_df.timestamp - ref_ts).abs().idxmin()
        if abs(cam_df.loc[nearest].timestamp - ref_ts) < 1_000_000:  # 1 ms
            direct = pandas.concat([direct, cam_df.loc[[nearest]]])
        else:
            # That camera genuinely missed this microscope frame
            # (very rare; would require a real frame drop in addition
            # to the labeling drift). Decide whether to skip or
            # interpolate.
            pass

# direct now has up to 16 rows, one per camera, all from the same
# physical microscope flyback K, even when 1-2 cams labeled their
# row K+1 instead of K.
```

The PTP fallback in step 4 handles the labeling-drift case (rare ±1
mismatch) and ALSO handles the rare-but-real frame-drop case — if a
cam genuinely missed the frame, no PTP within ±1 ms exists and you
explicitly know that camera lost this microscope cycle.

### Known: rare ±1 mid-recording `lj_edge_index` drift across cameras

You may occasionally see one or two cameras (more often on the rigs than
the master) whose `lj_edge_index` skips by 2 instead of 1 on a single row
— e.g. master cam goes `…, 1707, 1708, 1709, …` but a dosa1 cam goes
`…, 1707, 1709, 1710, …`. This **is not a frame drop**.

**The offset is sticky** — once a cam skips an edge in its labels, every
subsequent label on that cam stays +1 ahead of master for the rest of the
recording (and if it drifts again, +2, etc.). It does not self-correct.
That's because each camera's "next label" is computed from its own
previous label (`last_seen + 1`), so the offset compounds. So the
mitigation is *not* "skip one row on the affected cam" — it's "do
per-frame matching by PTP timestamp" (the recipe above), which works
regardless of how many sticky drifts have accumulated.

Verification: check the camera-internal `timestamp` (PTP) at that frame.
If all 16 cameras' PTP timestamps for the affected `frame_id` are within
~50 ns of each other, every camera grabbed the frame at the same physical
instant — no frame was lost; only the label disagrees.

What's actually happening: when an edge arrives, master broadcasts
LJEDGE; clients ingest it and notify their camera-thread waiters. If a
camera thread happens to be slow returning to `wait_for_next_edge`
(typically because it was finishing an `EVT_CameraGetFrame` + push to
encoder), the next edge may arrive before the thread reads
`edge_counter`. `wait_for_next_edge` returns the *latest* counter value,
so the thread labels the next frame with that — skipping the
intermediate edge in the label even though the frame physically lands
at the right PTP time.

Diagnostic recipe (works from dosa-live with sshfs mounts up):

```bash
session=<session_dir>
# Find any cameras whose last edge differs from the median:
for f in /home/ratan/orange_data{,_dosa0,_dosa1}/exp/unsorted/$session/Cam*_meta.csv; do
  echo "$(tail -1 "$f" | awk -F',' '{print $4}') $(basename $f)"
done | sort
# For an outlier cam, compare PTP timestamps frame-by-frame against master:
diff <(awk -F',' 'NR>1 {print $1, $2}' .../Cam<master>_meta.csv) \
     <(awk -F',' 'NR>1 {print $1, $2}' .../Cam<outlier>_meta.csv)
# If PTP-column diffs are <100 ns at every row, no frames were lost.
```

Implications for analysis:

1. **Align by PTP `timestamp` for the strictest sub-µs cross-camera
   alignment** — this column is rock solid and unaffected by the LJ
   labeling race.
2. **`lj_edge_index` is the right key for "which ScanImage frame did
   this IR frame correspond to"**, with the caveat that a slow camera
   thread may attribute a frame to edge K+1 instead of K. The frame is
   still physically on the boundary between edges K and K+1; it just
   gets labeled with whichever was current at thread wakeup.
3. If you're computing per-microscope-frame statistics (e.g.
   "average IR brightness during edge 1708"), and one or two cameras
   have no row with `lj_edge_index=1708`, **don't drop those cameras**
   from that bin — their frame at "edge 1709" is the one you want
   (it's physically the same moment as everyone else's "edge 1708"
   row). Match by PTP timestamp instead, or accept the ±1 jitter.

If strict 1:1 sequential labels (no skipping) matter more than physical
edge attribution, the fix is one line in `video_capture.cpp::get_one_frame`:
change `wait_for_next_edge` to return `last_seen + 1` instead of the
current `edge_counter`. This trades the rare cosmetic skip for a small
mis-attribution when a camera thread is genuinely slow (the frame is
physically near edge N+2 but gets labeled N+1). Not currently the
default — current behavior reports the truth about which edge was
current at trigger time.

## Hardware setup

1. ScanImage exported frame-clock BNC → LabJack T7 **AIN2 + GND**. The signal
   is ~5V TTL; AIN2 default range is ±10V so it's fine. Threshold for edge
   detection is hardcoded at 2.5 V (see `THRESH` in `src/labjack_trigger.cpp`).
2. T7 connected to dosa-live over USB. Confirmed serial: 470033341 (LJM driver
   at `/usr/local/lib/libLabJackM.so.1.20.1`).
3. The 16 cameras are wired and reachable on the camera VLAN as before — no
   new wiring needed for this integration.

## File map

| File | Role |
|---|---|
| `src/labjack_trigger.{h,cpp}` | T7 streamer + edge detector + thread-safe wait/notify. HEADLESS branch is the client stub. |
| `src/video_capture.cpp:get_one_frame` | Per-frame: wait for LJ edge → fire `TriggerSoftware` → `EVT_CameraGetFrame`. Burst counter implements 1:N. |
| `src/camera.cpp:ptp_camera_sync` | Camera config (`AcquisitionMode=MultiFrame`, `AcquisitionFrameCount=N`, etc.). Now takes `frames_per_edge` arg. |
| `src/orange.cpp` (master) | CLI flag parsing, GUI panel, "Search for LabJack" button, IR-rate dropdown, OPENCAMERA broadcast. |
| `src/orange_headless_client.cpp` | Receives OPENCAMERA + LJEDGE; injects edges; routes lj_trigger_mode/lj_frames_per_edge into `camera_control`. |
| `src/enet_thread.h` | Master-side ENet event loop. Drains `LabJackTrigger::pending_` and broadcasts LJEDGE messages from the same thread as `service_network` (avoids ENet host race). |
| `src/project.{h,cpp}` | `host_broadcast_open_cameras` (now carries lj flags) and `host_broadcast_lj_edge` (new). |
| `schema/fetch.fbs` | FlatBuffers schema. Adds `LJEDGE` ServerControl + `lj_trigger_mode`/`lj_edge_index`/`lj_edge_timestamp_ns`/`lj_frames_per_edge` fields. Regenerate with `flatc --cpp -o src/ schema/fetch.fbs` (need flatc 23.5.26 to match the static_assert). |
| `src/fetch_generated.h` | Regenerated. Don't hand-edit. |
| `src/gpu_video_encoder.{h,cpp}` | `WORKER_ENTRY` carries the lj edge fields; `write_metadata` writes the new CSV columns. |
| `src/image_processing.h` | `WORKER_ENTRY` struct definition. |
| `src/video_capture.h` | `CameraControl` carries `lj_trigger_mode`, `lj_trigger`, `lj_frames_per_edge`. |
| `src/gui.h` | `start_camera_streaming` passes `lj_frames_per_edge` into `ptp_camera_sync`. |

## Build

Local (master), full build:

```bash
cd /home/ratan/src/lime
./build.sh         # wrapper for ./quick_build/orange.sh
./targets/orange [--lj-trigger]
```

`--lj-trigger` is optional: if set, master calls `lj_trigger.start()` at
launch (pre-clicks the GUI button). Without it, the user can click "Search for
LabJack" in the GUI to do the same thing at runtime.

Headless clients (lime is **not** under git on the rigs — push files via
rsync, see `reference_lime_rig_deploy.md` in the assistant memory):

```bash
# from dosa-live, after editing
for h in vlan-dosa0 vlan-dosa1; do
  rsync -aR src/<changed_files...> "$h:/home/ratan/src/lime/"
  ssh "$h" 'cd src/lime && bash quick_build/orange_client.sh'
done
```

`nvcc: command not found` is benign on the rigs: `kernel.cu` rarely changes
and the existing `targets/kernel.o` is reused. If you do change `kernel.cu`,
prepend `PATH=/usr/local/cuda/bin:$PATH` to the ssh.

`./targets/orange_client` does **not** link `libLabJackM` (the HEADLESS branch
of `labjack_trigger.cpp` provides empty stubs). Verify with
`ldd targets/orange_client | grep -i labjack` — should be empty.

## Operating procedure

1. Start `./targets/orange` on dosa-live. Network panel shows
   `(disconnected)` and a `Search for LabJack` button under
   **LabJack T7 / ScanImage sync**.
2. Plug ScanImage's frame-clock BNC into AIN2 if not already.
3. Click **Search for LabJack**. On success the line turns green:
   `[OK] LabJack connected — edges seen: N` (counter increments at 40 Hz
   while the scope is running).
4. In the rig selector (radio buttons at the top of the Network panel), pick
   the matching config: **mini2p_40hz** for 1:1, **mini2p_80hz** for 1:2,
   **mini2p_160hz** for 1:4. Each config has `frame_rate` set so the camera
   paces frames evenly across the microscope cycle.
5. Pick the **IR rate** from the dropdown (1:1 / 1:2 / 1:4) — must match the
   rig config you just selected.
6. Click **Open Cameras**. Master broadcasts OPENCAMERA carrying
   `lj_trigger_mode=true` and `lj_frames_per_edge=N`. Clients read both and
   set their own `camera_control` accordingly.
7. Click **Clients start camera threads**. PTP sync runs. Cameras enter the
   per-frame loop, each blocked on `wait_for_next_edge`.
8. Click **Start Recording**. PTP gate opens; cameras start grabbing. Each
   ScanImage falling edge fires a `TriggerSoftware` burst across all 16
   cameras; each frame is recorded with its `lj_edge_index` in the metadata
   CSV.
9. Click **Stop Recording** to end. The "Search for LabJack" button stays
   connected (T7 stream keeps running until the binary exits).

## Latency budget (rule of thumb)

```
ScanImage falling edge
        │
        │ ≤ 1 ms     (T7 stream chunk: SCAN_RATE=10kHz, SCANS_PER_READ=10)
        ▼
LabJackTrigger reader detects edge, notifies cv
        │
        │ ≤ 100 µs   (cv wakeup + thread scheduling)
        ▼            ──▶ master local cameras fire here
enet_thread loop dequeues + broadcasts LJEDGE
        │
        │ ≤ 1 ms     (ENet over LAN)
        ▼
Client enet handler → inject_edge → cv notify
        │
        │ ≤ 100 µs
        ▼            ──▶ remote cameras fire here
```

End-to-end **master → remote camera trigger**: typically ~2–3 ms after the
ScanImage edge. Well inside the 25 ms microscope period at 40 Hz, but tight
relative to the ~1 ms flyback window — that's why `SCANS_PER_READ` was
reduced from 1000 to 10 (was originally giving ~50 ms latency, which would
have missed flyback entirely).

## Smoke-test priorities (verify on first real run)

These are the assumptions most likely to misbehave on first light. Test in
this order — each is independent.

1. **Emergent MultiFrame + N>1 trigger semantics**. The code assumes one
   `TriggerSoftware` in `MultiFrame` mode with `AcquisitionFrameCount=N`
   produces a burst of N frames. The PTP-only path uses N=1, so this is
   only stressed by the new 1:N modes. **Test**: pick `mini2p_80hz` rig +
   `1:2` dropdown. Plug a 40 Hz square wave into AIN2 (function generator).
   You should see exactly two `Cam<serial>_meta.csv` rows per
   `lj_edge_index` per camera. If you see only one row per edge, the camera
   is NOT bursting — we need to fire N `TriggerSoftware`s per edge instead
   of relying on `AcquisitionFrameCount`.

2. **PTP gate phantom first trigger**. Possible explanation for how the
   existing PTP path produces frames without an explicit per-frame
   trigger. If the PTP gate auto-fires one trigger when it opens, then in
   LJ mode the recording's first frame has `lj_edge_index=0`. **Test**:
   start a recording, immediately stop, look at the first row of any
   `Cam<serial>_meta.csv`. If `lj_edge_index=0` for that row but
   subsequent rows are 1, 2, 3..., it's the gate phantom. Workaround for
   analysis: drop rows where `lj_edge_index=0`. Workaround for code: don't
   set `PtpAcquisitionGateTime` in LJ mode (skip that part of
   `start_ptp_sync`) — but this loses the cross-camera start barrier, so
   only do it if the phantom turns out to be a real problem.

3. **Master/client clock skew on `lj_edge_timestamp_ns`**. Master timestamps
   each edge with its own `CLOCK_REALTIME` and broadcasts to clients.
   Clients write that into their `Cam<serial>_meta.csv`. `timestamp_sys`
   is the local `CLOCK_REALTIME` on the machine that wrote the row.
   `lj_edge_timestamp_ns` is always master's clock. If master and clients
   aren't NTP-synced, the two columns are in different clock frames and
   may differ by seconds. **Test**: `ssh vlan-dosa0 'date +%s.%N'` vs
   `date +%s.%N` on master, look for offset. If > a few hundred ms, set up
   NTP on the rigs (chrony recommended).

## Known gotchas / future work

- **PMT IR filter**: confirm whether the 2P PMT optics filter the IR
  wavelength before running 1:N modes (see "IR rate" section).
- **Flyback duration is not measured** — the docs assume ~1 ms based on the
  MINI2P paper but this is rig-specific and we should measure with
  `/tmp/lj_edges` or `/tmp/lj_plot.py` (small scratch tools that stream AIN2
  and report LOW/HIGH/PERIOD pulse widths).
- **Rig folder ↔ IR-rate dropdown must match**. The rig JSON has a baked-in
  `frame_rate`; the IR rate dropdown sets the burst size N. They must
  agree (`mini2p_40hz` ↔ 1:1, `mini2p_80hz` ↔ 1:2, `mini2p_160hz` ↔ 1:4)
  or the camera will time out / drop triggers. The GUI prints an orange
  warning under the dropdown if you mismatch them, but doesn't *prevent*
  it — verify before clicking Open Cameras.
- **LJ controls lock at "Open Cameras"**. The Search button and the IR-rate
  dropdown are disabled once `camera_control->open` is true: their values
  have already been broadcast to clients in OPENCAMERA. If you need to
  change them, close the cameras first.
- **`camera_setup_lj_trigger` is dead code** — left declared in `camera.h`
  for now; unused since PTP+LJ compose. Safe to remove in a cleanup pass.
- **Volume clock not yet wired** — the user-facing model treats AIN2 as
  generic "edges per second", so you can plug ScanImage's volume clock into
  AIN2 instead of the frame clock and it Just Works (slower IR rate, longer
  flyback per cycle). Document the choice in the recording metadata so post-
  processing knows what the edges mean.
- **The `lj_trigger_mode` flag is a plain `bool`** read from camera threads
  while written from the GUI thread — technically a data race. Reads and
  writes of aligned bool are atomic on x86 in practice, so no observed
  issues. Make atomic if migrating to a stricter platform or running with
  TSAN.
- **Master button-enable check** (`ptp_counter == num_cameras` at
  `orange.cpp:~426`) gates the recording buttons on PTP completion. With
  PTP+LJ composing, ptp_counter still increments because PTP is not skipped,
  so the buttons enable normally. (Earlier drafts skipped PTP and needed a
  bypass — that draft was reverted; no bypass is currently present.)

## Smoke testing

Without ScanImage, you can validate the LJ→camera→metadata pipeline by
plugging a function generator (5 V square wave, 40 Hz, 50% duty) into AIN2
in place of the ScanImage clock. The "edges seen" counter should tick at the
generator's rate, cameras should record at IR-rate × N, and meta CSV rows
should carry monotonic `lj_edge_index` values.

Two scratch tools live in `/tmp/`:

- `/tmp/lj_edges` — prints one line per detected falling edge with rolling
  flyback / cycle / rate stats. Build with
  `gcc /tmp/lj_edges.c -o /tmp/lj_edges -lLabJackM`.
- `/tmp/lj_plot.py` — live matplotlib plot of AIN2 with TTL pulse-width
  readout overlay. Pipe `/tmp/lj_live` into it.

Both are independent of the lime build and useful for sanity-checking the
signal before involving the cameras.
