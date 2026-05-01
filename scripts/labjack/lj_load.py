"""Load a labjack analog-stream bin written by lime LabJackTrigger.

Filename: `labjack_analog.bin` for new recordings (May 2026 onward).
Older sessions used `labjack_ain2.bin` — `find_lj_bin()` falls back to
that automatically.

New format (24-byte header, multi-channel):
  uint64 start_ns + double rate_hz + uint32 num_channels + uint32 reserved
  followed by interleaved float64 samples (num_channels per scan,
  scan_list order: AIN2 then AIN0).

Old format (16-byte header, AIN2-only):
  uint64 start_ns + double rate_hz + float64 samples.

`load_lj()` returns (start_ns, rate, channels) where `channels` is a dict
{"AIN2": np.array, "AIN0": np.array_or_None}.
"""
import os, struct, glob
import numpy as np


def find_lj_bin(session_dir):
    """Resolve the analog-stream bin path inside a session folder.
    Tries new name first, falls back to legacy name."""
    for name in ("labjack_analog.bin", "labjack_ain2.bin"):
        p = os.path.join(session_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No labjack_analog.bin or labjack_ain2.bin in {session_dir}")


def load_lj(path):
    with open(path, "rb") as f:
        head8 = f.read(8)
        head16 = f.read(8)
        start_ns = struct.unpack("Q", head8)[0]
        rate = struct.unpack("d", head16)[0]

        # Try to read 8 more bytes for num_channels + reserved. If the
        # remaining data is not divisible by num_channels * 8, it's the
        # old single-channel format.
        peek = f.read(8)
        rest_pos_new = f.tell()
        rest = f.read()  # everything after possible 24B header

    file_size = os.path.getsize(path)
    if len(peek) == 8:
        num_channels = struct.unpack("I", peek[:4])[0]
        # New format if claimed-channel-count is sane AND samples align.
        new_payload = file_size - 24
        if 1 <= num_channels <= 8 and new_payload % (num_channels * 8) == 0:
            samples = np.frombuffer(rest, dtype=np.float64).copy()
            samples = samples.reshape(-1, num_channels)
            chans = {"AIN2": samples[:, 0]}
            if num_channels >= 2:
                chans["AIN0"] = samples[:, 1]
            return start_ns, rate, chans

    # Old format: rewind, treat everything after 16B as float64
    with open(path, "rb") as f:
        f.read(16)
        samples = np.frombuffer(f.read(), dtype=np.float64).copy()
    return start_ns, rate, {"AIN2": samples, "AIN0": None}
