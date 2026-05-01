#!/usr/bin/env python3
"""Verbose FIO0 pulse test for scope debugging.

Explicitly:
  1. Disables DIO0 extended-feature mode (in case something else
     configured DIO_EF on FIO0)
  2. Sets DIO_DIRECTION bit 0 to 1 (output)
  3. Pulses 10 ms HIGH at 10 Hz
  4. Reads back FIO0 state to confirm it toggled

If you scope FIO0+GND and see nothing, but the readback prints
HIGH/LOW correctly, the chip thinks it's toggling but the pin
isn't physically driven — wiring/labeling issue.

Build/run: python3 /tmp/lj_pulse_10hz.py
"""
import ctypes
import time
import sys

ljm = ctypes.CDLL("libLabJackM.so")

LJM_dtT7 = 7
LJM_ctANY = 0
LJM_MAX_NAME_SIZE = 256

ljm.LJM_Open.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)
]
ljm.LJM_Open.restype = ctypes.c_int
ljm.LJM_Close.argtypes = [ctypes.c_int]
ljm.LJM_Close.restype = ctypes.c_int
ljm.LJM_eWriteName.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_double]
ljm.LJM_eWriteName.restype = ctypes.c_int
ljm.LJM_eReadName.argtypes = [
    ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)
]
ljm.LJM_eReadName.restype = ctypes.c_int
ljm.LJM_ErrorToString.argtypes = [ctypes.c_int, ctypes.c_char_p]
ljm.LJM_ErrorToString.restype = ctypes.c_int


def err_str(err):
    buf = ctypes.create_string_buffer(LJM_MAX_NAME_SIZE)
    ljm.LJM_ErrorToString(err, buf)
    return buf.value.decode()


def write_name(handle, name, value, fatal=True):
    err = ljm.LJM_eWriteName(handle, name.encode(), value)
    if err:
        msg = f"WRITE {name}={value}: error {err} ({err_str(err)})"
        if fatal:
            print(msg, file=sys.stderr)
            sys.exit(1)
        else:
            print(f"  warning: {msg}")
    else:
        print(f"  WRITE {name} = {value}")
    return err


def read_name(handle, name):
    v = ctypes.c_double(0)
    err = ljm.LJM_eReadName(handle, name.encode(), ctypes.byref(v))
    if err:
        print(f"  READ {name}: error {err} ({err_str(err)})")
        return None
    return v.value


# --- main ---
handle = ctypes.c_int(0)
err = ljm.LJM_Open(LJM_dtT7, LJM_ctANY, b"ANY", ctypes.byref(handle))
if err:
    print(f"LJM_Open failed: {err} ({err_str(err)})", file=sys.stderr)
    sys.exit(1)
print(f"T7 opened, handle={handle.value}\n")

print("=== reset DIO0 / FIO0 to plain digital output ===")
# Disable any extended-feature configuration on DIO0
write_name(handle, "DIO0_EF_ENABLE", 0, fatal=False)
write_name(handle, "DIO0_EF_INDEX", 0, fatal=False)
# Make sure DIO_INHIBIT doesn't block bit 0
write_name(handle, "DIO_INHIBIT", 0, fatal=False)
# Set direction: bit 0 = 1 (output)
write_name(handle, "DIO_DIRECTION", 1, fatal=False)
# Init state low
write_name(handle, "FIO0", 0)

print("\n=== readback ===")
print(f"  DIO_DIRECTION = {read_name(handle, 'DIO_DIRECTION')} (expect 1)")
print(f"  FIO0          = {read_name(handle, 'FIO0')} (expect 0)")
print(f"  DIO0          = {read_name(handle, 'DIO0')} (expect 0)")

print("\n=== single test pulse: HIGH for 1 second, then LOW ===")
print("(scope FIO0 + GND now — should see ~3.3V for 1s)")
write_name(handle, "FIO0", 1)
print(f"  readback after HIGH: FIO0 = {read_name(handle, 'FIO0')} (expect 1)")
time.sleep(1.0)
write_name(handle, "FIO0", 0)
print(f"  readback after LOW: FIO0  = {read_name(handle, 'FIO0')} (expect 0)")

print("\n=== now pulsing at 10 Hz, 10 ms wide. Ctrl-C to stop ===")
try:
    while True:
        ljm.LJM_eWriteName(handle, b"FIO0", 1.0)
        time.sleep(0.010)
        ljm.LJM_eWriteName(handle, b"FIO0", 0.0)
        time.sleep(0.090)
except KeyboardInterrupt:
    print("\nstopping")

ljm.LJM_eWriteName(handle, b"FIO0", 0.0)
ljm.LJM_Close(handle)
print("done")
