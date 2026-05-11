"""
Level 3 test — on-device Raspberry Pi integration test.

Checks every hardware dependency before a full deployment:
  1. Microphone — detects the device and records a short clip
  2. DSP pipeline — runs the recording through the full feature pipeline
  3. UART — opens /dev/serial0, sends a sync packet, checks for ACK

Run on the Raspberry Pi:
    python test_hardware.py

All checks print PASS / FAIL.  Exit code 0 = all passed.
"""

import sys
import os
import time
import struct
import threading

sys.path.insert(0, os.path.dirname(__file__))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []


def check(label, condition, detail=""):
    tag = PASS if condition else FAIL
    line = f"  [{tag}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append(condition)
    return condition


# ---------------------------------------------------------------------------
# 1. Microphone
# ---------------------------------------------------------------------------

def test_microphone():
    print("\n--- Microphone ---")
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError as e:
        check("sounddevice importable", False, str(e))
        return None

    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    check("at least one input device found", len(input_devices) > 0,
          f"{len(input_devices)} device(s)")

    if not input_devices:
        return None

    default = sd.query_devices(kind='input')
    print(f"  Using: {default['name']}")

    DURATION = 2
    SR = 44100
    try:
        audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1,
                       dtype='int16')
        sd.wait()
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        check("recording returned data",  audio.size > 0,   f"{audio.size} samples")
        check("audio is not silent",      rms > 10,         f"RMS={rms:.1f}")
        return audio.flatten(), SR
    except Exception as e:
        check("recording succeeded", False, str(e))
        return None


# ---------------------------------------------------------------------------
# 2. DSP pipeline
# ---------------------------------------------------------------------------

def test_pipeline(audio, sample_rate):
    print("\n--- DSP Pipeline ---")
    try:
        from rpi_test_final import (
            Datum, FeatureEngineeringPipeline, Decimation,
            FrequencyPeakFinder, RPM, DerivedDataKey,
            DECIMATED_RATE, EVENTS_PER_CYCLE,
        )
    except Exception as e:
        check("rpi_test_final importable", False, str(e))
        return

    import numpy as np

    pipeline = FeatureEngineeringPipeline()
    pipeline.add_block(Decimation(target_rate=DECIMATED_RATE))
    pipeline.add_block(FrequencyPeakFinder())
    pipeline.add_output_block(RPM(events_per_crankshaft_cycle=EVENTS_PER_CYCLE))

    try:
        datum = Datum(audio_array=audio.astype(np.float32), sample_rate=sample_rate)
        datum = pipeline.run(datum)

        rpm    = datum.get_derived_data(DerivedDataKey.RPM)
        status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS)
        freq   = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK)

        check("pipeline ran without error",  True)
        check("RPM is a finite number",      np.isfinite(rpm),    f"{rpm:.1f}")
        check("freq is a finite number",     np.isfinite(freq),   f"{freq:.2f} Hz")
        check("engine status is 0 or 1",     status in (0, 1),    str(status))

        print(f"  Result: RPM={rpm:.1f}  status={status}  freq={freq:.2f} Hz")

    except Exception as e:
        check("pipeline ran without error", False, str(e))


# ---------------------------------------------------------------------------
# 3. UART
# ---------------------------------------------------------------------------

def test_uart():
    print("\n--- UART ---")
    try:
        import serial
    except ImportError as e:
        check("pyserial importable", False, str(e))
        return

    from rpi_test_final import UART_PORT, UART_BAUDRATE

    # Port exists
    port_exists = os.path.exists(UART_PORT)
    check(f"{UART_PORT} exists", port_exists)
    if not port_exists:
        return

    # Can open port
    try:
        ser = serial.Serial(UART_PORT, UART_BAUDRATE, timeout=2,
                            write_timeout=2.0, exclusive=True)
        time.sleep(0.5)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        check("serial port opened", True)
    except Exception as e:
        check("serial port opened", False, str(e))
        return

    # Send sync packet and check no exception
    try:
        ts = int(time.time() * 1000)
        payload = struct.pack('<Q', ts)
        crc = _crc8(payload)
        packet = bytearray([0xAB, len(payload)]) + payload + bytearray([crc, 0x55])
        written = ser.write(packet)
        ser.flush()
        check("sync packet written", written == len(packet),
              f"{written}/{len(packet)} bytes")
    except Exception as e:
        check("sync packet written", False, str(e))

    # Wait briefly for any response (optional — no ACK expected for sync)
    time.sleep(0.5)
    waiting = ser.in_waiting
    print(f"  Bytes in RX buffer after sync: {waiting}")

    # Send a DATA packet and wait for ACK
    try:
        seq = 0xBEEF & 0xFFFF
        rpm_val = 1234.5
        payload = struct.pack('<H', seq)       # seq
        payload += struct.pack('<Q', ts)       # timestamp
        payload += struct.pack('<f', rpm_val)  # RPM
        payload += struct.pack('<B', 1)        # engine_status
        payload += struct.pack('<f', 41.15)    # peak_freq
        crc = _crc8(payload)
        packet = bytearray([0xAA, len(payload)]) + payload + bytearray([crc, 0x55])

        ser.reset_input_buffer()
        ser.write(packet)
        ser.flush()

        ser.timeout = 3.0
        response = ser.read(3)

        got_ack = (len(response) == 3 and
                   response[0] == 0x06 and
                   struct.unpack('<H', response[1:3])[0] == seq)
        check("ESP32 ACK received", got_ack,
              f"response={response.hex() if response else 'empty'}")

    except Exception as e:
        check("DATA packet ACK check", False, str(e))

    ser.close()


def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if (crc & 0x80) else (crc << 1)
            crc &= 0xFF
    return crc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("Hardware integration test — run on Raspberry Pi")
    print("=" * 50)

    mic_result = test_microphone()

    if mic_result is not None:
        audio, sr = mic_result
        test_pipeline(audio, sr)
    else:
        print("\n--- DSP Pipeline ---")
        print("  [SKIP] No audio recorded — skipping pipeline test")

    test_uart()

    print("\n" + "=" * 50)
    passed = sum(results)
    total  = len(results)
    if all(results):
        print(f"\033[92mAll {total} checks passed.\033[0m")
        sys.exit(0)
    else:
        print(f"\033[91m{total - passed}/{total} checks FAILED.\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
