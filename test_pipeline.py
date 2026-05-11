"""
Level 1 test — DSP pipeline only, no hardware required.

Feeds simulation WAV files through the feature engineering pipeline and
checks that RPM, engine status, and peak frequency are within sane ranges.

Run from BANJI_WORK_CODE/:
    python test_pipeline.py
"""

import sys
import os
import scipy.io.wavfile as wav
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rpi_test_final import (
    Datum,
    FeatureEngineeringPipeline,
    Decimation,
    FrequencyPeakFinder,
    RPM,
    DerivedDataKey,
    DECIMATED_RATE,
    EVENTS_PER_CYCLE,
)

SIMULATION_DIR = os.path.join(os.path.dirname(__file__), "simulation_data", "audios")

# (file, expected_status, rpm_min, rpm_max)
TEST_CASES = [
    ("1000_rpm.wav", 1,  800, 1200),  # 2nd harmonic at 66.96 Hz → ~1004 RPM with EVENTS_PER_CYCLE=4
    ("volo1.wav",    1,    0, 5000),
]

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def build_pipeline():
    pipeline = FeatureEngineeringPipeline()
    pipeline.add_block(Decimation(target_rate=DECIMATED_RATE))
    pipeline.add_block(FrequencyPeakFinder())
    pipeline.add_output_block(RPM(events_per_crankshaft_cycle=EVENTS_PER_CYCLE))
    return pipeline


def run_on_file(pipeline, path):
    sr, data = wav.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    datum = Datum(audio_array=data, sample_rate=sr)
    return pipeline.run(datum)


def check(label, condition, got):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}  (got: {got})")
    return condition


def test_file(pipeline, filename, expected_status, rpm_min, rpm_max):
    path = os.path.join(SIMULATION_DIR, filename)
    if not os.path.exists(path):
        print(f"\n{filename}: SKIP — file not found at {path}")
        return True  # not a failure

    print(f"\n{filename}")
    datum = run_on_file(pipeline, path)

    rpm    = datum.get_derived_data(DerivedDataKey.RPM)
    status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS)
    freq   = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK)

    print(f"  RPM={rpm:.1f}  status={status}  peak_freq={freq:.2f} Hz")

    results = [
        check("rpm is a finite number",      np.isfinite(rpm),                   rpm),
        check("freq is a finite number",     np.isfinite(freq),                  freq),
        check("engine status is 0 or 1",     status in (0, 1),                   status),
        check(f"engine status == {expected_status}",
                                             status == expected_status,           status),
        check(f"rpm in [{rpm_min}, {rpm_max}]",
                                             rpm_min <= rpm <= rpm_max,           rpm),
    ]
    return all(results)


def test_silent_audio(pipeline):
    """Silent signal must yield rpm=0 and status=0."""
    print("\nsilent signal (all zeros)")
    silent = np.zeros(44100 * 5, dtype=np.float32)
    datum = Datum(audio_array=silent, sample_rate=44100)
    datum = pipeline.run(datum)

    rpm    = datum.get_derived_data(DerivedDataKey.RPM)
    status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS)

    results = [
        check("rpm == 0.0",      rpm == 0.0,    rpm),
        check("status == 0",     status == 0,   status),
    ]
    return all(results)


def test_known_tone(pipeline):
    """
    Synthesise a pure 83.3 Hz tone — corresponds to 1250 RPM when treated as
    the 2nd harmonic (EVENTS_PER_CYCLE=4): RPM = 83.3 * 60 / 4 = 1249.5.
    """
    print("\npure 83.3 Hz tone (expected ~1250 RPM)")
    sr = 44100
    duration = 5
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    tone = (np.sin(2 * np.pi * 83.3 * t) * 32767).astype(np.float32)

    datum = Datum(audio_array=tone, sample_rate=sr)
    datum = pipeline.run(datum)

    rpm    = datum.get_derived_data(DerivedDataKey.RPM)
    status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS)

    results = [
        check("engine status == 1",        status == 1,           status),
        check("rpm in [1000, 1500]",        1000 <= rpm <= 1500,   rpm),
    ]
    return all(results)


def main():
    print("=" * 50)
    print("Pipeline test — no hardware required")
    print("=" * 50)

    pipeline = build_pipeline()
    all_passed = True

    # Simulation WAV files
    for filename, exp_status, rpm_min, rpm_max in TEST_CASES:
        passed = test_file(pipeline, filename, exp_status, rpm_min, rpm_max)
        all_passed &= passed

    # Synthetic signal tests
    all_passed &= test_silent_audio(pipeline)
    all_passed &= test_known_tone(pipeline)

    print("\n" + "=" * 50)
    if all_passed:
        print("\033[92mAll tests passed.\033[0m")
        sys.exit(0)
    else:
        print("\033[91mSome tests FAILED.\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
