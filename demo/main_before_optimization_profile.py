"""
profile_harness.py
==================
Full profiling suite for the recycling-sorter pipeline (main.py).

Tools used
----------
- cProfile   : per-function call counts, cumulative / total time
- pstats     : sorts & prints cProfile results; writes .prof binary
- snakeviz   : optional flame-graph viewer (pip install snakeviz → snakeviz out.prof)
- CodeCarbon : CO₂ / kWh emissions estimate per logical block
- pyJoules   : hardware-level energy sampling (RAPL / CPU perf counters)
- timeit     : tight-loop microbenchmarks of pure-Python / NumPy hot paths

Hardware stubs
--------------
All RPi / camera / sensor imports are replaced with lightweight fakes so
the harness runs on any x86/ARM Linux machine without real peripherals.
"""

# ---------------------------------------------------------------------------
# 0.  Output — redirect all print() and pstats output to a report file
# ---------------------------------------------------------------------------
import sys, os, datetime

REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"profile_report_notopimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

class _Tee:
    """Write to both the report file and the original stdout simultaneously."""
    def __init__(self, filepath, original):
        self._file     = open(filepath, "w", buffering=1)
        self._original = original
    def write(self, msg):
        self._file.write(msg)
        self._original.write(msg)
    def flush(self):
        self._file.flush()
        self._original.flush()
    def close(self):
        self._file.close()

_tee = _Tee(REPORT_PATH, sys.stdout)
sys.stdout = _tee

print(f"Profile report — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output file    : {REPORT_PATH}\n")

# ---------------------------------------------------------------------------
# 0b.  Dependency guard — install missing packages automatically
# ---------------------------------------------------------------------------
import subprocess

REQUIRED = [
    "numpy", "opencv-python-headless", "codecarbon",
    "pyJoules", "paho-mqtt",
]
for pkg in REQUIRED:
    try:
        __import__(pkg.replace("-", "_").split(".")[0])
    except ImportError:
        print(f"[SETUP] Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg]
        )

# ---------------------------------------------------------------------------
# 1.  Hardware / RPi stubs  (must be in sys.modules BEFORE importing main.py)
# ---------------------------------------------------------------------------
import sys, types, math, time, random

def _stub_module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m

# --- RPi.GPIO ----------------------------------------------------------------
gpio_mod = _stub_module("RPi")
gpio_mod.GPIO = _stub_module("RPi.GPIO",
    BCM=11, OUT=0, IN=1, BOARD=10,
    setmode=lambda *a,**k: None,
    setwarnings=lambda *a,**k: None,
    setup=lambda *a,**k: None,
    cleanup=lambda *a,**k: None,
)
class _FakePWM:
    def __init__(self, *a): pass
    def start(self, *a): pass
    def stop(self, *a): pass
    def ChangeDutyCycle(self, *a): pass
gpio_mod.GPIO.PWM = _FakePWM
sys.modules["RPi.GPIO"] = gpio_mod.GPIO

# --- gpiozero ----------------------------------------------------------------
class _FakeDID:
    value = 1      # non-metal by default
    def __init__(self, *a, **k): pass

class _FakeButton:
    is_pressed = False
    def __init__(self, *a, **k): pass

_stub_module("gpiozero", DigitalInputDevice=_FakeDID, Button=_FakeButton)

# --- hx711 -------------------------------------------------------------------
class _FakeHX711:
    def __init__(self, *a, **k): pass
    def reset(self): pass
    def get_raw_data(self, n=20):
        base = 492.22 * 150 + 0     # ≈ 150 g object
        return [base + random.gauss(0, 200) for _ in range(n)]

_stub_module("hx711", HX711=_FakeHX711)

# --- as7343 ------------------------------------------------------------------
class _FakeAS7343:
    def set_gain(self, *a): pass
    def set_integration_time(self, *a): pass
    def set_illumination_led(self, *a): pass
    def bank_select(self, *a): pass
    def get_data(self):
        channels = {
            "F1": random.uniform(0.05, 0.15),
            "F2": random.uniform(0.08, 0.18),
            "F3": random.uniform(0.10, 0.20),
            "F4": random.uniform(0.12, 0.22),
            "F5": random.uniform(0.07, 0.17),
            "F6": random.uniform(0.05, 0.15),
            "NIR": random.uniform(0.03, 0.10),
        }
        return [channels]

_stub_module("as7343", AS7343=_FakeAS7343)

# --- paho.mqtt ---------------------------------------------------------------
class _FakeMQTT:
    CallbackAPIVersion = type("V", (), {"VERSION2": 2})()
    class Client:
        def __init__(self, *a, **k): pass
        def connect(self, *a, **k): pass
        def loop_start(self): pass
        def loop_stop(self): pass
        def disconnect(self): pass
        def publish(self, topic, payload):
            class _Res:
                def wait_for_publish(self, timeout=2): pass
            return _Res()
        on_connect = on_disconnect = None

_stub_module("paho", mqtt=_stub_module("paho.mqtt", client=_FakeMQTT))
_stub_module("paho.mqtt", client=_FakeMQTT)
_stub_module("paho.mqtt.client", Client=_FakeMQTT.Client,
             CallbackAPIVersion=_FakeMQTT.CallbackAPIVersion)

# --- cv2 VideoCapture stub ---------------------------------------------------
import numpy as np
import cv2 as _cv2_real

class _FakeCap:
    def __init__(self, *a, **k):
        self._open = True
    def set(self, *a): pass
    def isOpened(self): return True
    def read(self):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        return True, frame
    def release(self): pass

_real_VideoCapture = _cv2_real.VideoCapture
def _patched_VideoCapture(index, *a, **k):
    return _FakeCap()
_cv2_real.VideoCapture = _patched_VideoCapture

# --- subprocess stub (v4l2-ctl calls) ----------------------------------------
import subprocess as _real_subprocess
_real_run = _real_subprocess.run
def _mocked_run(cmd, *a, **k):
    if isinstance(cmd, list) and "v4l2-ctl" in cmd[0]:
        return type("R", (), {"returncode": 0})()
    return _real_run(cmd, *a, **k)
_real_subprocess.run = _mocked_run

# --- Helper stubs (mobileNet_helpers / pick_model / monitoring) --------------
def _fake_run_inference(**k):
    n = 7
    probs = np.random.dirichlet(np.ones(n)).tolist()
    best  = int(np.argmax(probs))
    return best, probs[best], 3.5, probs

def _fake_load_model(*a, **k):
    return object(), [], [], "/fake/model"

def _fake_return_latest_version_path(name):
    return "/fake/model"

_stub_module("mobileNet_helpers",
             run_inference=_fake_run_inference,
             load_model=_fake_load_model)
_stub_module("pick_model",
             return_latest_version_path=_fake_return_latest_version_path)
_stub_module("monitoring",
             track_ram=lambda: 0.0,
             track_temp=lambda: 0.0,
             track_power=lambda: 0.0,
             returns_latest_file_number=lambda: 0)

# --- Fake labels.txt on disk -------------------------------------------------
import os, tempfile
_fake_model_dir = tempfile.mkdtemp()
with open(os.path.join(_fake_model_dir, "labels.txt"), "w") as f:
    f.write("\n".join([
        "carton_tetrapack","disposable_cup","glass_bottle",
        "glass_cup","metal_bottle","metal_can","plastic_bottle",
    ]))

# Patch the path returned by return_latest_version_path to our temp dir
import mobileNet_helpers as _mnh  # noqa
import pick_model as _pm
_pm.return_latest_version_path = lambda name: _fake_model_dir

# ---------------------------------------------------------------------------
# 2.  Import the real pipeline (hardware ops are now all faked)
# ---------------------------------------------------------------------------
print("[HARNESS] Importing pipeline…")
os.chdir(tempfile.mkdtemp())          # avoid cwd artefacts from main.py

# Prevent main_pipeline() from running at import time
import unittest.mock as _mock
with _mock.patch("builtins.__import__", side_effect=__import__):
    # Write a stub calibration.json so the spectrometer skips interactive setup
    import json
    _cal = {
        "plastic": {
            "mean": {"F1": 0.14, "F2": 0.17, "F3": 0.19, "F4": 0.21,
                     "F5": 0.15, "F6": 0.10, "NIR": 0.04},
            "std":  {"F1": 0.01, "F2": 0.01, "F3": 0.01, "F4": 0.01,
                     "F5": 0.01, "F6": 0.01, "NIR": 0.005},
            "n": 10
        },
        "glass": {
            "mean": {"F1": 0.08, "F2": 0.12, "F3": 0.18, "F4": 0.22,
                     "F5": 0.20, "F6": 0.14, "NIR": 0.06},
            "std":  {"F1": 0.01, "F2": 0.01, "F3": 0.01, "F4": 0.01,
                     "F5": 0.01, "F6": 0.01, "NIR": 0.005},
            "n": 10
        }
    }
    with open("calibration.json", "w") as _f:
        json.dump(_cal, _f)

# Re-point pick_model before main.py auto-loads the model
import pick_model as pm
pm.return_latest_version_path = lambda name: _fake_model_dir

import importlib, io, cProfile, pstats, timeit, textwrap
import traceback

# Minimal os.chdir guard — main.py does os.chdir(__file__ dir)
# We already changed to tmpdir above; patch __file__ resolution
with _mock.patch("os.path.abspath", return_value=os.getcwd() + "/main.py"):
    with _mock.patch("os.path.dirname", return_value=os.getcwd()):
        import main as pipeline   # noqa

print("[HARNESS] Import OK — all hardware stubs active.\n")

# Seed the profiles so spec functions have data
pipeline.plastic_profile = _cal["plastic"]
pipeline.glass_profile   = _cal["glass"]

# ---------------------------------------------------------------------------
# 3.  Build synthetic test data
# ---------------------------------------------------------------------------
FRAME_320x240  = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
FRAME_640x480  = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Synthetic spectral fingerprints
SPEC_PLASTIC_FP = {k: v for k, v in _cal["plastic"]["mean"].items()}
SPEC_GLASS_FP   = {k: v for k, v in _cal["glass"]["mean"].items()}

PLASTIC_PROFILE = _cal["plastic"]
GLASS_PROFILE   = _cal["glass"]

RAW_DATA_20 = [492.22 * 150 + random.gauss(0, 200) for _ in range(20)]

VISION_LABELS = [
    "glass_bottle", "plastic_bottle", "metal_can",
    "metal_bottle", "disposable_cup", "carton_tetrapack",
]

# ---------------------------------------------------------------------------
# 4.  Functions under test  (mapped name → callable + setup_globals)
# ---------------------------------------------------------------------------
BENCHMARKS = {
    # ── Color correction ──────────────────────────────────────────────────
    "correct_frame": dict(
        fn  = lambda: pipeline.correct_frame(FRAME_320x240),
        n   = 500,
        doc = "Per-frame white-balance LUT (BGR channels, 320×240)",
    ),

    # ── IQR weight cleaning ───────────────────────────────────────────────
    "_iqr_clean": dict(
        fn  = lambda: pipeline._iqr_clean(RAW_DATA_20),
        n   = 5000,
        doc = "IQR outlier removal on 20-sample weight read",
    ),

    # ── Servo math ────────────────────────────────────────────────────────
    "shortest_angle_diff": dict(
        fn  = lambda: pipeline.shortest_angle_diff(85.0, 220.0),
        n   = 50000,
        doc = "Single shortest-arc angle diff calculation",
    ),
    "decelerated_speed": dict(
        fn  = lambda: pipeline.decelerated_speed(pipeline.SPEED_FWD, 45.0),
        n   = 50000,
        doc = "Trapezoidal speed ramp calculation",
    ),

    # ── ArUco preprocessing ───────────────────────────────────────────────
    "_aruco_preprocess": dict(
        fn  = lambda: pipeline._aruco_preprocess(FRAME_320x240),
        n   = 200,
        doc = "2× upscale → grayscale → CLAHE (full image pipeline)",
    ),
    "detect_aruco_angle": dict(
        fn  = lambda: pipeline.detect_aruco_angle(FRAME_320x240),
        n   = 100,
        doc = "Full ArUco marker detection + angle computation",
    ),

    # ── Spectral metrics ──────────────────────────────────────────────────
    "_spec_sam": dict(
        fn  = lambda: pipeline._spec_sam(SPEC_PLASTIC_FP, PLASTIC_PROFILE["mean"]),
        n   = 10000,
        doc = "Spectral Angle Mapper (SAM) — plastic vs profile",
    ),
    "_spec_chi_squared": dict(
        fn  = lambda: pipeline._spec_chi_squared(SPEC_PLASTIC_FP, PLASTIC_PROFILE["mean"]),
        n   = 10000,
        doc = "Chi-squared distance between fingerprint and profile mean",
    ),
    "_spec_euclidean": dict(
        fn  = lambda: pipeline._spec_euclidean(SPEC_PLASTIC_FP, SPEC_GLASS_FP),
        n   = 10000,
        doc = "Euclidean distance between two spectral fingerprints",
    ),
    "_spec_confidence_sam": dict(
        fn  = lambda: pipeline._spec_confidence_sam(
            5.2, 18.7, 0.0031, 0.0098,
            PLASTIC_PROFILE["mean"], GLASS_PROFILE["mean"],
        ),
        n   = 5000,
        doc = "SAM+Chi² confidence score computation",
    ),
    "_spec_reject_outliers": dict(
        fn  = lambda: pipeline._spec_reject_outliers(
            [SPEC_PLASTIC_FP.copy() for _ in range(12)]
        ),
        n   = 500,
        doc = "Outlier rejection across 12 spectral fingerprints",
    ),
    "_spec_average": dict(
        fn  = lambda: pipeline._spec_average(
            [SPEC_PLASTIC_FP.copy() for _ in range(6)]
        ),
        n   = 2000,
        doc = "Channel-wise average across 6 fingerprints",
    ),

    # ── Fusion ────────────────────────────────────────────────────────────
    "fuse_results_glass": dict(
        fn  = lambda: pipeline.fuse_results(
            "glass_bottle", pipeline.Material.GLASS, 82.5, 175.0
        ),
        n   = 1000,
        doc = "Sensor-fusion decision — vision+spec agree on GLASS",
    ),
    "fuse_results_metal_heavy": dict(
        fn  = lambda: pipeline.fuse_results(
            "metal_can", pipeline.Material.METAL, 100.0, 620.0
        ),
        n   = 1000,
        doc = "Sensor-fusion decision — METAL over weight limit → GENERAL_WASTE",
    ),
    "fuse_results_disagree": dict(
        fn  = lambda: pipeline.fuse_results(
            "glass_bottle", pipeline.Material.PLASTIC, 45.0, 80.0
        ),
        n   = 1000,
        doc = "Sensor-fusion decision — vision/spec disagree, weight tiebreaker",
    ),
}

# ---------------------------------------------------------------------------
# 5a.  timeit microbenchmarks
# ---------------------------------------------------------------------------
print("=" * 72)
print("  SECTION A — timeit MICROBENCHMARKS")
print("=" * 72)
print(f"{'Function':<32}  {'N':>6}  {'Total(s)':>10}  {'Per-call(µs)':>13}  Note")
print("-" * 72)

timeit_results = {}
for name, cfg in BENCHMARKS.items():
    fn  = cfg["fn"]
    n   = cfg["n"]
    doc = cfg["doc"]
    try:
        total_s = timeit.timeit(fn, number=n)
        per_us  = (total_s / n) * 1e6
        timeit_results[name] = {"total_s": total_s, "per_us": per_us, "n": n}
        flag = "  ⚠  SLOW" if per_us > 5000 else ""
        print(f"  {name:<30}  {n:>6}  {total_s:>10.4f}  {per_us:>11.2f} µs{flag}")
    except Exception as exc:
        print(f"  {name:<30}  ERROR: {exc}")
        timeit_results[name] = {"error": str(exc)}

print()

# ---------------------------------------------------------------------------
# 5b.  cProfile — full pipeline simulation
# ---------------------------------------------------------------------------
print("=" * 72)
print("  SECTION B — cProfile CALL GRAPH  (50 simulated classifications)")
print("=" * 72)

def _simulate_pipeline(n_cycles=50):
    """
    Replays the core classification loop without blocking I/O.
    Each iteration mirrors what main_pipeline() does per object.
    """
    for i in range(n_cycles):
        # Weight read
        vals = RAW_DATA_20[:]
        pipeline._iqr_clean(vals)

        # Color correction
        corrected = pipeline.correct_frame(FRAME_320x240)

        # ArUco
        pipeline._aruco_preprocess(FRAME_320x240)
        pipeline.detect_aruco_angle(FRAME_320x240)

        # Servo math
        pipeline.shortest_angle_diff(85.0, 220.0)
        pipeline.decelerated_speed(pipeline.SPEED_FWD, 45.0)

        # Spectrometer path  (non-metal branch)
        fps = [SPEC_PLASTIC_FP.copy() for _ in range(6)]
        clean, _ = pipeline._spec_reject_outliers(fps)
        scan = pipeline._spec_average(clean)
        sam_p = pipeline._spec_sam(scan, PLASTIC_PROFILE["mean"])
        sam_g = pipeline._spec_sam(scan, GLASS_PROFILE["mean"])
        chi_p = pipeline._spec_chi_squared(scan, PLASTIC_PROFILE["mean"])
        chi_g = pipeline._spec_chi_squared(scan, GLASS_PROFILE["mean"])
        pipeline._spec_confidence_sam(
            sam_p, sam_g, chi_p, chi_g,
            PLASTIC_PROFILE["mean"], GLASS_PROFILE["mean"],
        )

        # Sensor fusion
        vl = random.choice(VISION_LABELS)
        mat = random.choice([pipeline.Material.PLASTIC, pipeline.Material.GLASS])
        pipeline.fuse_results(vl, mat, 75.0, 130.0)

_prof = cProfile.Profile()
_prof.enable()
_simulate_pipeline(50)
_prof.disable()

_stream = io.StringIO()
_ps = pstats.Stats(_prof, stream=_stream)
_ps.sort_stats("cumulative")
_ps.print_stats(30)
print(_stream.getvalue())

_prof_path = "sorter_pipeline.prof"
_prof.dump_stats(_prof_path)
print(f"[cProfile] Binary .prof written → {_prof_path}")
print(f"           View interactively:   snakeviz {_prof_path}\n")

# ---------------------------------------------------------------------------
# 5c.  CodeCarbon — emissions per logical block
# ---------------------------------------------------------------------------
print("=" * 72)
print("  SECTION C — CodeCarbon EMISSIONS TRACKER")
print("=" * 72)

try:
    from codecarbon import EmissionsTracker

    _blocks = {
        "spectral_batch_100":    lambda: [_simulate_pipeline(2) for _ in range(5)],
        "aruco_detection_500":   lambda: [pipeline.detect_aruco_angle(FRAME_320x240)
                                          for _ in range(500)],
        "color_correction_1000": lambda: [pipeline.correct_frame(FRAME_320x240)
                                          for _ in range(1000)],
        "fusion_1000":           lambda: [pipeline.fuse_results(
                                              random.choice(VISION_LABELS),
                                              pipeline.Material.GLASS, 75.0, 130.0)
                                          for _ in range(1000)],
    }

    carbon_results = {}
    for block_name, block_fn in _blocks.items():
        tracker = EmissionsTracker(
            project_name=f"sorter_{block_name}",
            output_dir=".",
            log_level="error",
            save_to_file=False,
            tracking_mode="process",
        )
        tracker.start()
        t0 = time.perf_counter()
        block_fn()
        elapsed = time.perf_counter() - t0
        emissions = tracker.stop()   # kg CO₂-eq

        kwh      = getattr(tracker, "_total_energy", None)
        kwh_val  = float(kwh.kWh) if kwh is not None else float("nan")

        carbon_results[block_name] = {
            "elapsed_s":   elapsed,
            "co2_kg":      emissions,
            "energy_kWh":  kwh_val,
        }
        print(f"  {block_name:<30}  "
              f"time={elapsed:.3f}s  "
              f"CO₂={emissions*1e6:.3f} µg  "
              f"energy={kwh_val*1e6:.3f} µWh")

except ImportError:
    print("  [SKIP] codecarbon not installed — pip install codecarbon")
    carbon_results = {}
except Exception as e:
    print(f"  [WARN] CodeCarbon error: {e}")
    carbon_results = {}

print()

# ---------------------------------------------------------------------------
# 5d.  pyJoules — hardware energy measurement
# ---------------------------------------------------------------------------
print("=" * 72)
print("  SECTION D — pyJoules ENERGY MEASUREMENT  (RAPL / CPU perf)")
print("=" * 72)

try:
    from pyJoules.energy_meter import EnergyMeter
    from pyJoules.device.rapl_device import RaplPackageDomain
    from pyJoules.handler.csv_handler import CSVHandler

    _domains = [RaplPackageDomain(0)]
    _handler = CSVHandler("pyjoules_results.csv")
    _meter   = EnergyMeter(_domains)

    joules_results = {}
    _joules_blocks = {
        "aruco_100":     lambda: [pipeline.detect_aruco_angle(FRAME_320x240)
                                  for _ in range(100)],
        "spec_math_500": lambda: [
            pipeline._spec_sam(SPEC_PLASTIC_FP, PLASTIC_PROFILE["mean"])
            for _ in range(500)
        ],
        "correction_500": lambda: [pipeline.correct_frame(FRAME_320x240)
                                   for _ in range(500)],
    }

    for tag, fn in _joules_blocks.items():
        _meter.start(tag=tag)
        fn()
        _meter.stop()
        trace = _meter.get_trace()
        if trace:
            sample = trace[-1]
            energy_J = sum(sample.energy[d] for d in sample.energy) / 1e6
            joules_results[tag] = energy_J
            print(f"  {tag:<28}  energy ≈ {energy_J*1000:.3f} mJ")
        _handler.process(trace)

    _handler.save_data()
    print("  [pyJoules] Per-sample CSV → pyjoules_results.csv")

except Exception as e:
    # pyJoules requires real RAPL MSR access — falls back gracefully on VMs
    print(f"  [INFO] pyJoules RAPL unavailable ({type(e).__name__}: {e})")
    print("  Running software-timer fallback instead...\n")

    joules_results = {}
    _sw_blocks = {
        "aruco_100":      lambda: [pipeline.detect_aruco_angle(FRAME_320x240)
                                   for _ in range(100)],
        "spec_math_500":  lambda: [
            pipeline._spec_sam(SPEC_PLASTIC_FP, PLASTIC_PROFILE["mean"])
            for _ in range(500)
        ],
        "correction_500": lambda: [pipeline.correct_frame(FRAME_320x240)
                                   for _ in range(500)],
        "fusion_1000":    lambda: [pipeline.fuse_results(
                                       "glass_bottle",
                                       pipeline.Material.GLASS, 75.0, 130.0)
                                   for _ in range(1000)],
    }

    print(f"  {'Block':<28}  {'Wall time (s)':>14}  {'µs/call':>10}")
    print("  " + "-" * 56)
    for tag, fn in _sw_blocks.items():
        n_calls = int(tag.rsplit("_", 1)[-1])
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        joules_results[tag] = elapsed
        print(f"  {tag:<28}  {elapsed:>14.4f}  {elapsed/n_calls*1e6:>9.2f} µs")

print()

# ---------------------------------------------------------------------------
# 6.  Consolidated summary report
# ---------------------------------------------------------------------------
print("=" * 72)
print("  SUMMARY — TOP 10 SLOWEST FUNCTIONS (timeit, µs/call)")
print("=" * 72)
ranked = sorted(
    [(k, v["per_us"]) for k, v in timeit_results.items() if "per_us" in v],
    key=lambda x: x[1], reverse=True,
)
for rank, (name, us) in enumerate(ranked[:10], 1):
    bar = "█" * min(40, max(1, int(us / max(ranked[0][1], 1) * 40)))
    print(f"  {rank:2d}. {name:<34} {us:>9.1f} µs  {bar}")

print()
print("=" * 72)
print("  OPTIMISATION HINTS")
print("=" * 72)
hints = []
for name, us in ranked:
    if "aruco" in name.lower() and us > 1000:
        hints.append(f"• {name}: {us:.0f} µs — consider reducing DETECT_EVERY_N "
                     f"or downscaling before CLAHE instead of after.")
    if "correct_frame" in name and us > 500:
        hints.append(f"• {name}: {us:.0f} µs — LUT already fast; "
                     f"pre-split BGR channels once at capture time.")
    if "spec" in name and us > 100:
        hints.append(f"• {name}: {us:.0f} µs — dict lookups dominate; "
                     f"convert calibration profile means to np.ndarray at load time.")
    if "fuse" in name and us > 200:
        hints.append(f"• {name}: {us:.0f} µs — pure logic; "
                     f"acceptable. Verify beam_sensor.is_pressed isn't adding GPIO latency.")

for h in hints or ["  No obvious hotspots above thresholds."]:
    print(f"  {h}")

print()
print("[HARNESS] Done.")
print(f"          cProfile binary : sorter_pipeline.prof")
print(f"          Snakeviz viewer : snakeviz sorter_pipeline.prof")
print(f"          Full report     : {REPORT_PATH}")

# Restore stdout and close the file
sys.stdout = _tee._original
_tee.close()
print(f"\n[HARNESS] Report saved → {REPORT_PATH}")