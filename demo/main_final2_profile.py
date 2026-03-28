"""
main_final2_profile.py
======================
Full profiling suite for main_final2.py (optimised recycling-sorter pipeline).

Tools used
----------
- perf stat   : hardware CPU counters (cycles, instructions, cache-misses, branches)
                Attach externally:  sudo perf stat -p <PID> -e cycles,instructions,cache-misses,branch-misses
- cProfile    : per-function call counts & cumulative / total time
- pstats      : sorts & prints cProfile results; writes .prof binary
- CodeCarbon  : CO₂ / kWh emissions estimate per logical block
- pyJoules    : hardware-level energy sampling (RAPL / CPU perf counters)
- timeit      : tight-loop microbenchmarks of pure-Python / NumPy hot paths

Hardware stubs
--------------
All RPi / camera / sensor imports are replaced with lightweight fakes so the
harness runs without real peripherals attached.
"""

# ---------------------------------------------------------------------------
# 0.  Output — redirect all print() and pstats output to a timestamped file
# ---------------------------------------------------------------------------
import sys, os, datetime, time as _time_mod

REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"profile_report_final2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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
# 0a.  Script-wide timing and RSS memory tracker
# ---------------------------------------------------------------------------
_t_script_start = _time_mod.perf_counter()

try:
    import psutil as _psutil
    _SELF_PROC = _psutil.Process()
    def _rss_mb():
        return _SELF_PROC.memory_info().rss / 1024**2
except ImportError:
    _psutil = None
    try:
        import resource as _resource
        def _rss_mb():
            # Linux: ru_maxrss is in kB; macOS: bytes
            import platform
            kb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            return kb / 1024 if platform.system() == "Linux" else kb / 1024**2
    except ImportError:
        def _rss_mb():
            return float("nan")

_rss_baseline = _rss_mb()
print(f"[PROFILER] RSS baseline : {_rss_baseline:.1f} MB\n")

# ---------------------------------------------------------------------------
# 0b.  perf stat — print PID and pause so the user can attach externally
# ---------------------------------------------------------------------------
print(f"[PROFILER] PID = {os.getpid()}")
print(f"[PROFILER] To capture hardware counters, open a second terminal and run:")
print(f"[PROFILER]   sudo perf stat -p {os.getpid()} \\")
print(f"[PROFILER]       -e cycles,instructions,cache-references,cache-misses,branches,branch-misses")
print(f"[PROFILER] Press Enter here once perf stat is running (or skip with Enter)...")
sys.stdout.flush()
try:
    input()
except EOFError:
    print("[PROFILER] Non-interactive mode — skipping perf stat pause.")

# ---------------------------------------------------------------------------
# 1.  Hardware / RPi stubs  (must be in sys.modules BEFORE importing pipeline)
# ---------------------------------------------------------------------------
import types, math, time, random
import numpy as np

def _stub_module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m

# --- RPi.GPIO ----------------------------------------------------------------
gpio_mod = _stub_module("RPi")
gpio_mod.GPIO = _stub_module("RPi.GPIO",
    BCM=11, OUT=0, IN=1, BOARD=10,
    setmode=lambda *a, **k: None,
    setwarnings=lambda *a, **k: None,
    setup=lambda *a, **k: None,
    cleanup=lambda *a, **k: None,
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
    value = 1   # non-metal by default
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
        base = 120.20 * 150 + 0   # ≈ 150 g at default RATIO
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
import cv2 as _cv2_real

class _FakeCapTracking:
    """320×240 fake tracking camera (index 0)."""
    def set(self, *a): pass
    def isOpened(self): return True
    def read(self):
        return True, np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    def release(self): pass

class _FakeCapVision:
    """640×480 fake vision camera (index 2)."""
    def set(self, *a): pass
    def isOpened(self): return True
    def read(self):
        return True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    def release(self): pass

def _patched_VideoCapture(index, *a, **k):
    return _FakeCapTracking() if index == 0 else _FakeCapVision()

_cv2_real.VideoCapture = _patched_VideoCapture

# cv2.imshow / cv2.waitKey — no-ops in headless profiling
_cv2_real.imshow   = lambda *a, **k: None
_cv2_real.waitKey  = lambda *a, **k: -1
_cv2_real.namedWindow = lambda *a, **k: None
_cv2_real.destroyAllWindows = lambda *a, **k: None
_cv2_real.destroyWindow = lambda *a, **k: None

# --- subprocess stub (v4l2-ctl calls) ----------------------------------------
import subprocess as _real_subprocess
_real_run = _real_subprocess.run
def _mocked_run(cmd, *a, **k):
    if isinstance(cmd, (list, tuple)) and cmd and "v4l2-ctl" in str(cmd[0]):
        return type("R", (), {"returncode": 0})()
    return _real_run(cmd, *a, **k)
_real_subprocess.run = _mocked_run

# --- mobileNet_helpers / pick_model / monitoring stubs ----------------------
def _fake_run_inference(**k):
    n     = 7
    probs = np.random.dirichlet(np.ones(n)).tolist()
    best  = int(np.argmax(probs))
    return best, probs[best], 3.5, probs

def _fake_load_model(*a, **k):
    return object(), [], [], "/fake/model"

import tempfile as _tempfile
_fake_model_dir = _tempfile.mkdtemp()
with open(os.path.join(_fake_model_dir, "labels.txt"), "w") as _f:
    _f.write("\n".join([
        "carton_tetrapack", "disposable_cup", "glass_bottle",
        "glass_cup", "metal_bottle", "metal_can", "plastic_bottle",
    ]))

# return_latest_version_path now returns (path, version_number) — matches fix in pick_model.py
def _fake_return_latest_version_path(name):
    return _fake_model_dir, 14

_stub_module("mobileNet_helpers",
             run_inference=_fake_run_inference,
             load_model=_fake_load_model)
_stub_module("pick_model",
             return_latest_version_path=_fake_return_latest_version_path)
_stub_module("monitoring",
             track_ram=lambda: 0.0,
             track_temp=lambda: 0.0,
             track_power=lambda: 0.0,
             returns_latest_file_number=lambda x: 1)

# ---------------------------------------------------------------------------
# 2.  Import the real pipeline (hardware ops are now all faked)
# ---------------------------------------------------------------------------
import json, unittest.mock as _mock, importlib.util

print("[HARNESS] Importing main_final2 pipeline...")

# Write a stub calibration.json so spectrometer skips interactive setup
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

_pipeline_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_pipeline_dir, "calibration_profile_stub.json"), "w") as _f:
    json.dump(_cal, _f)

# Patch SPEC_CALIB_FILE so main_final2 reads our stub instead of calibration.json
# We do this by patching os.path.exists and open inside the module load
_orig_exists = os.path.exists
def _patched_exists(p):
    if "calibration.json" in str(p):
        return True   # pretend calibration exists → skips interactive setup
    return _orig_exists(p)

_orig_open = open
def _patched_open(p, *a, **k):
    if "calibration.json" in str(p) and ("r" in (a[0] if a else k.get("mode", "r"))):
        return _orig_open(os.path.join(_pipeline_dir, "calibration_profile_stub.json"), *a, **k)
    return _orig_open(p, *a, **k)

# Load main_final2 as a module (does NOT call main_pipeline — guarded by __name__)
_spec = importlib.util.spec_from_file_location(
    "main_final2",
    os.path.join(_pipeline_dir, "main_final2.py"),
)
pipeline = importlib.util.module_from_spec(_spec)
pipeline.__name__ = "main_final2"   # prevent __name__ == "__main__" guard

with _mock.patch("builtins.open", side_effect=_patched_open), \
     _mock.patch("os.path.exists", side_effect=_patched_exists):
    try:
        _spec.loader.exec_module(pipeline)
    except Exception as e:
        print(f"[HARNESS] Import warning (non-fatal): {e}")

# Seed calibration profiles
pipeline.plastic_profile = _cal["plastic"]
pipeline.glass_profile   = _cal["glass"]

print("[HARNESS] Import OK — all hardware stubs active.\n")

# ---------------------------------------------------------------------------
# 3.  Build synthetic test data
# ---------------------------------------------------------------------------
FRAME_320x240 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
FRAME_640x480 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

SPEC_PLASTIC_FP = dict(_cal["plastic"]["mean"])
SPEC_GLASS_FP   = dict(_cal["glass"]["mean"])
PLASTIC_PROFILE = _cal["plastic"]
GLASS_PROFILE   = _cal["glass"]

RAW_DATA_20 = [120.20 * 150 + random.gauss(0, 200) for _ in range(20)]

VISION_LABELS = [
    "glass_bottle", "plastic_bottle", "metal_can",
    "metal_bottle", "disposable_cup", "carton_tetrapack",
]

# ---------------------------------------------------------------------------
# 4.  Functions under test
# ---------------------------------------------------------------------------
BENCHMARKS = {
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
        doc = "Trapezoidal speed ramp with CRAWL_FACTOR",
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

    # ── Sensor fusion ────────────────────────────────────────────────────
    "fuse_results_glass": dict(
        fn  = lambda: pipeline.fuse_results(
            "glass_bottle", pipeline.Material.GLASS, 82.5, 175.0
        ),
        n   = 1000,
        doc = "Fusion — vision+spec agree on GLASS",
    ),
    "fuse_results_metal_heavy": dict(
        fn  = lambda: pipeline.fuse_results(
            "metal_can", pipeline.Material.METAL, 100.0, 620.0
        ),
        n   = 1000,
        doc = "Fusion — METAL over weight limit → GENERAL_WASTE",
    ),
    "fuse_results_disagree": dict(
        fn  = lambda: pipeline.fuse_results(
            "glass_bottle", pipeline.Material.PLASTIC, 45.0, 80.0
        ),
        n   = 1000,
        doc = "Fusion — vision/spec disagree, weight tiebreaker",
    ),
}

# ---------------------------------------------------------------------------
# 5a.  timeit microbenchmarks
# ---------------------------------------------------------------------------
import timeit, io, cProfile, pstats

_t_sA_start  = _time_mod.perf_counter()
_rss_sA_start = _rss_mb()

print("=" * 72)
print("  SECTION A — timeit MICROBENCHMARKS")
print("=" * 72)
print(f"{'Function':<32}  {'N':>6}  {'Total(s)':>10}  {'Per-call(µs)':>13}  {'ΔRSS(MB)':>9}  Note")
print("-" * 84)

timeit_results = {}
for name, cfg in BENCHMARKS.items():
    fn  = cfg["fn"]
    n   = cfg["n"]
    try:
        _rss_before = _rss_mb()
        total_s = timeit.timeit(fn, number=n)
        _drss   = _rss_mb() - _rss_before
        per_us  = (total_s / n) * 1e6
        timeit_results[name] = {"total_s": total_s, "per_us": per_us, "n": n,
                                "drss_mb": _drss}
        flag = "  ⚠  SLOW" if per_us > 5000 else ""
        print(f"  {name:<30}  {n:>6}  {total_s:>10.4f}  {per_us:>11.2f} µs  {_drss:>+7.1f} MB{flag}")
    except Exception as exc:
        print(f"  {name:<30}  ERROR: {exc}")
        timeit_results[name] = {"error": str(exc)}

_t_sA    = _time_mod.perf_counter() - _t_sA_start
_rss_sA  = _rss_mb()
print(f"\n  [Section A] wall time = {_t_sA:.3f} s   RSS now = {_rss_sA:.1f} MB  (Δ {_rss_sA - _rss_sA_start:+.1f} MB)")
print()

# ---------------------------------------------------------------------------
# 5b.  cProfile — full pipeline simulation
# ---------------------------------------------------------------------------
_t_sB_start  = _time_mod.perf_counter()
_rss_sB_start = _rss_mb()

print("=" * 72)
print("  SECTION B — cProfile CALL GRAPH  (50 simulated classifications)")
print("=" * 72)

def _simulate_pipeline(n_cycles=50):
    for _ in range(n_cycles):
        pipeline._iqr_clean(RAW_DATA_20[:])
        pipeline._aruco_preprocess(FRAME_320x240)
        pipeline.detect_aruco_angle(FRAME_320x240)
        pipeline.shortest_angle_diff(85.0, 220.0)
        pipeline.decelerated_speed(pipeline.SPEED_FWD, 45.0)

        fps   = [SPEC_PLASTIC_FP.copy() for _ in range(6)]
        clean, _ = pipeline._spec_reject_outliers(fps)
        scan  = pipeline._spec_average(clean)
        sam_p = pipeline._spec_sam(scan, PLASTIC_PROFILE["mean"])
        sam_g = pipeline._spec_sam(scan, GLASS_PROFILE["mean"])
        chi_p = pipeline._spec_chi_squared(scan, PLASTIC_PROFILE["mean"])
        chi_g = pipeline._spec_chi_squared(scan, GLASS_PROFILE["mean"])
        pipeline._spec_confidence_sam(
            sam_p, sam_g, chi_p, chi_g,
            PLASTIC_PROFILE["mean"], GLASS_PROFILE["mean"],
        )

        vl  = random.choice(VISION_LABELS)
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

_prof_path = os.path.join(_pipeline_dir, "main_final2.prof")
_prof.dump_stats(_prof_path)
print(f"[cProfile] Binary .prof written → {_prof_path}")
print(f"           View interactively:   snakeviz {_prof_path}")

_t_sB   = _time_mod.perf_counter() - _t_sB_start
_rss_sB = _rss_mb()
print(f"\n  [Section B] wall time = {_t_sB:.3f} s   RSS now = {_rss_sB:.1f} MB  (Δ {_rss_sB - _rss_sB_start:+.1f} MB)\n")

# ---------------------------------------------------------------------------
# 5c.  CodeCarbon — emissions per logical block
# ---------------------------------------------------------------------------
_t_sC_start  = _time_mod.perf_counter()
_rss_sC_start = _rss_mb()

print("=" * 72)
print("  SECTION C — CodeCarbon EMISSIONS TRACKER")
print("=" * 72)

try:
    from codecarbon import EmissionsTracker

    _blocks = {
        "pipeline_sim_250":      lambda: _simulate_pipeline(5),
        "aruco_detection_500":   lambda: [pipeline.detect_aruco_angle(FRAME_320x240)
                                          for _ in range(500)],
        "color_correction_1000": lambda: [pipeline.correct_frame(FRAME_320x240)
                                          for _ in range(1000)],
        "spec_classify_50":      lambda: [
            pipeline._spec_scan_and_classify(PLASTIC_PROFILE, GLASS_PROFILE, samples=2)
            for _ in range(50)
        ],
        "fusion_1000":           lambda: [
            pipeline.fuse_results(
                random.choice(VISION_LABELS),
                pipeline.Material.GLASS, 75.0, 130.0)
            for _ in range(1000)
        ],
    }

    carbon_results = {}
    for block_name, block_fn in _blocks.items():
        tracker = EmissionsTracker(
            project_name=f"final2_{block_name}",
            output_dir=_pipeline_dir,
            log_level="error",
            save_to_file=False,
            tracking_mode="process",
        )
        tracker.start()
        t0 = time.perf_counter()
        block_fn()
        elapsed = time.perf_counter() - t0
        emissions = tracker.stop()   # kg CO₂-eq

        kwh     = getattr(tracker, "_total_energy", None)
        kwh_val = float(kwh.kWh) if kwh is not None else float("nan")

        carbon_results[block_name] = {
            "elapsed_s":  elapsed,
            "co2_kg":     emissions,
            "energy_kWh": kwh_val,
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

_t_sC   = _time_mod.perf_counter() - _t_sC_start
_rss_sC = _rss_mb()
print(f"  [Section C] wall time = {_t_sC:.3f} s   RSS now = {_rss_sC:.1f} MB  (Δ {_rss_sC - _rss_sC_start:+.1f} MB)")
print()

# ---------------------------------------------------------------------------
# 5d.  pyJoules — hardware energy measurement (RAPL) with software fallback
# ---------------------------------------------------------------------------
_t_sD_start  = _time_mod.perf_counter()
_rss_sD_start = _rss_mb()

print("=" * 72)
print("  SECTION D — pyJoules ENERGY MEASUREMENT  (RAPL / CPU perf counters)")
print("=" * 72)

try:
    from pyJoules.energy_meter import EnergyMeter
    from pyJoules.device.rapl_device import RaplPackageDomain
    from pyJoules.handler.csv_handler import CSVHandler

    _domains = [RaplPackageDomain(0)]
    _handler = CSVHandler(os.path.join(_pipeline_dir, "pyjoules_final2.csv"))
    _meter   = EnergyMeter(_domains)

    _joules_blocks = {
        "aruco_100":      lambda: [pipeline.detect_aruco_angle(FRAME_320x240)
                                   for _ in range(100)],
        "spec_sam_500":   lambda: [pipeline._spec_sam(SPEC_PLASTIC_FP, PLASTIC_PROFILE["mean"])
                                   for _ in range(500)],
        "correction_500": lambda: [pipeline.correct_frame(FRAME_320x240)
                                   for _ in range(500)],
        "fusion_500":     lambda: [pipeline.fuse_results(
                                       "glass_bottle", pipeline.Material.GLASS, 75.0, 130.0)
                                   for _ in range(500)],
    }

    joules_results = {}
    for tag, fn in _joules_blocks.items():
        _meter.start(tag=tag)
        fn()
        _meter.stop()
        trace = _meter.get_trace()
        if trace:
            sample   = trace[-1]
            energy_J = sum(sample.energy[d] for d in sample.energy) / 1e6
            joules_results[tag] = energy_J
            print(f"  {tag:<28}  energy ≈ {energy_J*1000:.3f} mJ")
        _handler.process(trace)

    _handler.save_data()
    print(f"  [pyJoules] Per-sample CSV → pyjoules_final2.csv")

except Exception as e:
    print(f"  [INFO] pyJoules RAPL unavailable ({type(e).__name__}: {e})")
    print("  Running software-timer fallback...\n")

    _sw_blocks = {
        "aruco_100":      (lambda: [pipeline.detect_aruco_angle(FRAME_320x240) for _ in range(100)],      100),
        "spec_sam_500":   (lambda: [pipeline._spec_sam(SPEC_PLASTIC_FP, PLASTIC_PROFILE["mean"]) for _ in range(500)], 500),
        "correction_500": (lambda: [pipeline.correct_frame(FRAME_320x240) for _ in range(500)],           500),
        "fusion_500":     (lambda: [pipeline.fuse_results("glass_bottle", pipeline.Material.GLASS, 75.0, 130.0) for _ in range(500)], 500),
    }

    print(f"  {'Block':<28}  {'Wall time (s)':>14}  {'µs/call':>10}")
    print("  " + "-" * 56)
    for tag, (fn, n_calls) in _sw_blocks.items():
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        print(f"  {tag:<28}  {elapsed:>14.4f}  {elapsed/n_calls*1e6:>9.2f} µs")

_t_sD   = _time_mod.perf_counter() - _t_sD_start
_rss_sD = _rss_mb()
print(f"\n  [Section D] wall time = {_t_sD:.3f} s   RSS now = {_rss_sD:.1f} MB  (Δ {_rss_sD - _rss_sD_start:+.1f} MB)")
print()

# ---------------------------------------------------------------------------
# 6.  Consolidated summary
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
    if "aruco" in name and us > 1000:
        hints.append(f"• {name}: {us:.0f} µs — consider reducing DETECT_EVERY_N "
                     f"or skip CLAHE when ambient light is stable.")
    if "spec" in name and us > 100:
        hints.append(f"• {name}: {us:.0f} µs — dict key lookups dominate; "
                     f"convert calibration means to np.ndarray at load time.")
    if "fuse" in name and us > 200:
        hints.append(f"• {name}: {us:.0f} µs — pure logic; "
                     f"check beam_sensor.is_pressed doesn't add GPIO latency in prod.")

for h in hints or ["  No obvious hotspots above thresholds."]:
    print(f"  {h}")

print()

# ---------------------------------------------------------------------------
# 7.  Section E — Memory & Execution Time Summary
# ---------------------------------------------------------------------------
_t_total = _time_mod.perf_counter() - _t_script_start
_rss_end  = _rss_mb()

print("=" * 72)
print("  SECTION E — MEMORY & EXECUTION TIME SUMMARY")
print("=" * 72)
print(f"  {'Metric':<35}  {'Value':>15}")
print("  " + "-" * 54)
print(f"  {'Total script wall time':<35}  {_t_total:>14.3f} s")
print(f"  {'RSS at baseline':<35}  {_rss_baseline:>13.1f} MB")
print(f"  {'RSS at end':<35}  {_rss_end:>13.1f} MB")
print(f"  {'RSS net delta':<35}  {_rss_end - _rss_baseline:>+13.1f} MB")
print()
print(f"  {'Section':<35}  {'Wall time (s)':>13}  {'RSS end (MB)':>12}  {'ΔRSS (MB)':>10}")
print("  " + "-" * 74)
print(f"  {'A  timeit microbenchmarks':<35}  {_t_sA:>13.3f}  {_rss_sA:>12.1f}  {_rss_sA - _rss_sA_start:>+9.1f}")
print(f"  {'B  cProfile call graph':<35}  {_t_sB:>13.3f}  {_rss_sB:>12.1f}  {_rss_sB - _rss_sB_start:>+9.1f}")
print(f"  {'C  CodeCarbon emissions':<35}  {_t_sC:>13.3f}  {_rss_sC:>12.1f}  {_rss_sC - _rss_sC_start:>+9.1f}")
print(f"  {'D  pyJoules / sw-timer energy':<35}  {_t_sD:>13.3f}  {_rss_sD:>12.1f}  {_rss_sD - _rss_sD_start:>+9.1f}")
print()

print("[HARNESS] Done.")
print(f"          cProfile binary  : {_prof_path}")
print(f"          Snakeviz viewer  : snakeviz {_prof_path}")
print(f"          Full report      : {REPORT_PATH}")

# Restore stdout and close report file
sys.stdout = _tee._original
_tee.close()

# Clean up stub calibration file
try:
    os.remove(os.path.join(_pipeline_dir, "calibration_profile_stub.json"))
except OSError:
    pass

print(f"\n[HARNESS] Report saved → {REPORT_PATH}")
