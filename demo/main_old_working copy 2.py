import os
import time
import sys
import math
import json
import cv2
import numpy as np
import RPi.GPIO as GPIO
import concurrent.futures
import threading
from enum import Enum
from collections import Counter

import paho.mqtt.client as mqtt

# Ensure all relative paths (mobnet_models/, calibration.json, etc.)
# resolve correctly regardless of where the script is launched from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import subprocess

subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "--set-ctrl=auto_exposure=3"],          check=False)
subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "--set-ctrl=brightness=128"],            check=False)
subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "--set-ctrl=gain=0"],                   check=False)
subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "--set-ctrl=white_balance_automatic=1"], check=False)
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=auto_exposure=1"],               check=False)
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=exposure_time_absolute=25"],      check=False)
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=brightness=255"],                 check=False)
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=gain=200"],                       check=False)
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=white_balance_automatic=0"],      check=False)
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=white_balance_temperature=6500"], check=False)

# =============================================================================
# NOTE: setMouseCallback is NOT used anywhere in this file.
# Qt-backed OpenCV builds on Pi OS fail to create window handles synchronously,
# making setMouseCallback crash with "NULL window handler" regardless of how
# long you wait. All calibration UIs use keyboard input only.
# =============================================================================

# --- Sensor Imports ---
from gpiozero import DigitalInputDevice, Button
from hx711 import HX711
from as7343 import AS7343

# --- Vision Model Imports ---
from mobileNet_helpers import run_inference, load_model
from pick_model import return_latest_version_path

try:
    from monitoring import track_ram, track_temp, track_power, returns_latest_file_number
except ImportError:
    track_ram   = lambda: 0.0
    track_temp  = lambda: 0.0
    track_power = lambda: 0.0


# =============================================================================
# 0. ENUMS
# =============================================================================

class Material(Enum):
    METAL         = "METAL"
    GLASS         = "GLASS"
    PLASTIC       = "PLASTIC"
    GENERAL_WASTE = "GENERAL_WASTE"

class Compartment(Enum):
    METAL         = 220
    GLASS         = 65
    PLASTIC       = 320
    GENERAL_WASTE = 120

MATERIAL_TO_COMPARTMENT: dict[Material, Compartment] = {
    Material.METAL:         Compartment.METAL,
    Material.GLASS:         Compartment.GLASS,
    Material.PLASTIC:       Compartment.PLASTIC,
    Material.GENERAL_WASTE: Compartment.GENERAL_WASTE,
}


# =============================================================================
# COLOR CORRECTION — counteracts purple UV lighting (tracking camera only)
# =============================================================================

_CORRECTION_LUT_B = None
_CORRECTION_LUT_G = None
_CORRECTION_LUT_R = None

_hx711_lock = threading.Lock()


def correct_frame(frame):
    """White-balance correction for the tracking camera's UV cast."""
    global _CORRECTION_LUT_B, _CORRECTION_LUT_G, _CORRECTION_LUT_R
    if _CORRECTION_LUT_B is None:
        _CORRECTION_LUT_B = np.array([min(255, int(i * 0.457)) for i in range(256)], dtype=np.uint8)
        _CORRECTION_LUT_G = np.array([min(255, int(i * 0.85))  for i in range(256)], dtype=np.uint8)
        _CORRECTION_LUT_R = np.array([min(255, int(i * 0.926)) for i in range(256)], dtype=np.uint8)
    b, g, r = cv2.split(frame)
    return cv2.merge((
        cv2.LUT(b, _CORRECTION_LUT_B),
        cv2.LUT(g, _CORRECTION_LUT_G),
        cv2.LUT(r, _CORRECTION_LUT_R),
    ))


# =============================================================================
# 1. HARDWARE CONFIGURATION & INITIALIZATION
# =============================================================================

# -- Weight Sensor (HX711) --
hx = HX711(dout_pin=15, pd_sck_pin=14)
RATIO  = 120.20
OFFSET = 0
WEIGHT_TRIGGER_THRESHOLD = 3.0

METAL_CONTAMINATION_WEIGHT_LIMIT = 40.0
PLASTIC_GLASS_WEIGHT_THRESHOLD = 25.0


# -- Inductive Sensor (SN04-N) --
PIN_INDUCTIVE = 16
metal_sensor  = DigitalInputDevice(PIN_INDUCTIVE, pull_up=True)

# -- Break Beam Sensor --
# beam intact → pin HIGH (is_pressed=False)
# beam broken → pin LOW  (is_pressed=True)  ← contaminated
PIN_BEAM    = 26
beam_sensor = Button(PIN_BEAM, pull_up=True)

# -- Spectrometer (AS7343) --
SPEC_GAIN                     = 2048
SPEC_INTEGRATION_TIME         = 200
SPEC_CALIBRATION_SAMPLES      = 12
SPEC_SCAN_SAMPLES             = 2
SPEC_LOW_CONFIDENCE_THRESHOLD = 15
SPEC_CALIB_FILE               = "calibration.json"
SPEC_EXCLUDE_CHANNELS         = {'flicker', 'clear'}

VISION_LABEL_MAP: dict[str, tuple[str, str]] = {
    "carton_tetrapack": ("tetra",   "carton"),
    "disposable_cup":   ("general", "others"),
    "glass_bottle":     ("glass",   "bottle"),
    "glass_cup":        ("glass",   "bottle"),
    "metal_bottle":     ("metal",   "bottle"),
    "metal_can":        ("metal",   "can"),
    "plastic_bottle":   ("plastic", "bottle"),
}
_VISION_FALLBACK = ("general", "others")


try:
    spectrometer = AS7343()
    spectrometer.set_gain(SPEC_GAIN)
    spectrometer.set_integration_time(SPEC_INTEGRATION_TIME)
    spectrometer.set_illumination_led(False)
    SPECTROMETER_READY = True
except Exception as e:
    print(f"[WARNING] Spectrometer init failed: {e}")
    SPECTROMETER_READY = False

plastic_profile = None
glass_profile   = None

# -- Servos --
PIN_20 = 20
PIN_21 = 21
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(PIN_20, GPIO.OUT)
pwm20 = GPIO.PWM(PIN_20, 50)
pwm20.start(0)
GPIO.setup(PIN_21, GPIO.OUT)
pwm21 = GPIO.PWM(PIN_21, 50)
pwm21.start(0)

MOTOR_DIRECTION_SIGN                             = 1
SPEED_BWD, SPEED_FWD, SPEED_NEUTRAL, SPEED_STOP = 9.0, 6.0, 7.5, 0
DECEL_START, DECEL_NEAR, CRAWL_FACTOR            = 60, 25,0.5
ANGLE_TOLERANCE                                  = 12

last_angle_21      = -1
current_angle_20   = 0.0
current_speed_20   = -1.0
target_angle_20    = None
outbound_direction = None
is_homing          = False
HOME_ANGLE         = 85
calibration_mode   = False

_latest_frame   = None
_frame_lock     = threading.Lock()
_tracking_pause = threading.Event()   # set = daemon paused, clear = running

center_x, center_y = 169, 113


# =============================================================================
# 1b. CAMERAS
#
#   cap_tracking  (index 0) — servo tracking daemon, 320×240
#   cap_vision    (index 2) — MobileNet inference,   640×480
# =============================================================================

# -- Tracking camera (rotation arm) --
cap_tracking = cv2.VideoCapture(0)
cap_tracking.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap_tracking.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap_tracking.set(cv2.CAP_PROP_BUFFERSIZE,   1)
# cap_tracking.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
# cap_tracking.set(cv2.CAP_PROP_EXPOSURE,     -5)
if not cap_tracking.isOpened():
    print("[ERROR] Cannot open tracking camera (index 0). Check /dev/video0.")
    sys.exit(1)

# -- Vision camera (object classification) --
VISION_CAM_INDEX = 2
cap_vision = cv2.VideoCapture(VISION_CAM_INDEX)
cap_vision.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap_vision.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_vision.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap_vision.isOpened():
    print(f"[WARNING] Vision camera (index {VISION_CAM_INDEX}) not available — "
          f"vision path will be skipped.")
    cap_vision = None


# =============================================================================
# 1c. VISION MODEL — loaded once at startup, shared by path_1_vision_model()
# =============================================================================

VISION_FRAME_W, VISION_FRAME_H = 640, 480
VISION_CAPTURE_BOX_SIZE        = 320
VISION_MODEL_INPUT_SIZE        = 224
VISION_QUANTIZED               = True

VISION_BUFFER_THRESHOLD = 10
VISION_FREQ_THRESHOLD   = 7
VISION_CONF_THRESHOLD   = 0.70
VISION_TIMEOUT_S        = 1.0

print("[VISION] Loading MobileNet model...")
latest_model_path, _ = return_latest_version_path("mobilenet")
vision_model, vision_input_details, vision_output_details, _vision_model_path = load_model(
    quantized=VISION_QUANTIZED,
    model_path=latest_model_path,
)

labels_path = os.path.join(latest_model_path, "labels.txt")
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
    print(f"[VISION] Loaded class names: {CLASS_NAMES}")
else:
    CLASS_NAMES = []
    print("[VISION] WARNING: labels.txt not found — class IDs will be used as names")


# =============================================================================
# 1d. MQTT — publishes one message per classification cycle
# =============================================================================

MQTT_BROKER = "10.254.93.107"
MQTT_PORT   = 1883
MQTT_TOPIC  = "pi/raw_transaction"

# _MATERIAL_TO_MQTT = {
#     Material.METAL:         "metal",
#     Material.GLASS:         "glass",
#     Material.PLASTIC:       "plastic",
#     Material.GENERAL_WASTE: "general",
# }

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

def _mqtt_on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[MQTT]  Connected to broker {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"[MQTT]  Connection failed — reason code {reason_code}")

def _mqtt_on_disconnect(client, userdata, flags, reason_code, properties):
    print(f"[MQTT]  Disconnected (reason={reason_code}) — attempting reconnect...")
    while True:
        try:
            client.reconnect()
            print("[MQTT]  Reconnected.")
            break
        except Exception as e:
            print(f"[MQTT]  Reconnect failed: {e}  — retrying in 5 s")
            time.sleep(5)

mqtt_client.on_connect    = _mqtt_on_connect
mqtt_client.on_disconnect = _mqtt_on_disconnect

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    mqtt_client.loop_start()
    print(f"[MQTT]  Connecting to {MQTT_BROKER}:{MQTT_PORT} ...")
except Exception as e:
    print(f"[MQTT]  WARNING: Could not connect to broker ({e}). Publishing will be skipped.")


def mqtt_publish_result(final_material, mqtt_material_str: str,
                        mqtt_shape_str: str, weight_g: float):
    payload = {
        "material": mqtt_material_str,
        "type":     mqtt_shape_str,
        "weight":   f"{weight_g:.1f}g",
    }
    try:
        result = mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
        result.wait_for_publish(timeout=2.0)
        print(f"[MQTT]  Published → {payload}")
    except Exception as e:
        print(f"[MQTT]  Publish failed: {e}")


# =============================================================================
# 1e. ARUCO MARKER — tracking daemon & compartment calibration
#
#   Mirrors servo_Controller.py exactly:
#     - 320×240 capture, detect on 640×480 (2× upscale)
#     - CLAHE for local contrast boost under UV lighting
#     - Frame skip: detector runs every DETECT_EVERY_N frames,
#       last known angle held in between
#     - Direction-reversal stop inside decel zone (servo_Controller.py logic)
#     - Angle from marker orientation only — no center_x/y required
# =============================================================================
_aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_aruco_params = cv2.aruco.DetectorParameters()

_aruco_params.adaptiveThreshWinSizeMin       = 3
_aruco_params.adaptiveThreshWinSizeMax       = 21
_aruco_params.adaptiveThreshWinSizeStep      = 2
_aruco_params.minMarkerPerimeterRate         = 0.1
_aruco_params.maxMarkerPerimeterRate         = 0.5
_aruco_params.errorCorrectionRate            = 1.0
_aruco_params.cornerRefinementMethod         = cv2.aruco.CORNER_REFINE_SUBPIX
_aruco_params.cornerRefinementWinSize        = 5
_aruco_params.cornerRefinementMaxIterations  = 30
_aruco_params.polygonalApproxAccuracyRate    = 0.05
_aruco_params.minCornerDistanceRate          = 0.03
_aruco_params.minDistanceToBorder            = 1

_aruco_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)

ARUCO_UPSCALE    = 2
DETECT_EVERY_N   = 2    # run detector every N frames, hold last result in between
TRACKING_TIMEOUT = 2.0  # seconds before watchdog stops motor on marker loss

_daemon_frame_ctr  = 0
_last_good_corners = None


def _aruco_preprocess(frame):
    """2× upscale → grayscale → CLAHE. Local contrast boost, UV-friendly."""
    big   = cv2.resize(frame, None, fx=ARUCO_UPSCALE, fy=ARUCO_UPSCALE,
                       interpolation=cv2.INTER_LINEAR)
    gray  = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_aruco_angle(frame):
    """
    Returns the arm angle purely from the marker's own orientation.
    Top edge points toward the rotation centre, so the bottom→top
    vector IS the inward direction — no center_x/y required.

    Detection runs on a 2× upscaled + CLAHE image.
    All returned corner coordinates are in original (320×240) frame space.

    Returns:
        (angle_degrees, marker_corners)  — or  (None, None) if not found.
    """
    gray = _aruco_preprocess(frame)
    corners, ids, _ = _aruco_detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None

    c = corners[0][0] / ARUCO_UPSCALE   # TL, TR, BR, BL in original coords

    top_x    = (c[0][0] + c[1][0]) / 2
    top_y    = (c[0][1] + c[1][1]) / 2
    bottom_x = (c[2][0] + c[3][0]) / 2
    bottom_y = (c[2][1] + c[3][1]) / 2

    dx    = top_x - bottom_x
    dy    = top_y - bottom_y
    angle = math.degrees(math.atan2(-dy, dx)) % 360

    return angle, c


# =============================================================================
# 2. DRAW HELPERS
# =============================================================================

def _draw_crosshair(img, x, y, color=(0, 255, 0), r=10):
    cv2.circle(img, (x, y), r, color, 2)
    cv2.line(img, (x - r - 5, y), (x + r + 5, y), color, 1)
    cv2.line(img, (x, y - r - 5), (x, y + r + 5), color, 1)

def _put_lines(img, lines, y0=14, dy=16, color=(220, 220, 220), scale=0.40):
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (5, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# =============================================================================
# 3. SERVO HELPERS
# =============================================================================

def set_speed_20(speed):
    global current_speed_20
    if current_speed_20 != speed:
        label = {SPEED_FWD: "FWD(CW)", SPEED_BWD: "BWD(CCW)",
                 SPEED_NEUTRAL: "NEUTRAL", SPEED_STOP: "STOP"}.get(speed, f"{speed:.2f}")
        print(f"[MOTOR] Speed → {label}  (duty={speed})")
        pwm20.ChangeDutyCycle(speed)
        current_speed_20 = speed

def set_angle_instant_21(angle):
    global last_angle_21
    if angle != last_angle_21:
        duty = round(float(angle) / 18.0 + 2.0, 2)
        print(f"[ARM]   Angle → {angle}°  (duty={duty})")
        pwm21.ChangeDutyCycle(duty)
        time.sleep(1.0)
        pwm21.ChangeDutyCycle(0)
        last_angle_21 = angle

def shortest_angle_diff(current, target):
    return (target - current + 180) % 360 - 180

def pick_direction(diff):
    return SPEED_FWD if MOTOR_DIRECTION_SIGN * diff > 0 else SPEED_BWD

def decelerated_speed(direction, distance):
    if   distance >= DECEL_START: t = 1.0
    elif distance <= DECEL_NEAR:  t = CRAWL_FACTOR
    else: t = ((distance - DECEL_NEAR) / (DECEL_START - DECEL_NEAR)) ** 2
    return round(SPEED_NEUTRAL + t * (direction - SPEED_NEUTRAL), 2)

def _set_target(angle):
    global target_angle_20, outbound_direction, is_homing
    target_angle_20    = float(angle) % 360
    is_homing          = False
    outbound_direction = (SPEED_BWD
                          if target_angle_20 == float(Compartment.GENERAL_WASTE.value)
                          else SPEED_FWD)
    print(f"[TARGET] Set → {target_angle_20:.1f}°  "
          f"direction={'CCW' if outbound_direction == SPEED_BWD else 'CW'}")

# =============================================================================
# 4. WEIGHT HELPERS
# =============================================================================

def _iqr_clean(vals: list) -> list:
    if len(vals) < 4:
        return vals
    s   = sorted(vals)
    q1  = s[len(s) // 4]
    q3  = s[(3 * len(s)) // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    clean = [v for v in s if lo <= v <= hi]
    return clean if clean else s
def tare_scale():
    global OFFSET
    input("[SCALE]  Ensure platform is EMPTY then press Enter to tare...")
    time.sleep(3)
    with _hx711_lock:
        vals = hx.get_raw_data(20)
    if vals:
        clean   = _iqr_clean(vals)
        OFFSET  = int(sum(clean) / len(clean))
        dropped = len(vals) - len(clean)
        print(f"[SCALE]  Tare complete.  IQR mean={OFFSET}  "
              f"dropped={dropped}/{len(vals)}  RATIO={RATIO}")
    else:
        print("[SCALE]  Tare FAILED — no data from HX711")

def calibrate_ratio():
    global RATIO
    known = float(input("[SCALE]  Enter known weight in grams (e.g. 500): "))
    input(f"[SCALE]  Place the {known:.0f}g object on the scale then press Enter...")
    time.sleep(1.5)
    with _hx711_lock:
        vals = hx.get_raw_data(20)
    if not vals:
        print("[SCALE]  Calibration FAILED — no data")
        return
    clean = _iqr_clean(vals)
    raw   = sum(clean) / len(clean)
    RATIO = (raw - OFFSET) / known
    print(f"[SCALE]  Calibration complete.  raw={raw:.0f}  RATIO={RATIO:.2f}")
    input("[SCALE]  Remove the object then press Enter...")

def get_weight() -> float:
    with _hx711_lock:
        vals = hx.get_raw_data(20)
    if not vals:
        return 0.0
    clean = _iqr_clean(vals)
    return (sum(clean) / len(clean) - OFFSET) / RATIO




# =============================================================================
# 5. CONTAMINATION CHECK — material-aware
# =============================================================================

# def is_contaminated(material: Material, weight: float) -> bool:
#     print(f"[CONTAM] Checking {material.value}  weight={weight:.1f} g  "
#           f"beam={'BROKEN' if beam_sensor.is_pressed else 'CLEAR'}")

#     if material == Material.METAL:
#         if weight > METAL_CONTAMINATION_WEIGHT_LIMIT:
#             print(f"[CONTAM] ✗ METAL too heavy "
#                   f"({weight:.1f} g > {METAL_CONTAMINATION_WEIGHT_LIMIT} g) → General Waste")
#             return True
#         print(f"[CONTAM] ✓ METAL weight OK ({weight:.1f} g ≤ {METAL_CONTAMINATION_WEIGHT_LIMIT} g)")
#         return False

#     if material in (Material.GLASS, Material.PLASTIC):
#         if beam_sensor.is_pressed:
#             print(f"[CONTAM] ✗ Beam BROKEN — liquid detected in "
#                   f"{material.value} container → General Waste")
#             return True
#         print(f"[CONTAM] ✓ Beam CLEAR — {material.value} container is clean")
#         return False

#     print(f"[CONTAM] ✓ {material.value} — no contamination check required")
#     return False


# =============================================================================
# 6. SPECTROMETER — FULL SPECTRAL FINGERPRINTING
# =============================================================================

def _spec_get_fingerprint():
    combined = {}
    for bank in [0, 1]:
        spectrometer.bank_select(bank)
        time.sleep(0.30)
        try:
            readings = spectrometer.get_data()[0]
            kept = {ch: max(float(val), 0.0)
                    for ch, val in readings.items()
                    if ch not in SPEC_EXCLUDE_CHANNELS}
            combined.update(kept)
            print(f"[SPEC]   Bank {bank} raw channels: "
                  + "  ".join(f"{k}={v:.0f}" for k, v in kept.items()))
        except Exception as e:
            print(f"[SPEC]   Bank {bank} read error: {e}")
    if not combined:
        print("[SPEC]   No data from either bank — fingerprint failed")
        return None
    total = sum(combined.values())
    if total < 1:
        print(f"[SPEC]   Total signal too low ({total:.1f}) — fingerprint rejected")
        return None
    return {k: v / total for k, v in combined.items()}

def _spec_average(fps):
    keys = fps[0].keys()
    return {k: sum(fp[k] for fp in fps) / len(fps) for k in keys}

def _spec_euclidean(a, b):
    common = set(a) & set(b)
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in common))

# def _spec_weighted_dist(scan_fp, profile):
#     mean, std = profile["mean"], profile["std"]
#     common    = set(scan_fp) & set(mean)
#     return math.sqrt(sum(
#         ((scan_fp[k] - mean[k]) / (std[k] + 1e-6)) ** 2 for k in common
#     ))

def _spec_reject_outliers(fps):
    if len(fps) < 3:
        return fps, 0
    avg   = _spec_average(fps)
    dists = [_spec_euclidean(fp, avg) for fp in fps]
    mu    = sum(dists) / len(dists)
    sigma = math.sqrt(sum((d - mu) ** 2 for d in dists) / len(dists))
    thr   = mu + 2 * sigma
    clean = [fp for fp, d in zip(fps, dists) if d <= thr]
    dropped = len(fps) - len(clean)
    if dropped:
        print(f"[SPEC]   Outlier rejection: mu={mu:.4f}  sigma={sigma:.4f}  "
              f"threshold={thr:.4f}  dropped={dropped}/{len(fps)}")
    return clean, dropped

def _spec_save_calibration(p, g):
    with open(SPEC_CALIB_FILE, "w") as f:
        json.dump({"plastic": p, "glass": g}, f, indent=2)
    print("[SPEC] Calibration saved to disk.")

def _spec_load_calibration():
    if not os.path.exists(SPEC_CALIB_FILE):
        return None, None
    try:
        with open(SPEC_CALIB_FILE) as f:
            d = json.load(f)
        print("[SPEC] Loaded saved calibration from disk.")
        return d["plastic"], d["glass"]
    except Exception as e:
        print(f"[SPEC] Could not load calibration ({e}). Will recalibrate.")
        return None, None

def _spec_calibrate_material(label, samples=SPEC_CALIBRATION_SAMPLES):
    print(f"\n  → Place {label} directly on the sensor.")
    input("    Press Enter when ready...")
    print(f"  Collecting {samples} samples...")
    fps = []
    for _ in range(samples):
        fp = _spec_get_fingerprint()
        if fp:
            fps.append(fp)
        time.sleep(0.12)
    if len(fps) < 3:
        print("  ERROR: Too few readings. Try again.")
        return None
    clean, dropped = _spec_reject_outliers(fps)
    if dropped:
        print(f"  (Dropped {dropped} noisy samples)")
    if len(clean) < 2:
        print("  ERROR: Too many rejected. Check placement.")
        return None
    mean = _spec_average(clean)
    std  = {k: math.sqrt(sum((fp[k] - mean[k]) ** 2 for fp in clean) / len(clean))
            for k in mean}
    print(f"  ✓ {label} calibrated from {len(clean)} clean samples.")
    return {"mean": mean, "std": std, "n": len(clean)}

# def _spec_confidence(dp, dg, pp, gp):
#     spread = _spec_euclidean(pp["mean"], gp["mean"])
#     if spread < 1e-6:
#         return 0.0
#     prox   = max(0.0, 1.0 - min(dp, dg) / spread)
#     margin = min(1.0, abs(dp - dg) / spread)
#     return (prox * 0.6 + margin * 0.4) * 100
# =============================================================================
# SPECTROMETER — UPDATED DISTANCE & CLASSIFICATION FUNCTIONS
#
# Drop these in as direct replacements for the following functions in main.py:
#   - _spec_weighted_dist        → REMOVED (replaced by SAM + chi-squared)
#   - _spec_confidence           → replaced by _spec_confidence_sam
#   - _spec_scan_and_classify    → updated in-place
#
# Everything else (_spec_get_fingerprint, _spec_average, _spec_euclidean,
# _spec_reject_outliers, _spec_calibrate_material, etc.) is unchanged.
#
# WHAT CHANGED AND WHY:
#   Old: weighted Euclidean distance with 1/sigma per channel
#        - treats channels as independent (wrong — adjacent bands correlate)
#        - sensitive to absolute intensity shifts from placement variance
#
#   New: SAM (primary, 70%) + Chi-squared (secondary, 30%)
#        - SAM measures the ANGLE between spectral vectors, so uniform
#          brightness shifts from LED flicker or bad placement are ignored
#        - Chi-squared down-weights noisy near-zero channels naturally
#        - Both metrics work directly on the normalised fingerprints already
#          produced by _spec_get_fingerprint — no extra calibration needed
# =============================================================================

import math
import numpy as np


# -----------------------------------------------------------------------------
# DISTANCE METRICS
# -----------------------------------------------------------------------------

def _spec_sam(scan_fp: dict, profile_mean: dict) -> float:
    """
    Spectral Angle Mapper — returns angle in degrees between scan and profile.
    Lower = closer match.  Range: [0, 90].

    Completely invariant to uniform illumination scaling, making it robust
    to LED intensity drift and inconsistent item placement height.
    """
    keys = set(scan_fp) & set(profile_mean)
    va   = np.array([scan_fp[k]      for k in keys], dtype=float)
    vb   = np.array([profile_mean[k] for k in keys], dtype=float)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm < 1e-9:
        return 90.0  # degenerate — treat as maximally different
    cos_theta = np.dot(va, vb) / norm
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def _spec_chi_squared(scan_fp: dict, profile_mean: dict) -> float:
    """
    Chi-squared distance between scan fingerprint and profile mean.
    Lower = closer match.

    Down-weights channels with low signal in both scan and profile,
    so noisy near-zero channels don't dominate the decision.
    """
    keys = set(scan_fp) & set(profile_mean)
    return float(sum(
        (scan_fp[k] - profile_mean[k]) ** 2 / (scan_fp[k] + profile_mean[k] + 1e-9)
        for k in keys
    ))


# -----------------------------------------------------------------------------
# CONFIDENCE — replaces _spec_confidence
# -----------------------------------------------------------------------------

def _spec_confidence_sam(
    sam_p: float, sam_g: float,
    chi_p: float, chi_g: float,
    pp_mean: dict, gp_mean: dict,
) -> float:
    """
    Confidence score [0, 100] using SAM (70%) + chi-squared margin (30%).

    SAM component:
      - profile_sep  : angular distance between the two calibration means
                       (how different plastic and glass look to the sensor)
      - prox         : how close the scan is to the winning profile,
                       relative to that separation  (0 = at the profile, 1 = far)
      - sam_margin   : how decisively one profile beats the other

    Chi-squared component:
      - simple normalised margin — which metric agrees with SAM and by how much

    Low confidence means the two profiles look similar in this environment
    (poor sensor placement, contaminated reference, or sensor degradation).
    """
    # Angular separation between the two calibration profiles
    profile_sep = _spec_sam(pp_mean, gp_mean)
    if profile_sep < 0.1:           # profiles indistinguishable — can't be confident
        return 0.0

    winner_angle = min(sam_p, sam_g)

    # Proximity: scan angle to winner, normalised by profile separation
    prox       = max(0.0, 1.0 - winner_angle / profile_sep)

    # SAM margin: decisiveness of the angle difference
    sam_margin = min(1.0, abs(sam_p - sam_g) / profile_sep)

    sam_conf   = prox * 0.6 + sam_margin * 0.4

    # Chi-squared margin (normalised, direction-agnostic)
    chi_total  = chi_p + chi_g + 1e-9
    chi_margin = abs(chi_p - chi_g) / chi_total   # 0–1

    combined   = (0.70 * sam_conf + 0.30 * chi_margin) * 100.0
    return min(combined, 100.0)


# -----------------------------------------------------------------------------
# CLASSIFICATION — drop-in replacement for _spec_scan_and_classify
# -----------------------------------------------------------------------------

def _spec_scan_and_classify(pp, gp, samples=6):
    """
    Collect `samples` spectral fingerprints, reject outliers, average, then
    classify as PLASTIC or GLASS using SAM (primary) + chi-squared (secondary).

    Ensemble decision:
      Each metric votes independently on normalised scores.  SAM carries 70%
      of the weight; chi-squared carries 30%.  A lower combined score means
      a better match.

    Returns:
        (Material or None,  confidence float,  debug dict)
    """

    print(f"[SPEC]   Starting scan — collecting {samples} fingerprints...")
    fps = []
    for i in range(samples):
        fp = _spec_get_fingerprint()
        if fp:
            fps.append(fp)
            print(f"[SPEC]   Sample {i+1}/{samples} collected  (channels={len(fp)})")
        else:
            print(f"[SPEC]   Sample {i+1}/{samples} FAILED — skipped")
        import time; time.sleep(0.08)

    print(f"[SPEC]   Raw collected: {len(fps)}/{samples}")
    if not fps:
        print("[SPEC]   No valid samples — classification aborted")
        return None, 0, {}

    clean, dropped = _spec_reject_outliers(fps)
    print(f"[SPEC]   After outlier rejection: {len(clean)} clean, {dropped} dropped")
    if not clean:
        print("[SPEC]   All samples rejected — classification aborted")
        return None, 0, {}

    scan = _spec_average(clean)

    # --- SAM distances -------------------------------------------------------
    sam_p = _spec_sam(scan, pp["mean"])
    sam_g = _spec_sam(scan, gp["mean"])

    # --- Chi-squared distances -----------------------------------------------
    chi_p = _spec_chi_squared(scan, pp["mean"])
    chi_g = _spec_chi_squared(scan, gp["mean"])

    # --- Ensemble score (lower = better match) --------------------------------
    # Normalise each metric so the two are on a common [0,1] scale before mixing
    sam_total   = sam_p + sam_g + 1e-9
    chi_total   = chi_p + chi_g + 1e-9

    SAM_WEIGHT  = 0.70
    CHI_WEIGHT  = 0.30

    score_p = SAM_WEIGHT * (sam_p / sam_total) + CHI_WEIGHT * (chi_p / chi_total)
    score_g = SAM_WEIGHT * (sam_g / sam_total) + CHI_WEIGHT * (chi_g / chi_total)

    result  = Material.PLASTIC if score_p <= score_g else Material.GLASS
    winner  = "PLASTIC" if score_p <= score_g else "GLASS"
    loser   = "GLASS"   if score_p <= score_g else "PLASTIC"

    # --- Confidence ----------------------------------------------------------
    conf = _spec_confidence_sam(sam_p, sam_g, chi_p, chi_g, pp["mean"], gp["mean"])

    # --- Logging -------------------------------------------------------------
    profile_sep = _spec_sam(pp["mean"], gp["mean"])
    print(
        f"[SPEC]   SAM  → plastic={sam_p:.2f}°  glass={sam_g:.2f}°  "
        f"profile_sep={profile_sep:.2f}°"
    )
    print(
        f"[SPEC]   Chi² → plastic={chi_p:.4f}  glass={chi_g:.4f}"
    )
    print(
        f"[SPEC]   Ensemble score → plastic={score_p:.4f}  glass={score_g:.4f}"
    )
    print(f"[SPEC]   → {winner} wins over {loser}  confidence={conf:.1f}%")

    debug = {
        "sam_plastic":     sam_p,
        "sam_glass":       sam_g,
        "chi_plastic":     chi_p,
        "chi_glass":       chi_g,
        "score_plastic":   score_p,
        "score_glass":     score_g,
        "profile_sep_deg": profile_sep,
        "samples_used":    len(clean),
        "samples_dropped": dropped,
    }
    return result, conf, debug

# =============================================================================
# 7. SERVO TRACKING DAEMON  (uses cap_tracking — camera index 0)
#
#   Frame-skip logic mirrors servo_Controller.py:
#     - Detector runs every DETECT_EVERY_N frames
#     - On skip frames: hold last known angle (arm can't jump far between frames)
#     - Direction-reversal guard inside decel zone stops motor cleanly
#     - Watchdog fires after TRACKING_TIMEOUT seconds of true marker loss
# =============================================================================

def servo_tracking_daemon():
    global current_angle_20, target_angle_20, outbound_direction, is_homing
    global _latest_frame, _daemon_frame_ctr, _last_good_corners

    last_marker_seen = time.time()
    _last_log_time   = 0.0
    _last_tape_state = None

    while True:
        if _tracking_pause.is_set():
            time.sleep(0.05)
            continue

        ret, frame = cap_tracking.read()
        if not ret:
            time.sleep(0.01)
            continue

        with _frame_lock:
            _latest_frame = frame.copy()
        
        cv2.imshow("Tracking Camera", frame)
        cv2.waitKey(1)

        # --- Frame-skip detection (mirrors servo_Controller.py) --------------
        _daemon_frame_ctr += 1
        if _daemon_frame_ctr % DETECT_EVERY_N == 0:
            detected_angle, aruco_corners = detect_aruco_angle(frame)
            if aruco_corners is not None:
                _last_good_corners = aruco_corners
        else:
            # Hold last known angle between detections
            detected_angle = current_angle_20 \
                if (time.time() - last_marker_seen) < TRACKING_TIMEOUT else None
            aruco_corners  = _last_good_corners

        marker_found = detected_angle is not None

        if marker_found:
            last_marker_seen = time.time()
            current_angle_20 = detected_angle

        if marker_found != _last_tape_state:
            if marker_found and _last_tape_state is not None:
                print(f"[DAEMON] Marker reacquired at {current_angle_20:.1f}° — resuming")
            _last_tape_state = marker_found

        # --- Watchdog --------------------------------------------------------
        if not marker_found:
            if target_angle_20 is not None:
                set_speed_20(SPEED_STOP)
            elapsed = time.time() - last_marker_seen
            if elapsed > TRACKING_TIMEOUT and target_angle_20 is not None:
                print(f"[DAEMON] Marker lost {elapsed:.1f} s — paused, awaiting reacquisition")

        # --- Motor control ---------------------------------------------------
        if target_angle_20 is not None and marker_found:
            diff = shortest_angle_diff(current_angle_20, target_angle_20)
            dist = abs(diff)

            now = time.time()
            if now - _last_log_time >= 1.0:
                mode = "HOMING" if is_homing else "TARGETING"
                print(f"[DAEMON] {mode}  current={current_angle_20:.1f}°  "
                      f"target={target_angle_20:.1f}°  dist={dist:.1f}°  "
                      f"speed={current_speed_20}")
                _last_log_time = now

            if dist <= ANGLE_TOLERANCE:
                set_speed_20(SPEED_STOP)
                if calibration_mode:
                    pass
                elif not is_homing:
                    print(f"\n[DAEMON] TARGET REACHED at {current_angle_20:.1f}°  "
                          f"(target={target_angle_20:.1f}°  dist={dist:.1f}°)")
                    # NEW — waits for marker to reappear before starting homing
                    print("[ARM]    Tilting up...")
                    set_angle_instant_21(100)
                    time.sleep(1.0)
                    print("[ARM]    Tilting down...")
                    set_angle_instant_21(0)

                    came_from_general_waste = (
                        abs(current_angle_20 - float(Compartment.GENERAL_WASTE.value))
                        <= ANGLE_TOLERANCE
                    )
                    outbound_direction = SPEED_FWD if came_from_general_waste else SPEED_BWD

                    print("[DAEMON] Tilt complete — waiting for marker to reappear before homing...")
                    _marker_wait_start = time.time()
                    while True:
                        ret, _wf = cap_tracking.read()
                        if not ret:
                            time.sleep(0.01)
                            continue
                        _recheck_angle, _ = detect_aruco_angle(_wf)
                        if _recheck_angle is not None:
                            current_angle_20 = _recheck_angle
                            last_marker_seen = time.time()
                            print(f"[DAEMON] Marker reacquired at {current_angle_20:.1f}°  "
                                f"(waited {time.time() - _marker_wait_start:.2f} s) — starting HOME")
                            break
                        if time.time() - _marker_wait_start > 5.0:
                            print("[DAEMON] Marker not reacquired after 5 s — starting HOME anyway")
                            break
                        time.sleep(0.01)

                    target_angle_20 = HOME_ANGLE
                    is_homing       = True
                else:
                    print(f"[DAEMON] HOME REACHED at {current_angle_20:.1f}°  "
                          f"(target={HOME_ANGLE}°  dist={dist:.1f}°)  System idle.")
                    target_angle_20    = None
                    outbound_direction = None
                    is_homing          = False
            else:
                    set_speed_20(decelerated_speed(outbound_direction, dist))

        time.sleep(0.01)


# =============================================================================
# 8. STARTUP CALIBRATION — STEP 1: Rotation Centre Point
# =============================================================================

_KEY_UP    = 82
_KEY_DOWN  = 84
_KEY_LEFT  = 81
_KEY_RIGHT = 83

def calibrate_center_point():
    global center_x, center_y

    WIN = "STEP 1/3  |  Rotation Centre  |  Arrows/WASD=move  R=reset  SPACE=confirm"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    cx, cy = center_x, center_y

    print("\n" + "=" * 60)
    print("  STEP 1/3 — ROTATION CENTRE CALIBRATION")
    print("  Use arrow keys (or WASD) to move the crosshair")
    print("  onto the arm's pivot point, then press SPACE.")
    print("  R = reset to image centre (160, 120)")
    print("=" * 60)

    while True:
        ret, frame = cap_tracking.read()
        if not ret:
            cv2.waitKey(30)
            continue

        disp = frame.copy()
        for r in [40, 80, 115]:
            cv2.circle(disp, (cx, cy), r, (30, 70, 30), 1)
        _draw_crosshair(disp, cx, cy, (0, 255, 80))
        _put_lines(disp, [
            "STEP 1/3 — Set Rotation Centre",
            f"Centre: ({cx}, {cy})",
            "Arrows / WASD = move  |  R = reset",
            "SPACE = confirm",
        ])

        cv2.imshow(WIN, disp)
        key = cv2.waitKey(30) & 0xFF
        raw = cv2.waitKey(1)
        k   = key if key != 255 else (raw & 0xFF)

        step = 1
        if   k == _KEY_UP    or k == ord('w'):  cy = max(0,               cy - step)
        elif k == _KEY_DOWN  or k == ord('s'):  cy = min(disp.shape[0]-1, cy + step)
        elif k == _KEY_LEFT  or k == ord('a'):  cx = max(0,               cx - step)
        elif k == _KEY_RIGHT or k == ord('d'):  cx = min(disp.shape[1]-1, cx + step)
        elif k == ord('r'):  cx, cy = 160, 120
        elif k == ord(' '):
            center_x, center_y = cx, cy
            print(f"[CALIB] Rotation centre confirmed → ({center_x}, {center_y})")
            break

    cv2.destroyWindow(WIN)
    cv2.waitKey(1)

def calibrate_weight_thresholds():
    global WEIGHT_TRIGGER_THRESHOLD, PLASTIC_GLASS_WEIGHT_THRESHOLD

    print("\n" + "=" * 60)
    print("  WEIGHT THRESHOLD CALIBRATION")
    print("=" * 60)

    # --- Plastic bottle → trigger threshold ----------------------------------
    input("\n  Place your EMPTY PLASTIC BOTTLE on the scale, then press Enter...")
    print("  Measuring plastic bottle weight...")
    time.sleep(0.5)
    with _hx711_lock:
        plastic_w = get_weight()
    # Trigger threshold = half the plastic bottle weight, minimum 2g
    WEIGHT_TRIGGER_THRESHOLD = max(2.0, round(plastic_w / 2.0, 1))
    print(f"  Plastic bottle  : {plastic_w:.1f} g")
    print(f"  Trigger threshold set to: {WEIGHT_TRIGGER_THRESHOLD:.1f} g")

    input("\n  Remove plastic bottle. Place your EMPTY GLASS BOTTLE on the scale, then press Enter...")
    print("  Measuring glass bottle weight...")
    time.sleep(0.5)
    glass_w = get_weight()
    # Midpoint between plastic and glass weight as the tiebreaker threshold
    PLASTIC_GLASS_WEIGHT_THRESHOLD = round((plastic_w + glass_w) / 2.0, 1)
    print(f"  Glass bottle    : {glass_w:.1f} g")
    print(f"  Plastic/Glass threshold set to: {PLASTIC_GLASS_WEIGHT_THRESHOLD:.1f} g")

    input("\n  Remove glass bottle. Press Enter to continue...")
    print(f"\n  ✓ Weight thresholds calibrated:")
    print(f"      Trigger (object detected) : ≥ {WEIGHT_TRIGGER_THRESHOLD:.1f} g")
    print(f"      Plastic vs Glass tiebreak : ≥ {PLASTIC_GLASS_WEIGHT_THRESHOLD:.1f} g → GLASS")
    print("=" * 60)
# =============================================================================
# 9. STARTUP CALIBRATION — STEP 2: Compartment Angles
# =============================================================================

def calibrate_compartment_angles():
    global calibration_mode

    calibration_mode = True

    WIN = "STEP 2/3  |  Compartment Angles  |  D/A=+-1  C/Z=+-5  SPACE=confirm"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    ORDER  = [Material.METAL, Material.GLASS, Material.PLASTIC, Material.GENERAL_WASTE]
    COLORS = {
        Material.METAL:         (190, 190, 190),
        Material.GLASS:         ( 80, 220, 255),
        Material.PLASTIC:       (255, 200,  50),
        Material.GENERAL_WASTE: ( 60,  60, 230),
    }
    calibrated: dict[Material, float] = {}

    print("\n" + "=" * 60)
    print("  STEP 2/3 — COMPARTMENT ANGLE CALIBRATION")
    print("  Fine-tune with D/A (+-1 deg) or C/Z (+-5 deg).")
    print("  Press SPACE when the arm aligns with the chute.")
    print("=" * 60)

    for idx, material in enumerate(ORDER):
        target    = float(MATERIAL_TO_COMPARTMENT[material].value)
        _set_target(target)
        color     = COLORS[material]
        confirmed = False

        print(f"\n  [{idx+1}/4]  Aligning  {material.value}  (default {target:.0f} deg) ...")

        while not confirmed:
            with _frame_lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None
            if frame is None:
                cv2.waitKey(30)
                continue

            disp = frame.copy()
            h, w = disp.shape[:2]

            detected_angle, aruco_corners = detect_aruco_angle(frame)
            tape_vis = detected_angle is not None
            if tape_vis:
                cv2.polylines(disp, [aruco_corners.astype(int)], True, (0, 230, 255), 2)
                c     = aruco_corners
                top_x = int((c[0][0] + c[1][0]) / 2)
                top_y = int((c[0][1] + c[1][1]) / 2)
                bot_x = int((c[2][0] + c[3][0]) / 2)
                bot_y = int((c[2][1] + c[3][1]) / 2)
                cv2.arrowedLine(disp, (bot_x, bot_y), (top_x, top_y),
                                (0, 255, 0), 2, tipLength=0.3)

            _put_lines(disp, [
                f"STEP 2/3 — {material.value}  [{idx+1}/4]",
                f"Target : {target:6.1f} deg",
                f"Current: {current_angle_20:6.1f} deg",
                f"Marker : {'OK' if tape_vis else 'NOT DETECTED'}",
            ])
            cv2.putText(disp, "D/A=+/-1  C/Z=+/-5  SPACE=confirm",
                        (5, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                        (100, 255, 100), 1, cv2.LINE_AA)

            badge = f" {material.value} "
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            bx, by = w - bw - 10, h - bh - 10
            cv2.rectangle(disp, (bx - 3, by - 3), (bx + bw + 3, by + bh + 3), color, -1)
            cv2.putText(disp, badge, (bx, by + bh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(30) & 0xFF

            if   key == ord('d'):  target = (target + 1.0) % 360; _set_target(target)
            elif key == ord('a'):  target = (target - 1.0) % 360; _set_target(target)
            elif key == ord('c'):  target = (target + 5.0) % 360; _set_target(target)
            elif key == ord('z'):  target = (target - 5.0) % 360; _set_target(target)
            elif key == ord(' '):
                calibrated[material] = target
                print(f"  ✓  {material.value:<14s} locked at {target:.1f} deg")
                confirmed = True

        set_speed_20(SPEED_STOP)
        target_angle_20 = None
        time.sleep(0.4)

    cv2.destroyWindow(WIN)
    cv2.waitKey(1)
    calibration_mode = False

    for material, angle in calibrated.items():
        MATERIAL_TO_COMPARTMENT[material] = angle  # type: ignore[assignment]

    print("\n[CALIB] Final compartment map:")
    for mat, val in calibrated.items():
        print(f"         {mat.value:<14s}: {val:.1f} deg")


# =============================================================================
# 10. STARTUP CALIBRATION — STEP 3: Spectrometer
# =============================================================================

def calibrate_spectrometer():
    global plastic_profile, glass_profile

    pp, gp = _spec_load_calibration()
    if pp is not None and gp is not None:
        ans = input("\n[SPEC] Saved calibration found. Use it? [Y/n]: ").strip().lower()
        if ans != 'n':
            plastic_profile = pp
            glass_profile   = gp
            print("[SPEC] Using saved calibration.")
            return

    print("\n" + "=" * 60)
    print("  STEP 3/3 — SPECTROMETER CALIBRATION")
    print("  Calibrate PLASTIC first, then GLASS.")
    print("=" * 60)

    pp = None
    gp = None
    while pp is None:
        pp = _spec_calibrate_material("PLASTIC (or open-air baseline)")
    while gp is None:
        gp = _spec_calibrate_material("GLASS")

    sep = _spec_euclidean(pp["mean"], gp["mean"])
    print(f"\n  Calibration separation: {sep:.4f}")
    if sep < 0.01:
        print("  WARNING: Materials look very similar — results may be unreliable.")
    else:
        print("  Good separation — sensor can distinguish these materials.")

    _spec_save_calibration(pp, gp)
    plastic_profile = pp
    glass_profile   = gp
    print("[SPEC] Spectrometer calibration complete.\n")


# =============================================================================
# 11a. SENSOR FUSION
# =============================================================================

def fuse_results(
    vision_label:    str,
    spec_material:   Material,
    spec_confidence: float,
    weight_g:        float,
) -> tuple[Material, str, str]:
    """
    Returns (final_material, mqtt_material_str, mqtt_shape_str).
    """
    # --- Step 1: Parse vision label ------------------------------------------
    mqtt_mat_hint, mqtt_shape = VISION_LABEL_MAP.get(vision_label, _VISION_FALLBACK)
 
    print(f"\n[FUSION] {'='*48}")
    print(f"[FUSION] vision_label  = {vision_label}")
    print(f"[FUSION] spec_material = {spec_material.value}  "
          f"conf={spec_confidence:.1f}%")
    print(f"[FUSION] weight        = {weight_g:.1f} g")
    print(f"[FUSION] vision_hint   = {mqtt_mat_hint}  shape={mqtt_shape}")
    print(f"[FUSION] beam_sensor   = "
          f"{'BROKEN — liquid' if beam_sensor.is_pressed else 'CLEAR'}")
    print(f"[FUSION] {'='*48}")
 
    # --- Step 2: Tetrapack ----------------------------------------------------
    if vision_label == "carton_tetrapack":
        if weight_g > METAL_CONTAMINATION_WEIGHT_LIMIT:
            print(f"[FUSION] ✗ Tetrapack too heavy "
                  f"({weight_g:.1f} g > {METAL_CONTAMINATION_WEIGHT_LIMIT} g) "
                  f"→ GENERAL_WASTE/others")
            return Material.GENERAL_WASTE, "general", "others"
        print(f"[FUSION] ✓ Tetrapack → METAL compartment  "
              f"[MQTT: material=tetra  type=carton]")
        return Material.METAL, "tetra", "carton"
 
    # --- Step 3: Metal (inductive confirmed) ----------------------------------
    if spec_material == Material.METAL:
        if weight_g > METAL_CONTAMINATION_WEIGHT_LIMIT:
            # Contaminated metal — shape for general waste
            gw_shape = mqtt_shape if mqtt_shape in ("bottle", "can") else "others"
            print(f"[FUSION] ✗ Metal too heavy "
                  f"({weight_g:.1f} g > {METAL_CONTAMINATION_WEIGHT_LIMIT} g) "
                  f"→ GENERAL_WASTE/{gw_shape}")
            return Material.GENERAL_WASTE, "general", gw_shape
 
        # Clean metal — shape must be bottle or can for broker
        metal_shape = mqtt_shape if mqtt_shape in ("bottle", "can") else "can"
        print(f"[FUSION] ✓ Metal clean → METAL/{metal_shape}")
        return Material.METAL, "metal", metal_shape
 
    # --- Step 4: Non-metal contamination gate (beam sensor) -------------------
    if beam_sensor.is_pressed:
        gw_shape = mqtt_shape if mqtt_shape in ("bottle", "can") else "others"
        print(f"[FUSION] ✗ Beam BROKEN — liquid in {spec_material.value} container "
              f"→ GENERAL_WASTE/{gw_shape}")
        return Material.GENERAL_WASTE, "general", gw_shape
 
    # --- Step 5: Disposable cup, non-metal ------------------------------------
    if vision_label == "disposable_cup":
        print(f"[FUSION] Disposable cup, non-metal → GENERAL_WASTE/others")
        return Material.GENERAL_WASTE, "general", "others"
 
    # --- Step 6: Vision vs spectrometer agreement / weight tiebreaker ---------
    # Vision material hint is "glass" or "plastic"
    # Spectrometer result is Material.GLASS or Material.PLASTIC
    spec_str   = spec_material.value.lower()   # "glass" or "plastic"
 
    if mqtt_mat_hint == spec_str:
        # --- Agreement ---
        final = spec_material
        print(f"[FUSION] ✓ Vision ({mqtt_mat_hint}) & spectrometer ({spec_str}) agree "
              f"→ {final.value}/bottle")
    else:
        # --- Disagreement → weight tiebreaker --------------------------------
        print(f"[FUSION] ⚠ Disagreement:  vision={mqtt_mat_hint}  "
              f"spec={spec_str}  spec_conf={spec_confidence:.1f}%")
        print(f"[FUSION]   Weight tiebreaker: {weight_g:.1f} g  "
              f"threshold={PLASTIC_GLASS_WEIGHT_THRESHOLD} g")
 
        if weight_g >= PLASTIC_GLASS_WEIGHT_THRESHOLD:
            final = Material.GLASS
            print(f"[FUSION]   Weight ≥ threshold → overriding to GLASS")
        else:
            final = Material.PLASTIC
            print(f"[FUSION]   Weight < threshold → overriding to PLASTIC")
 
    mqtt_final = "glass" if final == Material.GLASS else "plastic"
    print(f"[FUSION] ✓ Final → {final.value}/{mqtt_final}/bottle")
    return final, mqtt_final, "bottle"


# =============================================================================
# 11. PARALLEL DETECTION PATHS
# =============================================================================

def path_1_vision_model() -> str:
    if cap_vision is None:
        print("[PATH 1] Vision camera not available — returning 'unknown'")
        return "unknown"
    print(f"[PATH 1] Starting vision inference on camera index {VISION_CAM_INDEX}...")

    buffer      = []
    frame_count = 0
    t_start     = time.perf_counter()

    startx = VISION_FRAME_W  // 2 - VISION_CAPTURE_BOX_SIZE // 2
    starty = VISION_FRAME_H  // 2 - VISION_CAPTURE_BOX_SIZE // 2

    while True:
        elapsed = time.perf_counter() - t_start

        if elapsed >= VISION_TIMEOUT_S:
            if buffer:
                ids    = [x[0] for x in buffer]
                counts = Counter(ids)
                best_cls, freq = counts.most_common(1)[0]
                scores = [x[1] for x in buffer if x[0] == best_cls]
                avg_c  = sum(scores) / len(scores)
                label  = CLASS_NAMES[best_cls] if best_cls < len(CLASS_NAMES) \
                         else f"class_{best_cls}"
                print(f"\n[PATH 1] ⚠ Timeout after {elapsed:.1f} s — "
                      f"best guess: {label}  freq={freq}/{len(buffer)}  "
                      f"avg_conf={avg_c:.2%}")
                return label
            print(f"\n[PATH 1] ⚠ Timeout with empty buffer — returning 'unknown'")
            return "unknown"

        ret, frame = cap_vision.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_count += 1

        roi         = frame[starty:starty + VISION_CAPTURE_BOX_SIZE,
                            startx:startx + VISION_CAPTURE_BOX_SIZE]
        roi_resized = cv2.resize(roi, (VISION_MODEL_INPUT_SIZE, VISION_MODEL_INPUT_SIZE))
        img         = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

        class_id, confidence, inf_time_ms, probabilities = run_inference(
            quantized=VISION_QUANTIZED,
            inference_engine=vision_model,
            input_details=vision_input_details,
            output_details=vision_output_details,
            img_crop=img,
        )

        buffer.append((class_id, confidence, probabilities))
        if len(buffer) > VISION_BUFFER_THRESHOLD:
            buffer.pop(0)

        label_now = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) \
                    else f"class_{class_id}"
        print(f"[PATH 1] Frame {frame_count:4d} | {label_now:<20} {confidence:.2f}",
              end="\r")

        if len(buffer) == VISION_BUFFER_THRESHOLD:
            ids    = [x[0] for x in buffer]
            counts = Counter(ids)
            most_common_cls, freq = counts.most_common(1)[0]
            scores = [x[1] for x in buffer if x[0] == most_common_cls]
            avg_confidence = sum(scores) / len(scores)

            if freq >= VISION_FREQ_THRESHOLD and avg_confidence >= VISION_CONF_THRESHOLD:
                label = CLASS_NAMES[most_common_cls] if most_common_cls < len(CLASS_NAMES) \
                        else f"class_{most_common_cls}"

                sorted_scores = sorted(scores)
                median_conf   = sorted_scores[len(sorted_scores) // 2]
                avg_probs     = np.mean([x[2] for x in buffer], axis=0)
                ranked        = sorted(enumerate(avg_probs),
                                       key=lambda x: x[1], reverse=True)

                print(f"\n[PATH 1] {'='*50}")
                print(f"[PATH 1] ✅ DETECTED: {label}")
                print(f"[PATH 1] {'='*50}")
                print(f"[PATH 1] Frames to detection : {frame_count}")
                print(f"[PATH 1] Time to detection   : {elapsed:.2f} s")
                print(f"[PATH 1] Freq in buffer      : {freq}/{VISION_BUFFER_THRESHOLD}")
                print(f"[PATH 1] Avg confidence      : {avg_confidence:.2%}")
                print(f"[PATH 1] Min confidence      : {min(scores):.2%}")
                print(f"[PATH 1] Max confidence      : {max(scores):.2%}")
                print(f"[PATH 1] Median confidence   : {median_conf:.2%}")

                print(f"[PATH 1] Full buffer at trigger:")
                for i, (cid, conf, probs) in enumerate(buffer):
                    margin = sorted(probs, reverse=True)[0] - sorted(probs, reverse=True)[1]
                    cname  = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
                    print(f"           [{i+1:2d}] {cname:<20} conf={conf:.2f}  "
                          f"margin={margin:.2f}")

                print(f"[PATH 1] Final probabilities (averaged across buffer):")
                for cid, prob in ranked:
                    cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
                    bar   = "█" * int(prob * 40)
                    print(f"           {cname:<20} {prob:.2%}  {bar}")

                return label


def path_2_material_detection() -> tuple[Material, float]:
    print("[PATH 2] Starting material detection...")
 
    inductive_raw = metal_sensor.value
    print(f"[PATH 2] Inductive sensor raw value = {inductive_raw}  "
          f"({'TRIGGERED — metal' if inductive_raw == 0 else 'clear — non-metal'})")
 
    if inductive_raw == 0:
        print("[PATH 2] ✓ SN04-N → METAL confirmed")
        return Material.METAL, 100.0
 
    print("[PATH 2] Non-metal — querying spectrometer...")
    if not SPECTROMETER_READY:
        print("[PATH 2] ✗ Spectrometer not ready → defaulting to PLASTIC  conf=0")
        return Material.PLASTIC, 0.0
    if plastic_profile is None or glass_profile is None:
        print("[PATH 2] ✗ Calibration profiles missing → defaulting to PLASTIC  conf=0")
        return Material.PLASTIC, 0.0
 
    print(f"[PATH 2] Profiles loaded  "
          f"(plastic n={plastic_profile['n']}  glass n={glass_profile['n']})")
 
    result, confidence, debug = _spec_scan_and_classify(plastic_profile, glass_profile, samples=SPEC_SCAN_SAMPLES)
 
    if result is None:
        print("[PATH 2] ✗ Spectrometer classification failed → defaulting to PLASTIC  conf=0")
        return Material.PLASTIC, 0.0
 
    flag = " ⚠ LOW CONFIDENCE" if confidence < SPEC_LOW_CONFIDENCE_THRESHOLD else " ✓"
    print(
        f"[PATH 2] ✓ Spectrometer → {result.value}{flag}  "
        f"confidence={confidence:.1f}%  |  "
        f"sam_p={debug['sam_plastic']:.2f}°  sam_g={debug['sam_glass']:.2f}°  "
        f"chi_p={debug['chi_plastic']:.4f}  chi_g={debug['chi_glass']:.4f}  |  "
        f"samples_used={debug['samples_used']}  dropped={debug['samples_dropped']}"
    )
    return result, confidence


# =============================================================================
# 12. MAIN PIPELINE
# =============================================================================

def main_pipeline():
    global target_angle_20, outbound_direction, is_homing
    global plastic_profile, glass_profile

    hx.reset()
    tare_scale()
    calibrate_ratio()
    # calibrate_weight_thresholds()

    # calibrate_center_point()

    threading.Thread(target=servo_tracking_daemon, daemon=True).start()
    time.sleep(0.4)

    # calibrate_compartment_angles()

    if SPECTROMETER_READY:
        calibrate_spectrometer()
    else:
        print("[SYSTEM] Spectrometer not available — skipping step 3.")

    set_angle_instant_21(0)


    print("\n" + "=" * 60)
    print("  ALL CALIBRATION COMPLETE")
    print("  SYSTEM ONLINE — AWAITING OBJECT")
    print("=" * 60 + "\n")

    try:
        while True:
            weight = get_weight()

            if weight > WEIGHT_TRIGGER_THRESHOLD and target_angle_20 is None:
                print("\n" + "=" * 60)
                print(f"[TRIGGER] Object detected!")
                print(f"          Weight      : {weight:.1f} g  "
                      f"(threshold={WEIGHT_TRIGGER_THRESHOLD} g)")
                print(f"          Beam sensor  : "
                      f"{'BROKEN' if beam_sensor.is_pressed else 'CLEAR'}")
                print(f"          Inductive    : "
                      f"{'TRIGGERED' if metal_sensor.value == 0 else 'CLEAR'}")
                print("=" * 60)

                print("[PIPELINE] Waiting for scale to settle...")
                time.sleep(2.0)
                weight = get_weight()
                print(f"[PIPELINE] Settled weight: {weight:.1f} g")

                print("\n[PIPELINE] Launching parallel detection paths...")
                _tracking_pause.set()   # pause daemon — frees CPU for HX711 + inference
                t_start = time.time()
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                        fv = ex.submit(path_1_vision_model)
                        fm = ex.submit(path_2_material_detection)
                        vision_result              = fv.result()
                        material_result, spec_conf = fm.result()
                except Exception as e:
                    print(f"[PIPELINE] ✗ Detection path failed: {e} — defaulting to GENERAL_WASTE/others")
                    _tracking_pause.clear()
                    _set_target(float(MATERIAL_TO_COMPARTMENT[Material.GENERAL_WASTE].value))
                    continue

                # Guard None return from vision (timeout with empty buffer)
                if vision_result is None or vision_result == "unknown":
                    print(f"[PIPELINE] ✗ Vision returned '{vision_result}' — defaulting to GENERAL_WASTE/others")
                    mqtt_publish_result(Material.GENERAL_WASTE, "general", "others", weight)
                    _tracking_pause.clear()
                    _set_target(float(MATERIAL_TO_COMPARTMENT[Material.GENERAL_WASTE].value))
                    continue

                t_elapsed = time.time() - t_start
                print(f"[PIPELINE] Both paths complete in {t_elapsed:.2f} s")
 
                print("\n" + "-" * 50)
                print("  SENSOR FUSION & CLASSIFICATION")
                print("-" * 50)
                print(f"  Path 1 — Vision   : {vision_result}")
                print(f"  Path 2 — Material : {material_result.value}  "
                      f"conf={spec_conf:.1f}%")
                print(f"  Weight            : {weight:.1f} g")
 
                try:
                    final_decision, mqtt_mat_str, mqtt_shape_str = fuse_results(
                        vision_result, material_result, spec_conf, weight
                    )
                except Exception as e:
                    print(f"[PIPELINE] ✗ Fusion failed: {e} — defaulting to GENERAL_WASTE/others")
                    final_decision  = Material.GENERAL_WASTE
                    mqtt_mat_str    = "general"
                    mqtt_shape_str  = "others"
 
                compartment = MATERIAL_TO_COMPARTMENT.get(
                    final_decision, MATERIAL_TO_COMPARTMENT[Material.GENERAL_WASTE]
                )
                target_deg = compartment if isinstance(compartment, float) \
                             else float(compartment.value)
 
                print(f"\n[PIPELINE] ══════════════════════════════")
                print(f"[PIPELINE]  FINAL DECISION : {final_decision.value}")
                print(f"[PIPELINE]  MQTT PAYLOAD   : material={mqtt_mat_str}  "
                      f"type={mqtt_shape_str}")
                print(f"[PIPELINE]  COMPARTMENT    : {target_deg:.1f}°")
                print(f"[PIPELINE] ══════════════════════════════")
 
                mqtt_publish_result(final_decision, mqtt_mat_str, mqtt_shape_str, weight)
                _tracking_pause.clear()   # resume daemon so arm can move
                _set_target(target_deg)
                print(f"[PIPELINE] Arm in motion — waiting for cycle to complete...")

                while target_angle_20 is not None:
                    time.sleep(0.5)

                print(f"[PIPELINE] Cycle complete. Resuming idle.\n")

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutdown requested.")

    finally:
        set_speed_20(SPEED_STOP)
        pwm20.stop()
        pwm21.stop()
        GPIO.cleanup()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        cap_tracking.release()
        cap_vision.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Goodbye.")


if __name__ == "__main__":
    main_pipeline()