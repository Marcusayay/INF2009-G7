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
    METAL         = 37
    GLASS         = 100
    PLASTIC       = 222
    GENERAL_WASTE = 293

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
RATIO  = 492.22
OFFSET = 0
WEIGHT_TRIGGER_THRESHOLD = 5.0

METAL_CONTAMINATION_WEIGHT_LIMIT = 500.0

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
SPEC_SCAN_SAMPLES             = 6
SPEC_LOW_CONFIDENCE_THRESHOLD = 15
SPEC_CALIB_FILE               = "calibration.json"
SPEC_EXCLUDE_CHANNELS         = {'flicker', 'clear'}

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
SPEED_BWD, SPEED_FWD, SPEED_NEUTRAL, SPEED_STOP = 8.5, 6.5, 7.5, 0
DECEL_START, DECEL_NEAR, CRAWL_FACTOR            = 60, 25, 0.5
ANGLE_TOLERANCE                                  = 12

last_angle_21      = -1
current_angle_20   = 0.0
current_speed_20   = -1.0
target_angle_20    = None
outbound_direction = None
is_homing          = False
HOME_ANGLE         = 293
calibration_mode   = False

_latest_frame = None
_frame_lock   = threading.Lock()

center_x, center_y = 169, 113


# =============================================================================
# 1b. CAMERAS
#
#   cap_tracking  (index 0) — servo tracking daemon, 320×240
#   cap_vision    (index 1) — MobileNet inference,   640×480
#
# Two completely separate VideoCapture objects — neither path blocks the other.
# Adjust VISION_CAM_INDEX if your second camera appears at a different index.
# =============================================================================

# -- Tracking camera (rotation arm) --
cap_tracking = cv2.VideoCapture(0)
cap_tracking.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap_tracking.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap_tracking.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap_tracking.isOpened():
    print("[ERROR] Cannot open tracking camera (index 0). Check /dev/video0.")
    sys.exit(1)

# -- Vision camera (object classification) --
VISION_CAM_INDEX = 2          # change to 2 etc. if needed
cap_vision = cv2.VideoCapture(VISION_CAM_INDEX)
cap_vision.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap_vision.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_vision.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap_vision.isOpened():
    print(f"[ERROR] Cannot open vision camera (index {VISION_CAM_INDEX}). "
          "Check /dev/video1.")
    sys.exit(1)


# =============================================================================
# 1c. VISION MODEL — loaded once at startup, shared by path_1_vision_model()
# =============================================================================

# Crop & model params — kept identical to detect.py
VISION_FRAME_W, VISION_FRAME_H = 640, 480
VISION_CAPTURE_BOX_SIZE        = 320
VISION_MODEL_INPUT_SIZE        = 224
VISION_QUANTIZED               = True

# Detection thresholds — kept identical to detect.py
VISION_BUFFER_THRESHOLD = 10
VISION_FREQ_THRESHOLD   = 7
VISION_CONF_THRESHOLD   = 0.89

# Hard timeout: give up waiting for a confident detection after this many seconds
VISION_TIMEOUT_S = 15.0

print("[VISION] Loading MobileNet model...")
latest_model_path = return_latest_version_path("mobilenet")
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

MQTT_BROKER = "10.127.71.107"   # change to your broker IP if needed
MQTT_PORT   = 1883
MQTT_TOPIC  = "pi/raw_transaction"

# Maps the internal Material enum to the lowercase string the dashboard expects
_MATERIAL_TO_MQTT = {
    Material.METAL:         "metal",
    Material.GLASS:         "glass",
    Material.PLASTIC:       "plastic",
    Material.GENERAL_WASTE: "general",
}

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
    mqtt_client.loop_start()   # background thread handles ping / reconnect
    print(f"[MQTT]  Connecting to {MQTT_BROKER}:{MQTT_PORT} ...")
except Exception as e:
    print(f"[MQTT]  WARNING: Could not connect to broker ({e}). Publishing will be skipped.")

def mqtt_publish_result(final_material, vision_label, weight_g):
    """
    Publish one classification result to MQTT.

    Payload matches publisher_test.py structure:
        { "material": "plastic", "type": "bottle", "weight": "123.0g" }

    'type' comes from the vision model's predicted class name (e.g. 'bottle', 'can').
    """
    payload = {
        "material": _MATERIAL_TO_MQTT.get(final_material, "general"),
        "type":     vision_label.lower(),
        "weight":   f"{weight_g:.1f}g",
    }
    try:
        result = mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
        result.wait_for_publish(timeout=2.0)
        print(f"[MQTT]  Published → {payload}")
    except Exception as e:
        print(f"[MQTT]  Publish failed: {e}")


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
    """Drop outliers via 1.5×IQR rule, then return the clean cluster."""
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
    print("[SCALE]  Taring — ensure platform is empty...")
    time.sleep(1)
    vals = hx.get_raw_data(20)
    if vals:
        clean   = _iqr_clean(vals)
        OFFSET  = int(sum(clean) / len(clean))
        dropped = len(vals) - len(clean)
        print(f"[SCALE]  Tare complete.  IQR mean={OFFSET}  "
              f"dropped={dropped}/{len(vals)}  RATIO={RATIO}")
    else:
        print("[SCALE]  Tare FAILED — no data from HX711")

def get_weight() -> float:
    vals = hx.get_raw_data(20)
    if not vals:
        return 0.0
    clean  = _iqr_clean(vals)
    return (sum(clean) / len(clean) - OFFSET) / RATIO


# =============================================================================
# 5. CONTAMINATION CHECK — material-aware
# =============================================================================

def is_contaminated(material: Material, weight: float) -> bool:
    """
    Returns True if the object should be routed to General Waste.

    Metal         — contaminated if weight exceeds METAL_CONTAMINATION_WEIGHT_LIMIT.
    Glass/Plastic — contaminated if the break-beam sensor is triggered
                    (beam broken = liquid/water detected inside the container).
    General Waste — never re-evaluated; passes through as-is.
    """
    print(f"[CONTAM] Checking {material.value}  weight={weight:.1f} g  "
          f"beam={'BROKEN' if beam_sensor.is_pressed else 'CLEAR'}")

    if material == Material.METAL:
        if weight > METAL_CONTAMINATION_WEIGHT_LIMIT:
            print(f"[CONTAM] ✗ METAL too heavy "
                  f"({weight:.1f} g > {METAL_CONTAMINATION_WEIGHT_LIMIT} g) → General Waste")
            return True
        print(f"[CONTAM] ✓ METAL weight OK ({weight:.1f} g ≤ {METAL_CONTAMINATION_WEIGHT_LIMIT} g)")
        return False

    if material in (Material.GLASS, Material.PLASTIC):
        if beam_sensor.is_pressed:
            print(f"[CONTAM] ✗ Beam BROKEN — liquid detected in "
                  f"{material.value} container → General Waste")
            return True
        print(f"[CONTAM] ✓ Beam CLEAR — {material.value} container is clean")
        return False

    print(f"[CONTAM] ✓ {material.value} — no contamination check required")
    return False


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

def _spec_weighted_dist(scan_fp, profile):
    mean, std = profile["mean"], profile["std"]
    common    = set(scan_fp) & set(mean)
    return math.sqrt(sum(
        ((scan_fp[k] - mean[k]) / (std[k] + 1e-6)) ** 2 for k in common
    ))

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

def _spec_confidence(dp, dg, pp, gp):
    spread = _spec_euclidean(pp["mean"], gp["mean"])
    if spread < 1e-6:
        return 0.0
    prox   = max(0.0, 1.0 - min(dp, dg) / spread)
    margin = min(1.0, abs(dp - dg) / spread)
    return (prox * 0.6 + margin * 0.4) * 100

def _spec_scan_and_classify(pp, gp, samples=SPEC_SCAN_SAMPLES):
    print(f"[SPEC]   Starting scan — collecting {samples} fingerprints...")
    fps = []
    for i in range(samples):
        fp = _spec_get_fingerprint()
        if fp:
            fps.append(fp)
            print(f"[SPEC]   Sample {i+1}/{samples} collected  (channels={len(fp)})")
        else:
            print(f"[SPEC]   Sample {i+1}/{samples} FAILED — skipped")
        time.sleep(0.08)
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
    dp   = _spec_weighted_dist(scan, pp)
    dg   = _spec_weighted_dist(scan, gp)
    conf = _spec_confidence(dp, dg, pp, gp)
    result = Material.PLASTIC if dp <= dg else Material.GLASS
    winner = "PLASTIC" if dp <= dg else "GLASS"
    loser  = "GLASS"   if dp <= dg else "PLASTIC"
    margin = abs(dp - dg)
    print(f"[SPEC]   d_plastic={dp:.4f}  d_glass={dg:.4f}  "
          f"margin={margin:.4f}  confidence={conf:.1f}%")
    print(f"[SPEC]   → {winner} wins over {loser}")
    debug = {"d_plastic": dp, "d_glass": dg,
             "samples_used": len(clean), "samples_dropped": dropped}
    return result, conf, debug


# =============================================================================
# 7. SERVO TRACKING DAEMON  (uses cap_tracking — camera index 0)
# =============================================================================

def servo_tracking_daemon():
    global current_angle_20, target_angle_20, outbound_direction, is_homing
    global _latest_frame

    last_tape_seen   = time.time()
    _last_log_time   = 0.0
    _last_tape_state = None

    while True:
        ret, frame = cap_tracking.read()
        if not ret:
            time.sleep(0.01)
            continue

        with _frame_lock:
            _latest_frame = frame.copy()

        corrected = correct_frame(frame)
        hsv   = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 80, 45]),   np.array([10,  255, 90]))
        mask2 = cv2.inRange(hsv, np.array([170, 80, 45]), np.array([179, 255, 90]))
        mask  = mask1 | mask2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        tape_found = False
        tape_area  = 0
        if contours:
            largest   = max(contours, key=cv2.contourArea)
            tape_area = cv2.contourArea(largest)
            if tape_area > 250:
                tape_found     = True
                last_tape_seen = time.time()
                tip_x, tip_y   = max(
                    (pt[0] for pt in largest),
                    key=lambda p: (p[0] - center_x) ** 2 + (p[1] - center_y) ** 2
                )
                current_angle_20 = math.degrees(
                    math.atan2(center_y - tip_y, tip_x - center_x)
                ) % 360

        if tape_found != _last_tape_state:
            if tape_found:
                print(f"[DAEMON] Tape ACQUIRED  area={tape_area:.0f}  "
                      f"angle={current_angle_20:.1f}°")
            else:
                print(f"[DAEMON] Tape LOST  (largest_area={tape_area:.0f})")
            _last_tape_state = tape_found

        if not tape_found and (time.time() - last_tape_seen) > 2.0 \
                and target_angle_20 is not None:
            print("[DAEMON] Tape lost >2 s with active target — stopping motor")
            set_speed_20(SPEED_STOP)

        if target_angle_20 is not None and tape_found:
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
                    print("[ARM]    Tilting up...")
                    set_angle_instant_21(100)
                    time.sleep(1.0)
                    print("[ARM]    Tilting down...")
                    set_angle_instant_21(0)
                    time.sleep(1.0)
                    print(f"[DAEMON] Starting HOME sequence → {HOME_ANGLE}°  direction=CCW")
                    came_from_general_waste = (
                        abs(current_angle_20 - float(Compartment.GENERAL_WASTE.value))
                        <= ANGLE_TOLERANCE
                    )
                    outbound_direction = SPEED_FWD if came_from_general_waste else SPEED_BWD
                    target_angle_20    = HOME_ANGLE
                    is_homing          = True
                    for _ in range(10):
                        cap_tracking.read()
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
        ret, frame = cap_tracking.read()   # calibration reads from tracking camera
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

            _draw_crosshair(disp, center_x, center_y, (0, 255, 60), r=6)

            corrected = correct_frame(frame)
            hsv   = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 80, 45]),   np.array([10,  255, 90]))
            mask2 = cv2.inRange(hsv, np.array([170, 80, 45]), np.array([179, 255, 90]))
            mask  = mask1 | mask2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            conts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            tape_vis = False
            if conts:
                largest = max(conts, key=cv2.contourArea)
                if cv2.contourArea(largest) > 250:
                    tape_vis = True
                    tip = max(
                        (pt[0] for pt in largest),
                        key=lambda p: (p[0] - center_x) ** 2 + (p[1] - center_y) ** 2
                    )
                    cv2.circle(disp, tuple(tip), 7, (0, 0, 220), -1)
                    cv2.line(disp, (center_x, center_y), tuple(tip), (0, 230, 255), 2)

            rad = math.radians(target)
            ex  = int(center_x + 75 * math.cos(rad))
            ey  = int(center_y - 75 * math.sin(rad))
            cv2.arrowedLine(disp, (center_x, center_y), (ex, ey), color, 2, tipLength=0.25)

            _put_lines(disp, [
                f"STEP 2/3 — {material.value}  [{idx+1}/4]",
                f"Target : {target:6.1f} deg",
                f"Current: {current_angle_20:6.1f} deg",
                f"Tape   : {'OK' if tape_vis else 'NOT DETECTED'}",
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
#
# Edit ONLY this function to change how vision + material + weight are combined
# into a final routing decision. The pipeline just calls fuse_results() and
# acts on what it returns — nothing else needs to change.
#
# Inputs:
#   vision_label    (str)      — class name from the vision model, e.g. "bottle"
#   material        (Material) — material type from sensors (inductive / spectrometer)
#   weight_g        (float)    — measured weight in grams
#
# Returns:
#   final_material  (Material) — the material that will be routed to its compartment
# =============================================================================

def fuse_results(vision_label: str, material: Material, weight_g: float) -> Material:
    """
    Decide the final material category given all sensor inputs.

    Current logic (simple contamination gate):
      1. Run the contamination check for the sensor-detected material.
      2. If contaminated → route to GENERAL_WASTE regardless of vision.
      3. Otherwise      → trust the sensor material as-is.

    To change fusion behaviour (e.g. use vision to override sensor, add
    confidence weighting, introduce new rules), edit only this function.
    """
    print(f"[FUSION] vision={vision_label}  material={material.value}  weight={weight_g:.1f} g")

    # --- Contamination gate ---
    if is_contaminated(material, weight_g):
        print(f"[FUSION] ✗ Contaminated → GENERAL_WASTE")
        return Material.GENERAL_WASTE

    # --- Current rule: sensor material wins ---
    print(f"[FUSION] ✓ Clean → {material.value}")
    return material



# =============================================================================
# 11. PARALLEL DETECTION PATHS
# =============================================================================

def path_1_vision_model() -> str:
    """
    Runs MobileNet inference on the dedicated vision camera (cap_vision, index 1).

    Mirrors the buffer loop from detect.py exactly:
      - Captures frames directly from cap_vision (640×480)
      - Crops a centred VISION_CAPTURE_BOX_SIZE square, resizes to 224×224
      - Rolling window of VISION_BUFFER_THRESHOLD frames
      - Triggers when dominant class appears ≥ VISION_FREQ_THRESHOLD times
        AND its average confidence ≥ VISION_CONF_THRESHOLD
      - Falls back to best available class after VISION_TIMEOUT_S seconds

    Returns the predicted class name string.
    """
    print(f"[PATH 1] Starting vision inference on camera index {VISION_CAM_INDEX}...")

    buffer      = []
    frame_count = 0
    t_start     = time.perf_counter()

    # Centred crop coordinates — computed once for the 640×480 source frame
    startx = VISION_FRAME_W  // 2 - VISION_CAPTURE_BOX_SIZE // 2
    starty = VISION_FRAME_H  // 2 - VISION_CAPTURE_BOX_SIZE // 2

    while True:
        elapsed = time.perf_counter() - t_start

        # ---- Timeout fallback ------------------------------------------------
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

        # ---- Capture from vision camera --------------------------------------
        ret, frame = cap_vision.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_count += 1

        # ---- Preprocess (identical to detect.py) ----------------------------
        roi         = frame[starty:starty + VISION_CAPTURE_BOX_SIZE,
                            startx:startx + VISION_CAPTURE_BOX_SIZE]
        roi_resized = cv2.resize(roi, (VISION_MODEL_INPUT_SIZE, VISION_MODEL_INPUT_SIZE))
        img         = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

        # ---- Inference -------------------------------------------------------
        class_id, confidence, inf_time_ms, probabilities = run_inference(
            quantized=VISION_QUANTIZED,
            inference_engine=vision_model,
            input_details=vision_input_details,
            output_details=vision_output_details,
            img_crop=img,
        )

        # ---- Rolling buffer --------------------------------------------------
        buffer.append((class_id, confidence, probabilities))
        if len(buffer) > VISION_BUFFER_THRESHOLD:
            buffer.pop(0)

        label_now = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) \
                    else f"class_{class_id}"
        print(f"[PATH 1] Frame {frame_count:4d} | {label_now:<20} {confidence:.2f}",
              end="\r")

        # ---- Evaluate buffer -------------------------------------------------
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


def path_2_material_detection() -> Material:
    print("[PATH 2] Starting material detection...")

    inductive_raw = metal_sensor.value
    print(f"[PATH 2] Inductive sensor raw value = {inductive_raw}  "
          f"({'TRIGGERED — metal' if inductive_raw == 0 else 'clear — non-metal'})")

    if inductive_raw == 0:
        print("[PATH 2] ✓ SN04-N → METAL confirmed")
        return Material.METAL

    print("[PATH 2] Non-metal — querying spectrometer...")
    if not SPECTROMETER_READY:
        print("[PATH 2] ✗ Spectrometer not ready → defaulting to PLASTIC")
        return Material.PLASTIC
    if plastic_profile is None or glass_profile is None:
        print("[PATH 2] ✗ Calibration profiles missing → defaulting to PLASTIC")
        return Material.PLASTIC

    print(f"[PATH 2] Profiles loaded  "
          f"(plastic n={plastic_profile['n']}  glass n={glass_profile['n']})")

    result, confidence, debug = _spec_scan_and_classify(plastic_profile, glass_profile)

    if result is None:
        print("[PATH 2] ✗ Spectrometer classification failed → defaulting to PLASTIC")
        return Material.PLASTIC

    flag = " ⚠ LOW CONFIDENCE" if confidence < SPEC_LOW_CONFIDENCE_THRESHOLD else " ✓"
    print(
        f"[PATH 2] ✓ Spectrometer → {result.value}{flag}  "
        f"confidence={confidence:.1f}%  |  "
        f"d_plastic={debug['d_plastic']:.4f}  d_glass={debug['d_glass']:.4f}  |  "
        f"samples_used={debug['samples_used']}  dropped={debug['samples_dropped']}"
    )
    return result


# =============================================================================
# 12. MAIN PIPELINE
# =============================================================================

def main_pipeline():
    global target_angle_20, outbound_direction, is_homing
    global plastic_profile, glass_profile

    hx.reset()
    tare_scale()

    calibrate_center_point()

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

                print("\n[PIPELINE] Launching parallel detection paths...")
                t_start = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fv = ex.submit(path_1_vision_model)
                    fm = ex.submit(path_2_material_detection)
                    vision_result   = fv.result()
                    material_result = fm.result()
                t_elapsed = time.time() - t_start
                print(f"[PIPELINE] Both paths complete in {t_elapsed:.2f} s")

                print("\n" + "-" * 50)
                print("  SENSOR FUSION & CLASSIFICATION")
                print("-" * 50)
                print(f"  Path 1 — Vision   : {vision_result}")
                print(f"  Path 2 — Material : {material_result.value}")
                print(f"  Weight            : {weight:.1f} g")

                final_decision = fuse_results(vision_result, material_result, weight)

                compartment = MATERIAL_TO_COMPARTMENT.get(
                    final_decision, MATERIAL_TO_COMPARTMENT[Material.GENERAL_WASTE]
                )
                target_deg = compartment if isinstance(compartment, float) \
                             else float(compartment.value)

                print(f"\n[PIPELINE] ══════════════════════════════")
                print(f"[PIPELINE]  FINAL DECISION : {final_decision.value}")
                print(f"[PIPELINE]  COMPARTMENT    : {target_deg:.1f}°")
                print(f"[PIPELINE] ══════════════════════════════")

                mqtt_publish_result(final_decision, vision_result, weight)
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