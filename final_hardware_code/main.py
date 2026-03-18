import time
import sys
import math
import json
import os
import cv2
import numpy as np
import RPi.GPIO as GPIO
import concurrent.futures
import threading

# --- Sensor Imports ---
from gpiozero import DigitalInputDevice
from hx711 import HX711
from as7343 import AS7343

# =============================================================================
# 1. HARDWARE CONFIGURATION & INITIALIZATION
# =============================================================================

# -- Weight Sensor (HX711) --
hx = HX711(dout_pin=15, pd_sck_pin=14)
RATIO = 492.22
OFFSET = 0
WEIGHT_TRIGGER_THRESHOLD  = 15.0    # grams to trigger pipeline
CONTAMINATION_WEIGHT_LIMIT = 500.0  # grams; above this → CONTAMINATED

# -- Inductive Sensor (SN04-N) --
PIN_INDUCTIVE = 16
metal_sensor = DigitalInputDevice(PIN_INDUCTIVE, pull_up=True)

# -- Spectrometer (AS7343) --
SPEC_GAIN                     = 2048
SPEC_INTEGRATION_TIME         = 200
SPEC_CALIBRATION_SAMPLES      = 12
SPEC_SCAN_SAMPLES             = 6
SPEC_LOW_CONFIDENCE_THRESHOLD = 15   # % — warn below this
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

plastic_profile = None  # populated at startup
glass_profile   = None

# -- Servos --
PIN_20 = 20   # Base rotation (continuous-rotation motor)
PIN_21 = 21   # Arm actuator  (positional servo)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(PIN_20, GPIO.OUT)
pwm20 = GPIO.PWM(PIN_20, 50)
pwm20.start(0)
GPIO.setup(PIN_21, GPIO.OUT)
pwm21 = GPIO.PWM(PIN_21, 50)
pwm21.start(0)

MOTOR_DIRECTION_SIGN = 1
SPEED_FWD, SPEED_BWD, SPEED_NEUTRAL, SPEED_STOP = 9.0, 6.0, 7.5, 0
DECEL_START, DECEL_NEAR, CRAWL_FACTOR, ANGLE_TOLERANCE = 60, 25, 0.20, 7

last_angle_21    = -1
current_angle_20 = 0.0
current_speed_20 = -1.0
target_angle_20  = None
outbound_direction = None
is_homing        = False
HOME_ANGLE       = 90

# calibration_mode: when True the daemon parks at target but does NOT
# actuate the arm or home — lets the human inspect compartment alignment.
calibration_mode = False

# -- Camera & shared frame buffer --
# ONLY the servo daemon calls cap.read(); calibration UIs read _latest_frame.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

_latest_frame = None
_frame_lock   = threading.Lock()

# Rotation centre — set interactively in STEP 1 of startup calibration
center_x, center_y = 160, 120

# Compartment angles — overwritten during STEP 2 of startup calibration
COMPARTMENTS = {
    "METAL":        62,
    "GLASS":       131,
    "PLASTIC":     249,
    "CONTAMINATED": 295,
}


# =============================================================================
# 2. SERVO HELPERS
# =============================================================================

def set_speed_20(speed):
    global current_speed_20
    if current_speed_20 != speed:
        pwm20.ChangeDutyCycle(speed)
        current_speed_20 = speed

def set_angle_instant_21(angle):
    global last_angle_21
    if angle != last_angle_21:
        duty = round(float(angle) / 18.0 + 2.0, 2)
        pwm21.ChangeDutyCycle(duty)
        time.sleep(0.5)
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
    """Point the servo daemon at a new angle (thread-safe write)."""
    global target_angle_20, outbound_direction, is_homing
    diff = shortest_angle_diff(current_angle_20, float(angle))
    outbound_direction = pick_direction(diff)
    target_angle_20    = float(angle) % 360
    is_homing          = False


# =============================================================================
# 3. WEIGHT HELPERS
# =============================================================================

def tare_scale():
    global OFFSET
    print("[SYSTEM] Taring scale — ensure platform is empty...")
    time.sleep(1)
    vals = hx.get_raw_data(10)
    if vals:
        vals.sort()
        OFFSET = vals[len(vals) // 2]
    print(f"[SYSTEM] Tare complete. Offset = {OFFSET}")

def get_weight():
    vals = hx.get_raw_data(5)
    if not vals:
        return 0.0
    vals.sort()
    return (vals[len(vals) // 2] - OFFSET) / RATIO


# =============================================================================
# 4. SPECTROMETER — FULL SPECTRAL FINGERPRINTING
# =============================================================================
# KEY INSIGHT: Dividing every channel by the total intensity cancels ambient
# brightness; only the spectral *shape* survives — a stable material property.
# Classification uses Mahalanobis-style distance to calibrated profiles, not
# single-ratio thresholds (far more robust across lighting conditions).

def _spec_get_fingerprint():
    """Read both sensor banks; return normalised {channel: fraction} or None."""
    combined = {}
    for bank in [0, 1]:
        spectrometer.bank_select(bank)
        time.sleep(0.30)
        try:
            for ch, val in spectrometer.get_data()[0].items():
                if ch not in SPEC_EXCLUDE_CHANNELS:
                    combined[ch] = max(float(val), 0.0)
        except Exception:
            pass
    if not combined:
        return None
    total = sum(combined.values())
    if total < 1:
        return None
    return {k: v / total for k, v in combined.items()}

def _spec_average(fps):
    keys = fps[0].keys()
    return {k: sum(fp[k] for fp in fps) / len(fps) for k in keys}

def _spec_euclidean(a, b):
    common = set(a) & set(b)
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in common))

def _spec_weighted_dist(scan_fp, profile):
    """Mahalanobis-style: stable (low-variance) channels count more."""
    mean, std = profile["mean"], profile["std"]
    common = set(scan_fp) & set(mean)
    return math.sqrt(sum(
        ((scan_fp[k] - mean[k]) / (std[k] + 1e-6)) ** 2 for k in common
    ))

def _spec_reject_outliers(fps):
    if len(fps) < 3:
        return fps, 0
    avg    = _spec_average(fps)
    dists  = [_spec_euclidean(fp, avg) for fp in fps]
    mu     = sum(dists) / len(dists)
    sigma  = math.sqrt(sum((d - mu) ** 2 for d in dists) / len(dists))
    thr    = mu + 2 * sigma
    clean  = [fp for fp, d in zip(fps, dists) if d <= thr]
    return clean, len(fps) - len(clean)

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
    """Interactive single-material calibration. Returns profile dict or None."""
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
    """Classify one object. Returns (label, confidence%, debug_dict)."""
    fps = []
    for _ in range(samples):
        fp = _spec_get_fingerprint()
        if fp:
            fps.append(fp)
        time.sleep(0.08)
    if not fps:
        return "ERROR", 0, {}
    clean, dropped = _spec_reject_outliers(fps)
    if not clean:
        return "ERROR", 0, {}
    scan   = _spec_average(clean)
    dp     = _spec_weighted_dist(scan, pp)
    dg     = _spec_weighted_dist(scan, gp)
    spread = _spec_euclidean(pp["mean"], gp["mean"])
    debug  = {"d_plastic": dp, "d_glass": dg,
              "samples_used": len(clean), "samples_dropped": dropped}
    if min(dp, dg) > spread * 1.5:
        return "UNKNOWN", 0, debug
    conf   = _spec_confidence(dp, dg, pp, gp)
    result = "PLASTIC" if dp <= dg else "GLASS"
    return result, conf, debug


# =============================================================================
# 5. SERVO TRACKING DAEMON  (background thread — runs for entire session)
# =============================================================================

def servo_tracking_daemon():
    """
    Reads camera into _latest_frame continuously.
    Tracks green tape → derives current_angle_20.
    Drives pwm20 toward target_angle_20 with smooth deceleration.
    In calibration_mode: stops at target but skips arm actuation and homing.
    """
    global current_angle_20, target_angle_20, outbound_direction, is_homing
    global _latest_frame

    last_tape_seen = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        with _frame_lock:
            _latest_frame = frame.copy()

        # ── Green tape detection ──────────────────────────────────────────────
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        tape_found = False
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                tape_found     = True
                last_tape_seen = time.time()
                tip_x, tip_y = max(
                    (pt[0] for pt in largest),
                    key=lambda p: (p[0] - center_x) ** 2 + (p[1] - center_y) ** 2
                )
                current_angle_20 = math.degrees(
                    math.atan2(center_y - tip_y, tip_x - center_x)
                ) % 360

        # Halt if tape lost > 2 s
        if not tape_found and (time.time() - last_tape_seen) > 2.0 \
                and target_angle_20 is not None:
            set_speed_20(SPEED_STOP)

        # ── Targeting / homing ────────────────────────────────────────────────
        if target_angle_20 is not None and tape_found:
            diff = shortest_angle_diff(current_angle_20, target_angle_20)
            dist = abs(diff)

            if dist <= ANGLE_TOLERANCE:
                set_speed_20(SPEED_STOP)

                if calibration_mode:
                    pass  # hold position; human is inspecting

                elif not is_homing:
                    print("\n[SERVO] Arrived. Actuating arm...")
                    set_angle_instant_21(100)  # arm up
                    set_angle_instant_21(0)    # arm down
                    home_dir = SPEED_BWD if outbound_direction == SPEED_FWD else SPEED_FWD
                    outbound_direction = home_dir
                    target_angle_20    = HOME_ANGLE
                    is_homing          = True
                    for _ in range(10):
                        cap.read()  # flush stale frames

                else:
                    print("[SERVO] Homing complete. System idle.\n")
                    target_angle_20    = None
                    outbound_direction = None
                    is_homing          = False

            else:
                live_dir = pick_direction(diff)
                if dist < DECEL_START and live_dir != outbound_direction:
                    set_speed_20(SPEED_STOP)
                else:
                    set_speed_20(decelerated_speed(outbound_direction, dist))

        time.sleep(0.01)


# =============================================================================
# 6.  STARTUP CALIBRATION — STEP 1: Rotation Centre Point (mouse click)
# =============================================================================

def _draw_crosshair(img, x, y, color=(0, 255, 0), r=10):
    cv2.circle(img, (x, y), r, color, 2)
    cv2.line(img, (x - r - 5, y), (x + r + 5, y), color, 1)
    cv2.line(img, (x, y - r - 5), (x, y + r + 5), color, 1)

def _put_lines(img, lines, y0=14, dy=16, color=(220, 220, 220), scale=0.40):
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (5, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def calibrate_center_point():
    """
    STEP 1 — Live camera window; user left-clicks the pivot / rotation centre.

      Click           → sets candidate centre point
      SPACE           → confirm and continue
      R               → reset to image centre (160, 120)
    """
    global center_x, center_y

    WIN    = "STEP 1/3 — Rotation Centre  (click = set | SPACE = confirm | R = reset)"
    cv2.namedWindow(WIN)

    clicked = [False]
    tmp     = [center_x, center_y]

    def _mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            tmp[0], tmp[1] = x, y
            clicked[0]     = True

    cv2.setMouseCallback(WIN, _mouse)

    print("\n" + "=" * 54)
    print("  STEP 1/3 — ROTATION CENTRE CALIBRATION")
    print("  Click the arm's pivot point in the camera window.")
    print("  Press SPACE to confirm.")
    print("=" * 54)

    while True:
        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None
        if frame is None:
            time.sleep(0.04)
            continue

        disp   = frame.copy()
        cx, cy = tmp

        # Concentric guide rings
        for r in [40, 80, 115]:
            cv2.circle(disp, (cx, cy), r, (30, 70, 30), 1)

        _draw_crosshair(disp, cx, cy, (0, 255, 80))

        _put_lines(disp, [
            "STEP 1/3 — Set Rotation Centre",
            f"Centre: ({cx}, {cy})",
            "Left-click = place point",
            "SPACE = confirm   R = reset",
        ])

        status_txt = "Click to set centre..." if not clicked[0] \
                     else "Point set — press SPACE to confirm"
        status_col = (0, 200, 255) if not clicked[0] else (0, 255, 100)
        cv2.putText(disp, status_txt, (5, disp.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_col, 1)

        cv2.imshow(WIN, disp)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' ') and clicked[0]:
            center_x, center_y = tmp
            print(f"[CALIB] Rotation centre → ({center_x}, {center_y})")
            break
        elif key == ord('r'):
            tmp[0], tmp[1] = 160, 120
            clicked[0]     = False

    cv2.destroyWindow(WIN)


# =============================================================================
# 7.  STARTUP CALIBRATION — STEP 2: Compartment Angles (keyboard jog)
# =============================================================================

def calibrate_compartment_angles():
    """
    STEP 2 — For each compartment the servo drives to the stored default angle.
    The user jogs the arm with the keyboard until it visually aligns with the
    corresponding chute, then confirms.

      D / A   →  +1° / −1°   (fine)
      C / Z   →  +5° / −5°   (coarse)
      SPACE   →  confirm this compartment and move to the next
    """
    global COMPARTMENTS, target_angle_20, outbound_direction
    global is_homing, calibration_mode

    calibration_mode = True

    WIN = "STEP 2/3 — Compartment Angles  (D/A=±1°  C/Z=±5°  SPACE=confirm)"
    cv2.namedWindow(WIN)

    ORDER  = ["METAL", "GLASS", "PLASTIC", "CONTAMINATED"]
    COLORS = {
        "METAL":        (190, 190, 190),
        "GLASS":        ( 80, 220, 255),
        "PLASTIC":      (255, 200,  50),
        "CONTAMINATED": ( 60,  60, 230),
    }
    calibrated = {}

    print("\n" + "=" * 54)
    print("  STEP 2/3 — COMPARTMENT ANGLE CALIBRATION")
    print("  The arm rotates to each compartment in turn.")
    print("  Fine-tune with D/A (±1°) or C/Z (±5°).")
    print("  Press SPACE when the arm aligns with the chute.")
    print("=" * 54)

    for idx, name in enumerate(ORDER):
        target = float(COMPARTMENTS[name])
        _set_target(target)
        color  = COLORS[name]
        confirmed = False

        print(f"\n  [{idx+1}/4]  Aligning  {name}  (default {target:.0f}°) ...")

        while not confirmed:
            with _frame_lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None
            if frame is None:
                time.sleep(0.03)
                continue

            disp = frame.copy()
            h, w = disp.shape[:2]

            # Centre crosshair
            _draw_crosshair(disp, center_x, center_y, (0, 255, 60), r=6)

            # Detect tape; draw arm vector
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
            conts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            tape_vis = False
            if conts:
                largest = max(conts, key=cv2.contourArea)
                if cv2.contourArea(largest) > 100:
                    tape_vis = True
                    tip = max(
                        (pt[0] for pt in largest),
                        key=lambda p: (p[0] - center_x) ** 2 + (p[1] - center_y) ** 2
                    )
                    cv2.circle(disp, tuple(tip), 7, (0, 0, 220), -1)
                    cv2.line(disp, (center_x, center_y), tuple(tip),
                             (0, 230, 255), 2)

            # Target direction arrow (in compartment colour)
            rad = math.radians(target)
            ex  = int(center_x + 75 * math.cos(rad))
            ey  = int(center_y - 75 * math.sin(rad))
            cv2.arrowedLine(disp, (center_x, center_y), (ex, ey),
                            color, 2, tipLength=0.25)

            # Angle arc (small visual arc showing target)
            cv2.ellipse(disp, (center_x, center_y), (30, 30),
                        0, -target, -target + 5, color, 2)

            # Info overlay (top-left)
            _put_lines(disp, [
                f"STEP 2/3 — {name}  [{idx+1}/4]",
                f"Target : {target:6.1f} deg",
                f"Current: {current_angle_20:6.1f} deg",
                f"Tape   : {'OK' if tape_vis else 'NOT DETECTED'}",
            ])

            # Key legend (bottom)
            cv2.putText(disp,
                        "D/A=+/-1  C/Z=+/-5  SPACE=confirm",
                        (5, h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                        (100, 255, 100), 1, cv2.LINE_AA)

            # Compartment badge (bottom-right)
            badge = f" {name} "
            (bw, bh), _ = cv2.getTextSize(
                badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            bx = w - bw - 10
            by = h - bh - 10
            cv2.rectangle(disp, (bx - 3, by - 3),
                          (bx + bw + 3, by + bh + 3), color, -1)
            cv2.putText(disp, badge, (bx, by + bh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)

            cv2.imshow(WIN, disp)
            key = cv2.waitKey(30) & 0xFF

            if   key == ord('d'):
                target = (target + 1.0) % 360
                _set_target(target)
            elif key == ord('a'):
                target = (target - 1.0) % 360
                _set_target(target)
            elif key == ord('c'):
                target = (target + 5.0) % 360
                _set_target(target)
            elif key == ord('z'):
                target = (target - 5.0) % 360
                _set_target(target)
            elif key == ord(' '):
                calibrated[name] = target
                print(f"  ✓  {name:<14s} locked at {target:.1f}°")
                confirmed = True

        # Stop between compartments
        set_speed_20(SPEED_STOP)
        target_angle_20 = None
        time.sleep(0.4)

    cv2.destroyWindow(WIN)
    calibration_mode = False
    COMPARTMENTS.update(calibrated)

    print("\n[CALIB] Final compartment map:")
    for k, v in COMPARTMENTS.items():
        print(f"         {k:<14s}: {v:.1f}°")


# =============================================================================
# 8.  STARTUP CALIBRATION — STEP 3: Spectrometer (terminal prompts)
# =============================================================================

def calibrate_spectrometer():
    """
    STEP 3 — Two-material spectrometer calibration.
    Loads a saved profile from disk if one exists; asks whether to reuse it.
    Falls back to interactive calibration otherwise.
    Updates global plastic_profile / glass_profile.
    """
    global plastic_profile, glass_profile

    pp, gp = _spec_load_calibration()
    if pp is not None and gp is not None:
        ans = input(
            "\n[SPEC] Saved calibration found. Use it? [Y/n]: "
        ).strip().lower()
        if ans != 'n':
            plastic_profile = pp
            glass_profile   = gp
            print("[SPEC] Using saved calibration.")
            return

    print("\n" + "=" * 54)
    print("  STEP 3/3 — SPECTROMETER CALIBRATION")
    print("  Calibrate PLASTIC first, then GLASS.")
    print("  Keep sensor placement consistent for both.")
    print("=" * 54)

    pp = None
    gp = None

    while pp is None:
        pp = _spec_calibrate_material("PLASTIC (or open-air baseline)")
    while gp is None:
        gp = _spec_calibrate_material("GLASS")

    sep = _spec_euclidean(pp["mean"], gp["mean"])
    print(f"\n  Calibration separation: {sep:.4f}")
    if sep < 0.01:
        print("  ⚠ WARNING: Materials look very similar to the sensor.")
        print("    Results may be unreliable — adjust lighting or reposition sensor.")
    else:
        print("  ✓ Good separation — sensor can distinguish these materials.")

    _spec_save_calibration(pp, gp)
    plastic_profile = pp
    glass_profile   = gp
    print("[SPEC] Spectrometer calibration complete.\n")


# =============================================================================
# 9.  PARALLEL DETECTION PATHS
# =============================================================================

def path_1_vision_model():
    """Parallel Path 1: Object Detection — placeholder for TF Lite / PyTorch."""
    print("[PATH 1] Running vision model...")
    time.sleep(1.5)                        # TODO: replace with real inference
    predicted_object = "Bottle"
    print(f"[PATH 1] Vision result: {predicted_object}")
    return predicted_object


def path_2_material_detection():
    """
    Parallel Path 2: Inductive → Spectrometer pipeline.
      • Metal detected by SN04-N (fast, no spectrometer needed).
      • Non-metals classified by full spectral fingerprint (Mahalanobis distance).
    """
    print("[PATH 2] Initiating material detection...")

    # ── Inductive sensor (metal check) ───────────────────────────────────────
    if metal_sensor.value == 0:   # Active-low
        print("[PATH 2] SN04-N → METAL detected.")
        return "METAL"

    # ── Spectrometer ─────────────────────────────────────────────────────────
    print("[PATH 2] Non-metal — querying spectrometer...")
    if not SPECTROMETER_READY:
        print("[PATH 2] Spectrometer unavailable → defaulting to PLASTIC.")
        return "PLASTIC"
    if plastic_profile is None or glass_profile is None:
        print("[PATH 2] Profiles not loaded → defaulting to PLASTIC.")
        return "PLASTIC"

    result, confidence, debug = _spec_scan_and_classify(plastic_profile, glass_profile)

    if result in ("ERROR", "UNKNOWN"):
        print(f"[PATH 2] Spectrometer → {result} "
              f"(dp={debug.get('d_plastic', 0):.4f} "
              f"dg={debug.get('d_glass', 0):.4f}) — defaulting to PLASTIC.")
        return "PLASTIC"

    flag = " ⚠ LOW CONFIDENCE" if confidence < SPEC_LOW_CONFIDENCE_THRESHOLD else ""
    print(
        f"[PATH 2] Spectrometer → {result}  "
        f"Confidence: {confidence:.1f}%{flag}  |  "
        f"dp={debug['d_plastic']:.4f}  dg={debug['d_glass']:.4f}  "
        f"samples={debug['samples_used']} (dropped={debug['samples_dropped']})"
    )
    return result


# =============================================================================
# 10. MAIN PIPELINE
# =============================================================================

def main_pipeline():
    global target_angle_20, outbound_direction, is_homing
    global plastic_profile, glass_profile

    # ── Launch servo / camera daemon (populates _latest_frame immediately) ────
    threading.Thread(target=servo_tracking_daemon, daemon=True).start()
    time.sleep(0.4)   # let daemon grab first frame before calibration UIs open

    # ── Scale tare (no visual dependency) ────────────────────────────────────
    hx.reset()
    tare_scale()

    # =========================================================================
    # STARTUP CALIBRATION  (3 sequential steps)
    # =========================================================================

    # STEP 1 — Set arm rotation centre via mouse click on live camera feed
    calibrate_center_point()

    # STEP 2 — Drive servo to each compartment; user adjusts with D/A/C/Z keys
    calibrate_compartment_angles()

    # STEP 3 — Two-material spectrometer calibration (terminal prompts)
    if SPECTROMETER_READY:
        calibrate_spectrometer()
    else:
        print("[SYSTEM] Spectrometer not available — skipping step 3.")

    # ── Park arm at home / ready position ────────────────────────────────────
    set_angle_instant_21(0)

    print("\n" + "=" * 54)
    print("  ALL CALIBRATION COMPLETE")
    print("  SYSTEM ONLINE — AWAITING OBJECT")
    print("=" * 54 + "\n")

    # =========================================================================
    # MAIN AUTO-SORTING LOOP
    # =========================================================================
    try:
        while True:
            weight = get_weight()

            if weight > WEIGHT_TRIGGER_THRESHOLD and target_angle_20 is None:
                print(f"\n[TRIGGER] Object detected  Weight: {weight:.1f} g")

                # ── Run detection paths in parallel ───────────────────────────
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fv = ex.submit(path_1_vision_model)
                    fm = ex.submit(path_2_material_detection)
                    vision_result   = fv.result()
                    material_result = fm.result()

                # ── Sensor fusion & final classification ──────────────────────
                print("\n--- SENSOR FUSION & CLASSIFICATION ---")
                if weight > CONTAMINATION_WEIGHT_LIMIT:
                    print(f"[!] Heavy object ({weight:.1f} g > limit) → CONTAMINATED")
                    final_decision = "CONTAMINATED"
                else:
                    # TODO: fuse vision_result for object-level overrides,
                    #       e.g. if vision_result == "Battery": → CONTAMINATED
                    final_decision = material_result

                print(f"Final Decision: {final_decision}")

                # ── Route arm to correct compartment ──────────────────────────
                target_deg = COMPARTMENTS.get(
                    final_decision, COMPARTMENTS["CONTAMINATED"]
                )
                _set_target(target_deg)
                print(f"Routing {final_decision} → {target_deg:.1f}°")

                # Block until the full deposit-and-home cycle is done
                while target_angle_20 is not None:
                    time.sleep(0.5)

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutdown requested.")

    finally:
        set_speed_20(SPEED_STOP)
        pwm20.stop()
        pwm21.stop()
        GPIO.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Goodbye.")


if __name__ == "__main__":
    main_pipeline()