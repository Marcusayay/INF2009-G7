import cv2
import numpy as np
import math
import RPi.GPIO as GPIO
import time
import subprocess
subprocess.run(["v4l2-ctl",  "-d", "/dev/video0", "--set-ctrl=auto_exposure=1"],        check=False)  # manual mode
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=exposure_time_absolute=20"], check=False)  # very short
# subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=brightness=255"],            check=False)
# subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=gain=225"],  check=False)  # try this first
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=white_balance_automatic=0"],       check=False)
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=white_balance_temperature=6500"],   check=False)
# =============================================================================
# ARUCO MARKER DETECTION  — lightweight build for Pi
#
#   Three performance optimisations vs the heavy version:
#
#   1. Camera captures at 320x240. Upscaling 320->640 for detection is
#      identical in benefit to upscaling 640->1280, but the detection image
#      is 640x480 instead of 1280x960 — 4x less work for the detector.
#
#   2. equalizeHist instead of CLAHE. Single fast pass, good enough for UV.
#      CLAHE is tile-based and noticeably slower on a Pi.
#
#   3. Frame skip — ArUco only runs every DETECT_EVERY_N frames.
#      Between detections the last known angle is held. The arm cannot
#      physically jump more than a few degrees between consecutive frames
#      so this is safe and halves detector CPU at N=2.
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

# Upscale: capture is 320x240, detection runs on 640x480
UPSCALE          = 2
# Run ArUco every N frames — holds last angle in between
DETECT_EVERY_N   = 1
_frame_counter   = 0
_last_corners    = None   # cached for overlay drawing on skipped frames


def preprocess(frame):
    """
    Upscale 2x + grayscale + equalizeHist.
    Starts from 320x240 so the detection image is 640x480 — much lighter
    than upscaling from 640x480 which produces 1280x960.
    """
    big  = cv2.resize(frame, None, fx=UPSCALE, fy=UPSCALE,
                    interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_aruco_angle(frame):
    """
    Returns the arm angle purely from the marker's own orientation.
    Top edge points toward the rotation centre, so the bottom->top
    vector IS the inward direction — no center_x/y required.

    Returns:
        (angle_degrees, marker_corners)  — or  (None, None) if not found.
        Corners are in original (320x240) frame coordinates.
    """
    gray = preprocess(frame)
    corners, ids, _ = _aruco_detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None

    # Divide corners back to original 320x240 coords
    c = corners[0][0] / UPSCALE   # TL, TR, BR, BL

    top_x    = (c[0][0] + c[1][0]) / 2
    top_y    = (c[0][1] + c[1][1]) / 2
    bottom_x = (c[2][0] + c[3][0]) / 2
    bottom_y = (c[2][1] + c[3][1]) / 2

    dx    = top_x - bottom_x
    dy    = top_y - bottom_y
    angle = math.degrees(math.atan2(-dy, dx)) % 360

    return angle, c


# =============================================================================
# GPIO CONFIGURATION
# =============================================================================
PIN_20 = 20  # 360 Continuous Servo (Base Rotation)
PIN_21 = 21  # Standard 180 Servo (Arm/Actuator)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(PIN_20, GPIO.OUT)
pwm20 = GPIO.PWM(PIN_20, 50)
pwm20.start(0)

GPIO.setup(PIN_21, GPIO.OUT)
pwm21 = GPIO.PWM(PIN_21, 50)
pwm21.start(0)

# =============================================================================
# *** CALIBRATION — TUNE THIS FIRST ***
#
# MOTOR_DIRECTION_SIGN:
#   Press W (manual CW) and watch the displayed angle number in the video.
#   If the angle INCREASES  -> set MOTOR_DIRECTION_SIGN = +1
#   If the angle DECREASES  -> set MOTOR_DIRECTION_SIGN = -1
#
MOTOR_DIRECTION_SIGN = 1
# =============================================================================

SPEED_FWD     = 9.0
SPEED_BWD     = 6.0
SPEED_NEUTRAL = 7.5
SPEED_STOP    = 0

DECEL_START     = 60
DECEL_NEAR      = 25
CRAWL_FACTOR    = 0.20
ANGLE_TOLERANCE = 7

last_angle_21      = -1
current_angle_20   = 0.0
current_speed_20   = -1.0
target_angle_20    = None
outbound_direction = None
is_homing          = False
HOME_ANGLE         = 293

last_marker_seen_time = time.time()
TRACKING_TIMEOUT      = 2.0

# Camera — 320x240, upscaling handles the resolution boost for detection
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE,     -5)
_current_exposure = -5

actual_fps = cap.get(cv2.CAP_PROP_FPS)
actual_w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"[CAM] {int(actual_w)}x{int(actual_h)} @ {actual_fps}fps")


def set_angle_instant_21(angle):
    global last_angle_21
    if angle != last_angle_21:
        duty = round(float(angle) / 18.0 + 2.0, 2)
        pwm21.ChangeDutyCycle(duty)
        time.sleep(0.5)
        pwm21.ChangeDutyCycle(0)
        last_angle_21 = angle


def set_speed_20(speed):
    global current_speed_20
    if current_speed_20 != speed:
        pwm20.ChangeDutyCycle(speed)
        current_speed_20 = speed


def shortest_angle_diff(current, target):
    return (target - current + 180) % 360 - 180


def pick_direction(diff):
    if MOTOR_DIRECTION_SIGN * diff > 0:
        return SPEED_FWD
    else:
        return SPEED_BWD


def decelerated_speed(direction, angular_distance):
    if angular_distance >= DECEL_START:
        t = 1.0
    elif angular_distance <= DECEL_NEAR:
        t = CRAWL_FACTOR
    else:
        linear = (angular_distance - DECEL_NEAR) / (DECEL_START - DECEL_NEAR)
        t = linear ** 2
    return round(SPEED_NEUTRAL + t * (direction - SPEED_NEUTRAL), 2)


def flush_camera_buffer(n=10):
    for _ in range(n):
        cap.read()


def begin_target_sequence(target, label=""):
    global target_angle_20, outbound_direction, is_homing
    diff               = shortest_angle_diff(current_angle_20, target)
    outbound_direction = pick_direction(diff)
    target_angle_20    = target
    is_homing          = False
    dir_label = "CW" if outbound_direction == SPEED_FWD else "CCW"
    print(f"\n[TARGET] {label}{target}  |  from {int(current_angle_20)}"
          f"  |  diff={int(diff)}  |  dir={dir_label}")


print("\n=== Servo Tracking System (ArUco — lightweight) ===")
print(f"MOTOR_DIRECTION_SIGN = {MOTOR_DIRECTION_SIGN}")
print(f"Detect every {DETECT_EVERY_N} frames  |  Upscale x{UPSCALE}  |  equalizeHist\n")
print("AUTO TARGETS  -> 1 (62)  2 (131)  3 (249)  4 (295)")
print("MANUAL 360    -> W (CW)  S (CCW)  E (Stop)")
print("MANUAL 180    -> I (60)  J (0)")
print("EXPOSURE      -> + (less blur)  - (more light)")
print("QUIT          -> Q\n")

set_speed_20(SPEED_STOP)
set_angle_instant_21(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _frame_counter += 1

        # --- ArUco detection (skipped on non-detect frames) ------------------
        if _frame_counter % DETECT_EVERY_N == 0:
            detected_angle, aruco_corners = detect_aruco_angle(frame)
            if aruco_corners is not None:
                _last_corners = aruco_corners
        else:
            # Hold last known values — don't call the detector this frame
            detected_angle = current_angle_20 if last_marker_seen_time else None
            aruco_corners  = _last_corners

        marker_found = detected_angle is not None

        if marker_found:
            last_marker_seen_time = time.time()
            current_angle_20      = detected_angle

            if aruco_corners is not None:
                cv2.polylines(frame, [aruco_corners.astype(int)], True, (0, 0, 255), 2)
                c     = aruco_corners
                top_x = int((c[0][0] + c[1][0]) / 2)
                top_y = int((c[0][1] + c[1][1]) / 2)
                bot_x = int((c[2][0] + c[3][0]) / 2)
                bot_y = int((c[2][1] + c[3][1]) / 2)
                cv2.arrowedLine(frame, (bot_x, bot_y), (top_x, top_y),
                                (0, 255, 0), 2, tipLength=0.3)
            cv2.putText(frame, f"Angle: {int(current_angle_20)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Watchdog --------------------------------------------------------
        if not marker_found:
            elapsed = time.time() - last_marker_seen_time
            if elapsed > TRACKING_TIMEOUT and target_angle_20 is not None:
                print(f"\n[WARNING] Marker lost {elapsed:.1f}s -- motor halted.")
                set_speed_20(SPEED_STOP)
            cv2.putText(frame, "MARKER LOST", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- Auto-sequence ---------------------------------------------------
        if target_angle_20 is not None and marker_found:
            diff     = shortest_angle_diff(current_angle_20, target_angle_20)
            distance = abs(diff)

            if distance <= ANGLE_TOLERANCE:
                set_speed_20(SPEED_STOP)

                if not is_homing:
                    print(f"\n[TARGET REACHED] At {int(current_angle_20)}")
                    print("  -> Arm up...")
                    set_angle_instant_21(60)
                    time.sleep(1.0)
                    print("  -> Arm down...")
                    set_angle_instant_21(0)
                    time.sleep(1.0)

                    home_dir  = SPEED_BWD if outbound_direction == SPEED_FWD else SPEED_FWD
                    dir_label = "CW" if home_dir == SPEED_FWD else "CCW"
                    print(f"  -> Homing to {HOME_ANGLE} via {dir_label}...")
                    outbound_direction = home_dir
                    target_angle_20    = HOME_ANGLE
                    is_homing          = True
                    flush_camera_buffer()
                else:
                    print(f"\n[HOME REACHED] Idle at ~{int(current_angle_20)}")
                    target_angle_20    = None
                    outbound_direction = None
                    is_homing          = False
            else:
                live_dir = pick_direction(diff)
                if distance < DECEL_START and live_dir != outbound_direction:
                    set_speed_20(SPEED_STOP)
                else:
                    speed = decelerated_speed(outbound_direction, distance)
                    set_speed_20(speed)

        # --- HUD -------------------------------------------------------------
        if target_angle_20 is None:
            state_text = "Idle"
        elif is_homing:
            state_text = f"Homing -> {HOME_ANGLE}"
        else:
            state_text = f"Targeting -> {target_angle_20}"

        cv2.putText(frame, f"State: {state_text}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Exp: {_current_exposure}  Skip: 1/{DETECT_EVERY_N}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 0), 1)

        cv2.imshow("Live Tracking", frame)

        # --- Keys ------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('1'): begin_target_sequence(62,  "Compartment 1 -> ")
        elif key == ord('2'): begin_target_sequence(131, "Compartment 2 -> ")
        elif key == ord('3'): begin_target_sequence(249, "Compartment 3 -> ")
        elif key == ord('4'): begin_target_sequence(295, "Compartment 4 -> ")
        elif key == ord('w'):
            target_angle_20 = None
            set_speed_20(SPEED_FWD)
            print("Manual: CW")
        elif key == ord('s'):
            target_angle_20 = None
            set_speed_20(SPEED_BWD)
            print("Manual: CCW")
        elif key == ord('e'):
            target_angle_20 = None
            set_speed_20(SPEED_STOP)
            print("Manual: Stop")
        elif key == ord('i'):
            set_angle_instant_21(60)
            print("Manual: Arm -> 60")
        elif key == ord('j'):
            set_angle_instant_21(0)
            print("Manual: Arm -> 0")
        elif key == ord('=') or key == ord('+'):
            _current_exposure = max(-10, _current_exposure - 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, _current_exposure)
            print(f"Exposure -> {_current_exposure}")
        elif key == ord('-'):
            _current_exposure = min(0, _current_exposure + 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, _current_exposure)
            print(f"Exposure -> {_current_exposure}")

except KeyboardInterrupt:
    pass

finally:
    print("\nShutting down...")
    set_speed_20(SPEED_STOP)
    pwm20.stop()
    pwm21.stop()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()