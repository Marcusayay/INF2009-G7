import cv2
import numpy as np
import math
import RPi.GPIO as GPIO
import time
# Color correction LUT — counteracts purple UV lighting
_CORRECTION_LUT_B = None
_CORRECTION_LUT_G = None
_CORRECTION_LUT_R = None

_CORRECTION_LUT_B = None
_CORRECTION_LUT_G = None
_CORRECTION_LUT_R = None

def correct_frame(frame):
    """White-balance correction: neutralises purple UV cast so green tape reads as green."""
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
#   If the angle INCREASES  → set MOTOR_DIRECTION_SIGN = +1
#   If the angle DECREASES  → set MOTOR_DIRECTION_SIGN = -1
#
MOTOR_DIRECTION_SIGN = 1   # <-- flip to -1 if rotation direction is backwards
# =============================================================================

# Speed settings
SPEED_FWD     = 9.0
SPEED_BWD     = 6.0
SPEED_NEUTRAL = 7.5
SPEED_STOP    = 0

# Deceleration zones (degrees from target)
DECEL_START     = 60   # Start slowing earlier
DECEL_NEAR      = 25   # Cut to crawl with more runway
CRAWL_FACTOR    = 0.20 # Slightly more assertive crawl so it commits and stops
ANGLE_TOLERANCE = 7   # Wider dead-band — good enough, no more jitter

# State
last_angle_21    = -1
current_angle_20 = 0.0
current_speed_20 = -1.0
target_angle_20  = None
outbound_direction = None  # locked at sequence start, reversed for homing
is_homing        = False
HOME_ANGLE       = 293

# Watchdog
last_tape_seen_time = time.time()
TRACKING_TIMEOUT    = 2.0

# Camera
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

center_x, center_y = 198, 124


def set_center_callback(event, x, y, flags, param):
    global center_x, center_y
    if event == cv2.EVENT_LBUTTONDOWN:
        center_x, center_y = x, y
        print(f"\n[CALIBRATED] Pivot centre -> X:{x}  Y:{y}")

cv2.namedWindow("Live Tracking")
cv2.setMouseCallback("Live Tracking", set_center_callback)


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
    """Signed shortest path current->target. Range: (-180, +180]."""
    return (target - current + 180) % 360 - 180


def pick_direction(diff):
    """
    Convert signed diff to a motor direction using the calibration sign.
    MOTOR_DIRECTION_SIGN=+1: positive diff (need to go CCW in math) -> FWD
    MOTOR_DIRECTION_SIGN=-1: positive diff -> BWD (servo physically inverted)
    """
    if MOTOR_DIRECTION_SIGN * diff > 0:
        return SPEED_FWD
    else:
        return SPEED_BWD


def decelerated_speed(direction, angular_distance):
    """
    Quadratic ramp: speed drops off much faster as target approaches.
    Squaring the linear ratio means it slows down hard well before arrival
    rather than only braking at the last moment.
    """
    if angular_distance >= DECEL_START:
        t = 1.0
    elif angular_distance <= DECEL_NEAR:
        t = CRAWL_FACTOR
    else:
        linear = (angular_distance - DECEL_NEAR) / (DECEL_START - DECEL_NEAR)
        t = linear ** 2   # quadratic drop-off
    return round(SPEED_NEUTRAL + t * (direction - SPEED_NEUTRAL), 2)


def flush_camera_buffer(n=10):
    for _ in range(n):
        cap.read()


def begin_target_sequence(target, label=""):
    global target_angle_20, outbound_direction, is_homing
    diff = shortest_angle_diff(current_angle_20, target)
    outbound_direction = pick_direction(diff)   # locked — not recalculated
    target_angle_20    = target
    is_homing          = False
    dir_label = "CW" if outbound_direction == SPEED_FWD else "CCW"
    print(f"\n[TARGET] {label}{target}  |  from {int(current_angle_20)}  |  diff={int(diff)}  |  dir={dir_label}")


print("\n=== Servo Tracking System ===")
print(f"MOTOR_DIRECTION_SIGN = {MOTOR_DIRECTION_SIGN}")
print("  Press W and watch the angle. If it goes DOWN, flip sign to -1.\n")
print("AUTO TARGETS  -> 1 (62)  2 (131)  3 (249)  4 (295)")
print("MANUAL 360    -> W (CW)  S (CCW)  E (Stop)")
print("MANUAL 180    -> I (90)  J (0)")
print("QUIT          -> Q\n")

set_speed_20(SPEED_STOP)
set_angle_instant_21(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Vision ---
        corrected = correct_frame(frame)
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 55, 15]), np.array([15, 255, 160]))
        mask2 = cv2.inRange(hsv, np.array([170, 55  , 15]), np.array([179, 255, 160]))
       
        mask4 = cv2.inRange(hsv, np.array([150, 200  , 45]), np.array([179, 255, 70]))

        mask = mask1 | mask2  | mask4
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        


        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        tape_found = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                tape_found = True
                last_tape_seen_time = time.time()

                tip_x, tip_y = max(
                    (pt[0] for pt in largest),
                    key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2
                )

                current_angle_20 = math.degrees(
                    math.atan2(center_y - tip_y, tip_x - center_x)
                ) % 360

                cv2.drawContours(frame, [largest], -1, (0, 0, 255), 2)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                cv2.circle(frame, (tip_x, tip_y), 5, (0, 255, 0), -1)
                cv2.line(frame, (center_x, center_y), (tip_x, tip_y), (255, 255, 0), 2)
                cv2.putText(frame, f"Angle: {int(current_angle_20)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- Watchdog ---
        if not tape_found:
            elapsed = time.time() - last_tape_seen_time
            if elapsed > TRACKING_TIMEOUT and target_angle_20 is not None:
                print(f"\n[WARNING] Tape lost {elapsed:.1f}s -- motor halted.")
                set_speed_20(SPEED_STOP)
            cv2.putText(frame, "TAPE LOST", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- Auto-sequence ---
        if target_angle_20 is not None and tape_found:
            diff     = shortest_angle_diff(current_angle_20, target_angle_20)
            distance = abs(diff)

            if distance <= ANGLE_TOLERANCE:
                set_speed_20(SPEED_STOP)

                if not is_homing:
                    print(f"\n[TARGET REACHED] At {int(current_angle_20)}")
                    print("  -> Arm up (90)...")
                    set_angle_instant_21(60)
                    time.sleep(1.0)
                    print("  -> Arm down (0)...")
                    set_angle_instant_21(0)
                    time.sleep(1.0)

                    # Reverse the original outbound direction for homing
                    home_dir  = SPEED_BWD if outbound_direction == SPEED_FWD else SPEED_FWD
                    dir_label = "CW" if home_dir == SPEED_FWD else "CCW"
                    print(f"  -> Homing to {HOME_ANGLE} via {dir_label} (reverse of outbound)...")
                    outbound_direction = home_dir  # now locked for the return trip
                    target_angle_20    = HOME_ANGLE
                    is_homing          = True
                    flush_camera_buffer()
                else:
                    print(f"\n[HOME REACHED] Idle at ~{int(current_angle_20)}")
                    target_angle_20    = None
                    outbound_direction = None
                    is_homing          = False
            else:
                # Outside decel zone: use locked direction.
                # Inside decel zone: NEVER flip direction — just crawl in
                # and let the tolerance dead-band do the stopping.
                # Flipping direction inside the zone is what causes jitter.
                live_dir = pick_direction(diff)
                if distance < DECEL_START and live_dir != outbound_direction:
                    # We're close and diff has flipped — we already passed it
                    # slightly. Stop immediately rather than oscillating back.
                    set_speed_20(SPEED_STOP)
                else:
                    direction = outbound_direction
                    speed     = decelerated_speed(direction, distance)
                    set_speed_20(speed)

        # --- HUD ---
        if target_angle_20 is None:
            state_text = "Idle"
        elif is_homing:
            state_text = f"Homing -> {HOME_ANGLE}"
        else:
            state_text = f"Targeting -> {target_angle_20}"

        cv2.putText(frame, f"State: {state_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Live Tracking", frame)
        cv2.imshow("Filter", corrected    )
        cv2.imshow("Mask (White=Tape)", mask)

        # --- Keys ---
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