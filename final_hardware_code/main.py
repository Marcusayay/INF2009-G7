import time
import sys
import math
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
OFFSET = 0  # Will be set during tare
WEIGHT_TRIGGER_THRESHOLD = 15.0  # Grams required to start the pipeline
CONTAMINATION_WEIGHT_LIMIT = 500.0 # Grams. Anything heavier goes to general waste

# -- Inductive Sensor (HC06 / SN04-N) --
PIN_INDUCTIVE = 16 
metal_sensor = DigitalInputDevice(PIN_INDUCTIVE, pull_up=True)

# -- Spectrometer (AS7343) --
SAFE_ZONE_MARGIN = 0.02 
try:
    spectrometer = AS7343()
    spectrometer.set_gain(64)
    spectrometer.set_integration_time(200)
    SPECTROMETER_READY = True
except Exception as e:
    print(f"[WARNING] Spectrometer Error: {e}")
    SPECTROMETER_READY = False
base_ratio = 0.0

# -- Servos & Tracking (from servo_controller.py) --
PIN_20 = 20  # Base Rotation
PIN_21 = 21  # Arm/Actuator
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(PIN_20, GPIO.OUT)
pwm20 = GPIO.PWM(PIN_20, 50)
pwm20.start(0)

GPIO.setup(PIN_21, GPIO.OUT)
pwm21 = GPIO.PWM(PIN_21, 50)
pwm21.start(0)

# Servo State Variables
MOTOR_DIRECTION_SIGN = 1
SPEED_FWD, SPEED_BWD, SPEED_NEUTRAL, SPEED_STOP = 9.0, 6.0, 7.5, 0
DECEL_START, DECEL_NEAR, CRAWL_FACTOR, ANGLE_TOLERANCE = 60, 25, 0.20, 7

last_angle_21 = -1
current_angle_20 = 0.0
current_speed_20 = -1.0
target_angle_20 = None
outbound_direction = None
is_homing = False
HOME_ANGLE = 90

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
center_x, center_y = 127, 98

# Compartment Mapping (Angles from your script)
COMPARTMENTS = {
    "METAL": 62,
    "GLASS": 131,
    "PLASTIC": 249,
    "CONTAMINATED": 295
}


# =============================================================================
# 2. HELPER FUNCTIONS (Sensors & Servos)
# =============================================================================

def tare_scale():
    global OFFSET
    print("[SYSTEM] Taring scale... Please ensure it is empty.")
    time.sleep(1)
    vals = hx.get_raw_data(10)
    if vals:
        vals.sort()
        OFFSET = vals[len(vals) // 2]
    print(f"[SYSTEM] Tare complete. Offset: {OFFSET}")

def get_weight():
    vals = hx.get_raw_data(5)
    if not vals: return 0
    vals.sort()
    raw_val = vals[len(vals) // 2]
    return (raw_val - OFFSET) / RATIO

def calibrate_spectrometer(samples=8):
    global base_ratio
    print("[SYSTEM] Calibrating AS7343 baseline...")
    ratios = []
    for _ in range(samples):
        spectrometer.bank_select(1)
        time.sleep(0.1)
        d1 = spectrometer.get_data()[0]
        f1_uv = d1['fz']
        f3_blue = d1['fxl']
        ratios.append(f3_blue / f1_uv if f1_uv > 0 else 0)
    base_ratio = sum(ratios) / len(ratios)
    print(f"[SYSTEM] Baseline Ratio locked: {base_ratio:.3f}")

# (Includes abbreviated servo helpers to save space - insert your math functions here)
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
    if distance >= DECEL_START: t = 1.0
    elif distance <= DECEL_NEAR: t = CRAWL_FACTOR
    else: t = ((distance - DECEL_NEAR) / (DECEL_START - DECEL_NEAR)) ** 2
    return round(SPEED_NEUTRAL + t * (direction - SPEED_NEUTRAL), 2)


# =============================================================================
# 3. PARALLEL PROCESSING PATHS
# =============================================================================

def path_1_vision_model():
    """Parallel Path 1: Object Detection (Placeholder)"""
    print("[PATH 1] Initiating Object Detection Vision Model...")
    time.sleep(1.5) # Simulating inference time
    # TODO: Insert TF Lite / PyTorch inference here
    predicted_object = "Bottle" 
    print(f"[PATH 1] Vision Model Result: {predicted_object}")
    return predicted_object

def path_2_material_detection():
    """Parallel Path 2: Inductive -> Spectrometer Logic"""
    print("[PATH 2] Initiating Material Detection...")
    
    # 1. Check Inductive Sensor
    is_metal = (metal_sensor.value == 0) # 0 = Signal (Metal)
    
    if is_metal:
        print("[PATH 2] SN04-N detected METAL. Skipping Spectrometer.")
        return "METAL"
    
    # 2. If not metal, run Spectrometer
    print("[PATH 2] Non-metal detected. Triggering Spectrometer...")
    if not SPECTROMETER_READY:
        return "UNKNOWN_PLASTIC_OR_GLASS"

    spectrometer.bank_select(1)
    time.sleep(0.25)
    d1 = spectrometer.get_data()[0]
    f1_uv, f3_blue = d1['fz'], d1['fxl']
    ratio = f3_blue / f1_uv if f1_uv > 0 else 0
    diff = ratio - base_ratio
    
    if abs(diff) < SAFE_ZONE_MARGIN:
        result = "PLASTIC"
    elif diff < -SAFE_ZONE_MARGIN:
        result = "GLASS"
    else:
        result = "PLASTIC" # Default fallback
        
    print(f"[PATH 2] Spectrometer Result: {result} (Diff: {diff:+.3f})")
    return result


# =============================================================================
# 4. BACKGROUND CAMERA & SERVO THREAD
# =============================================================================

def servo_tracking_daemon():
    """Runs continuously in the background to handle OpenCV and Servo logic."""
    global current_angle_20, target_angle_20, outbound_direction, is_homing
    last_tape_seen_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        tape_found = False
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                tape_found = True
                last_tape_seen_time = time.time()
                tip_x, tip_y = max((pt[0] for pt in largest), key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
                current_angle_20 = math.degrees(math.atan2(center_y - tip_y, tip_x - center_x)) % 360

        # Halt if tape lost
        if not tape_found and (time.time() - last_tape_seen_time) > 2.0 and target_angle_20 is not None:
            set_speed_20(SPEED_STOP)

        # Homing/Targeting Logic
        if target_angle_20 is not None and tape_found:
            diff = shortest_angle_diff(current_angle_20, target_angle_20)
            dist = abs(diff)

            if dist <= ANGLE_TOLERANCE:
                set_speed_20(SPEED_STOP)
                if not is_homing:
                    print(f"\n[SERVO] Arrived at Target. Actuating Arm...")
                    set_angle_instant_21(100) # Arm Up
                    set_angle_instant_21(0)   # Arm Down
                    
                    # Begin homing
                    home_dir = SPEED_BWD if outbound_direction == SPEED_FWD else SPEED_FWD
                    outbound_direction = home_dir
                    target_angle_20 = HOME_ANGLE
                    is_homing = True
                    for _ in range(10): cap.read() # Flush buffer
                else:
                    print(f"[SERVO] Homing complete. System Idle.\n")
                    target_angle_20 = None
                    outbound_direction = None
                    is_homing = False
            else:
                live_dir = pick_direction(diff)
                if dist < DECEL_START and live_dir != outbound_direction:
                    set_speed_20(SPEED_STOP)
                else:
                    set_speed_20(decelerated_speed(outbound_direction, dist))
        time.sleep(0.01)

# =============================================================================
# 5. MAIN PIPELINE LOOP
# =============================================================================

def main_pipeline():
    global target_angle_20, outbound_direction, is_homing
    
    # Start the continuous servo tracking in the background
    threading.Thread(target=servo_tracking_daemon, daemon=True).start()
    
    hx.reset()
    tare_scale()
    if SPECTROMETER_READY: calibrate_spectrometer()
    set_angle_instant_21(0)
    
    print("\n" + "="*40)
    print(" SYSTEM ONLINE - AWAITING OBJECT")
    print("="*40)

    try:
        while True:
            # Step 1: Wait for weight trigger
            weight = get_weight()
            if weight > WEIGHT_TRIGGER_THRESHOLD and target_angle_20 is None:
                print(f"\n[TRIGGER] Object detected. Weight: {weight:.1f}g")
                
                # Step 2 & 3: Run parallel paths
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_vision = executor.submit(path_1_vision_model)
                    future_material = executor.submit(path_2_material_detection)
                    
                    vision_result = future_vision.result()
                    material_result = future_material.result()
                
                # Step 4: Contamination Check & Sensor Fusion
                print("\n--- SENSOR FUSION & CLASSIFICATION ---")
                if weight > CONTAMINATION_WEIGHT_LIMIT:
                    print(f"[!] Heavy Contamination Detected ({weight:.1f}g > Limit)")
                    final_decision = "CONTAMINATED"
                else:
                    # Could fuse vision_result here (e.g., if object is "Battery" -> Contaminated)
                    final_decision = material_result
                
                print(f"Final Decision: {final_decision}")
                
                # Step 5: Actuate to Compartment
                target_deg = COMPARTMENTS.get(final_decision, COMPARTMENTS["CONTAMINATED"])
                
                # Trigger the background daemon to handle the movement
                diff = shortest_angle_diff(current_angle_20, target_deg)
                outbound_direction = pick_direction(diff)
                target_angle_20 = target_deg
                is_homing = False
                
                print(f"Routing {final_decision} to Compartment @ {target_deg} degrees...")
                
                # Wait for system to clear before taking next reading
                while target_angle_20 is not None:
                    time.sleep(0.5)
                    
            time.sleep(0.2)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        set_speed_20(SPEED_STOP)
        pwm20.stop()
        pwm21.stop()
        GPIO.cleanup()
        cap.release()

if __name__ == "__main__":
    main_pipeline()