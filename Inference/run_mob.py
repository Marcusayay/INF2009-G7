import cv2
import numpy as np
import time
import os
import statistics
from collections import defaultdict, Counter

from mobileNet_helpers import run_inference , load_model
from pick_model import return_latest_version_path    
# --- Hardware Monitoring (From your monitoring.py) ---
try:
    from monitoring import track_ram, track_temp, track_power, returns_latest_file_number
except ImportError:
    print("⚠️ monitoring.py not found. Mocking hardware stats.")
    track_ram = lambda: 0.0
    track_temp = lambda: 0.0
    track_power = lambda: 0.0
    returns_latest_file_number = lambda x: 1


# ---------------- Config ----------------

FRAME_W, FRAME_H = 640, 480
CAPTURE_BOX_SIZE = 320
MODEL_INPUT_SIZE = 224  # MobileNet typically uses 224x224 input, but we can experiment with larger crops and resizing


QUANTIZED = True  # Set to False if you want to test the Keras FP32 path (watch your RAM!) 
if QUANTIZED:
    IMAGE_OUTPUT_DIR = "output/mobilenet/quantized"
else: 
    IMAGE_OUTPUT_DIR = "output/mobilenet/fp32"

# Benchmark Settings
TIME_TO_TEST = 0
WARMUP_FRAMES = 20
SHOW_VIDEO = True

# State Management (Same Object Logic)
buffer = []
BUFFER_THRESHOLD = 10
FREQ_THRESHOLD = 7
AVERAGE_CONF_THRESHOLD = 0.85
detected = False
alert_frames = 0
last_detected_name = ""
# Cooldown Management
COOLDOWN_SECONDS = 3.0
last_save_time = 0.0

os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
runNumber = returns_latest_file_number(IMAGE_OUTPUT_DIR) 
os.makedirs(os.path.join(IMAGE_OUTPUT_DIR, f"run_{runNumber}"), exist_ok=True)
IMAGE_OUTPUT_DIR = os.path.join(IMAGE_OUTPUT_DIR, f"run_{runNumber}") 

# ---------------- Load Model ----------------
# get latest version 

latest_model_path, _ = return_latest_version_path("mobilenet")
model, input_details, output_details, model_path = load_model(quantized=QUANTIZED, model_path=latest_model_path)

# from latest_model_path, read the labels.txt to get the class names (if it exists)
labels_path = os.path.join(latest_model_path, "labels.txt") 
if os.path.exists(labels_path): 
    with open(labels_path, "r") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded class names from {labels_path}") 
print(f"class names: {CLASS_NAMES}")


#! ==========================================================================
# print(f"\n[PROFILER] My PID is: {os.getpid()}")
# print("[PROFILER] Waiting 5 seconds for pidstat/perf stat...")
# time.sleep(5)
#!=======================================================================
# ---------------- Helpers ----------------
def safe_mean(x, default=0.0): return statistics.mean(x) if x else default
def safe_median(x, default=0.0): return statistics.median(x) if x else default

# ---------------- Main Loop ----------------
cap = cv2.VideoCapture("/dev/video_vision")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("✅ System Ready.")

sys_times, model_infs, ram_usages, cpu_temps, power_usages = [], [], [], [], []
frame_count = 0

def correct_white_balance(img_rgb):
    # Simple grey-world assumption
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
    overall_mean = (r_mean + g_mean + b_mean) / 3
    img_wb = img_rgb.copy().astype(np.float32)
    img_wb[:,:,0] *= (overall_mean / r_mean)
    img_wb[:,:,1] *= (overall_mean / g_mean)
    img_wb[:,:,2] *= (overall_mean / b_mean)
    return np.clip(img_wb, 0, 255).astype(np.uint8)


def draw_class_probabilities(frame, class_names, probabilities, x_start=430, y_start=100):
    ranked = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
    line_height = 28

    cv2.putText(frame, "Class Probabilities", (x_start, y_start - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    for row, (i, prob) in enumerate(ranked):
        label = class_names[i] if i < len(class_names) else f"class_{i}"
        color = (0, 255, 0) if row == 0 else (255, 255, 255)
        text = f"{label}: {prob:.2%}"

        cv2.putText(
            frame,
            text,
            (x_start, y_start + row * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )


t_start = time.perf_counter()
try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        elapsed = time.perf_counter() - t_start

        if TIME_TO_TEST > 0 and elapsed >= TIME_TO_TEST:
            break

        # ---- System Latency Start ----
        t0 = time.perf_counter()

        # 1. Preprocessing (Crop to Center)
        y, x, _ = frame.shape
        startx = x//2 - CAPTURE_BOX_SIZE//2
        starty = y//2 - CAPTURE_BOX_SIZE//2

    
        # USE CAPTURE BOX SIZE TO INCREASE THE AMOUNT OF CONTEXT THE MODEL SEES (INSTEAD OF CROPPING TO EXACT MODEL INPUT SIZE), THEN RESIZE IN THE MODEL CODE. THIS OFTEN HELPS WITH REAL-WORLD PERFORMANCE BECAUSE THE MODEL CAN LEARN TO FOCUS ON THE CENTER WHILE STILL HAVING SOME CONTEXT.
        roi = frame[starty:starty+CAPTURE_BOX_SIZE, startx:startx+CAPTURE_BOX_SIZE]

        # crop it to the model size 
        roi_resized = cv2.resize(roi, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))

        img = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        img = correct_white_balance(img)  # ← add this

        # 2. Inference & Model Latency Start
        class_id, confidence, inf_time_ms, probabilities  = run_inference(
            quantized=QUANTIZED,
            inference_engine=model, 
            input_details=input_details,
            output_details=output_details,
            img_crop=img)


        # Total System Latency
        dt = time.perf_counter() - t0 
        fps = (1.0 / dt) if dt > 0 else 0.0

        # ---- Hardware Monitoring ----
        ram_percent = track_ram()
        cpu_temp = track_temp()
        power = track_power() 

        # ---- Object State Logic (Center Crop Only) ----
        current_time = time.perf_counter() 
        time_since_last_save = current_time - last_save_time 

        if time_since_last_save > COOLDOWN_SECONDS:             
            buffer.append((class_id, confidence))

            if len(buffer) > BUFFER_THRESHOLD: 
                buffer.pop(0) 

            # eval only when buffer is full to avoid premature decisions 
            if len(buffer) == BUFFER_THRESHOLD: 
                ids = [x[0] for x in buffer]
                counts = Counter(ids)
                #! [(1, 7)] most_common_cls is the class id that appears most frequently in the buffer, freq is how many times it appears 
                most_common_cls, freq = counts.most_common(1)[0] 

                scores_for_most_common = [x[1] for x in buffer if x[0] == most_common_cls]
                avg_confidence = sum(scores_for_most_common) / len(scores_for_most_common) 
                if freq >= FREQ_THRESHOLD and avg_confidence >= AVERAGE_CONF_THRESHOLD: 
                    print(f"📦 Trash: {CLASS_NAMES[most_common_cls]} detected consistently with high confidence (avg {avg_confidence:.2f}).")
                    detected = True
                    last_detected_name = CLASS_NAMES[most_common_cls] 
                    last_save_time = current_time # ⏱️ START THE COOLDOWN!

                    buffer.clear() # Reset buffer after decision 
        else:
            buffer.clear() # Clear buffer during cooldown to avoid stale data 
            
            cooldown_left = COOLDOWN_SECONDS - time_since_last_save
            cv2.putText(frame, f"COOLDOWN: {cooldown_left:.1f}s", (startx, starty - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # ---- Save Image Logic ----
        if detected:
            annotated = frame.copy()
            cv2.rectangle(annotated, (startx, starty), (startx+CAPTURE_BOX_SIZE, starty+CAPTURE_BOX_SIZE), (0, 255, 0), 2)
            cv2.putText(annotated, f"{last_detected_name} {avg_confidence:.2f}%", (startx, starty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            next_num = returns_latest_file_number(IMAGE_OUTPUT_DIR)
            saved_path = os.path.join(IMAGE_OUTPUT_DIR, f"garbage_mbnet_{next_num}.jpg")
            cv2.imwrite(saved_path, annotated)
            print (f"💾 Saved to {saved_path}")
            detected = False

            # 2. Trigger the on-screen alert to last for the next 15 frames (about half a second)
            alert_frames = 15 


        # ---- UI Display ----
        if SHOW_VIDEO:
            if alert_frames > 0: 
                # Draw a THICK RED box and warning text to show it saved
                cv2.rectangle(frame, (startx, starty), (startx+CAPTURE_BOX_SIZE, starty+CAPTURE_BOX_SIZE), (0, 0, 255), 6)
                cv2.putText(frame, f"DETECTED AS: {last_detected_name.upper()}", (startx - 20, starty - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                alert_frames -= 1 # Countdown the timer

            else: 
                cv2.rectangle(frame, (startx, starty), (startx+CAPTURE_BOX_SIZE, starty+CAPTURE_BOX_SIZE), (0, 255, 0), 2)
                cv2.putText(frame, f"Class: {CLASS_NAMES[class_id]} ({confidence:.2f})", (startx, starty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
          
            
            # system stats
            cv2.putText(frame, f"FPS: {fps:.1f} | Temp: {cpu_temp:.1f}C | Power: {power:.2f}W", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"RAM: {ram_percent:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            

            # draw all class probabilities on the side
            draw_class_probabilities(frame, CLASS_NAMES, probabilities, x_start=430, y_start=100)
            
            cv2.imshow("Pi 5 MobileNet Benchmark", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # ---- Stats Recording ----
        if frame_count > WARMUP_FRAMES:
            sys_times.append(dt)
            model_infs.append(inf_time_ms)
            ram_usages.append(ram_percent)
            cpu_temps.append(cpu_temp)
            power_usages.append(power)

except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()

# ---- Final Print ----
if len(sys_times) > 0:
    times_ms = [t * 1000 for t in sys_times]
    mean_sys_ms = safe_mean(times_ms)
    print("\n================ BENCHMARK RESULTS ================")
    print(f"Model: {model_path}")
    print(f"Test duration: {(time.perf_counter() - t_start):.2f} s")
    print(f"Frames measured: {len(sys_times)}")
    print(f"System FPS Mean: {1000.0 / mean_sys_ms if mean_sys_ms > 0 else 0.0:.2f}")
    print(f"System Latency Mean: {mean_sys_ms:.2f} ms")
    print(f"Model Inference Mean: {safe_mean(model_infs):.2f} ms")
    print(f"Python Overhead: {mean_sys_ms - safe_mean(model_infs):.2f} ms")
    print(f"Max RAM (%): {max(ram_usages) if ram_usages else 0:.1f}")
    print(f"Max CPU Temp (°C): {max(cpu_temps) if cpu_temps else 0:.1f}")
    print(f"Max Power (W): {max(power_usages) if power_usages else 0:.2f}")
    print("===================================================")