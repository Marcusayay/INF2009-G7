import cv2
from ultralytics import YOLO
import time
import statistics
import pandas as pd
import psutil
#! ================================
#! PUT THIS IN YOUR PI 
#! PUT THIS ALONGSIDE YOUR MODEL 
#! IN THIS CASE 
#! best_ncnn_model_fp32
#! ===============================

# ---------------- Config ----------------
MODEL_PATH = "./best_ncnn_model_fp32"   # <- your model path (or .pt)
TIME_TO_TEST = 30           # seconds
WARMUP_FRAMES = 20          # ignore first N frames in stats
SHOW_VIDEO = True          # benchmark mode if False
PLOT_EVERY = 1
CONF = 0.4
IOU = 0.5

# ---------------- Helpers ----------------
def safe_mean(x, default=0.0):
    return statistics.mean(x) if x else default

def safe_median(x, default=0.0):
    return statistics.median(x) if x else default

def finalize_and_print(model_name, elapsed_s,
                       sys_times_s, model_infs_ms, model_pres_ms, model_posts_ms,
                       ram_usages, cpu_temps):
    measured = len(sys_times_s)
    if measured < 5:
        print("❌ Not enough frames measured. Try increasing TIME_TO_TEST.")
        return None

    times_ms = [t * 1000 for t in sys_times_s]
    mean_sys_ms = safe_mean(times_ms)
    median_sys_ms = safe_median(times_ms)
    min_sys_ms = min(times_ms)
    max_sys_ms = max(times_ms)
    mean_fps = 1000.0 / mean_sys_ms if mean_sys_ms > 0 else 0.0

    mean_inf_ms = safe_mean(model_infs_ms)
    mean_pre_ms = safe_mean(model_pres_ms)
    mean_post_ms = safe_mean(model_posts_ms)

    python_overhead = mean_sys_ms - (mean_inf_ms + mean_pre_ms + mean_post_ms)

    summary = {
        "model": model_name,
        "frames_measured": measured,
        "test_duration_s": round(elapsed_s, 2),
        "system_fps_mean": round(mean_fps, 2),
        "system_latency_mean_ms": round(mean_sys_ms, 2),
        "system_latency_median_ms": round(median_sys_ms, 2),
        "system_latency_min_ms": round(min_sys_ms, 2),
        "system_latency_max_ms": round(max_sys_ms, 2),
        "model_inference_ms": round(mean_inf_ms, 2),
        "model_preprocess_ms": round(mean_pre_ms, 2),
        "model_postprocess_ms": round(mean_post_ms, 2),
        "python_overhead_ms": round(python_overhead, 2),
        "max_ram_percent": round(max(ram_usages), 1) if ram_usages else 0.0,
        "max_cpu_temp_c": round(max(cpu_temps), 1) if cpu_temps else 0.0,
    }

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Model: {model_name}")
    print(f"Test duration: {elapsed_s:.2f} s")
    print(f"Warmup ignored: {WARMUP_FRAMES} frames")
    print(f"Frames measured: {measured}")
    print("-" * 50)
    for k, v in summary.items():
        if k == "model":
            continue
        print(f"{k:<30}{v}")
    print("===================================================")

    return summary

# ---------------- Main ----------------

model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Tracking arrays
sys_times = []
model_infs = []
model_pres = []
model_posts = []
ram_usages = []
cpu_temps = []

frame_count = 0
t_start = time.perf_counter()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        elapsed = time.perf_counter() - t_start

        # Stop after TIME_TO_TEST seconds automatically
        #! DONT STOP if TIME_TO_TEST is 0 or negative (for manual q exit)
        if TIME_TO_TEST > 0 and elapsed >= TIME_TO_TEST:
            break

        # ---- System Latency ----
        t0 = time.perf_counter()
        pred = model(frame, conf=CONF, iou=IOU, verbose=False)
        dt = time.perf_counter() - t0  # seconds

        # ---- Model Latency (Ultralytics Method) ----
        speeds = pred[0].speed or {}
        model_inf = float(speeds.get("inference", 0.0))     # ms
        model_pre = float(speeds.get("preprocess", 0.0))    # ms
        model_post = float(speeds.get("postprocess", 0.0))  # ms

        # ---- Hardware Monitoring ----
        ram_percent = psutil.virtual_memory().percent
        try:
            temps = psutil.sensors_temperatures()
            cpu_temp = temps["cpu_thermal"][0].current  # common on Raspberry Pi
        except Exception:
            cpu_temp = 0.0

        # ---- Warmup handling: only record after warmup ----
        if frame_count > WARMUP_FRAMES:
            sys_times.append(dt)
            model_infs.append(model_inf)
            model_pres.append(model_pre)
            model_posts.append(model_post)
            ram_usages.append(ram_percent)
            cpu_temps.append(cpu_temp)

        # ---- Optional display ----
        if SHOW_VIDEO:
            if frame_count % PLOT_EVERY == 0:
                annotated = pred[0].plot()
            else:
                annotated = frame

            fps = (1.0 / dt) if dt > 0 else 0.0
            cv2.putText(
                annotated, f"FPS: {fps:.1f} | Temp: {cpu_temp:.1f}C", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            cv2.putText(
                annotated, f"RAM: {ram_percent:.1f}%", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2
            )
            cv2.imshow("Garbage Classifier", annotated)

            # If you press q -> exit and print stats
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

# Print stats for THIS run/model (also runs if you pressed q)
summary = finalize_and_print(
    model_name=MODEL_PATH,
    elapsed_s=time.perf_counter() - t_start,
    sys_times_s=sys_times,
    model_infs_ms=model_infs,
    model_pres_ms=model_pres,
    model_posts_ms=model_posts,
    ram_usages=ram_usages,
    cpu_temps=cpu_temps,
)