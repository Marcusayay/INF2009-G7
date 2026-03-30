import sys

# 1. First, make sure you have the new package installed:
# Run this in your terminal first: pip install ai-edge-litert

try:
    import ai_edge_litert
    from ai_edge_litert.interpreter import Interpreter, load_delegate
    
    # 2. This is the "Magic Trick"
    # We manually inject the new library into the system's search path 
    # under the old name that Ultralytics expects.
    sys.modules["tflite_runtime"] = ai_edge_litert
    sys.modules["tflite_runtime.interpreter"] = ai_edge_litert.interpreter
    print("🚀 Successfully redirected tflite_runtime to ai_edge_litert")
except ImportError:
    print("⚠️ ai-edge-litert not found. Run: pip install ai-edge-litert")
    
    
import cv2
from ultralytics import YOLO
import time
import statistics
import pandas as pd
import os 
import psutil
from tqdm import tqdm 


MODELS = os.listdir("./models") 
#MODELS = ["models/yolo26n.pt"]
TIME_TO_TEST = 30          # seconds
WARMUP_FRAMES = 20         # ignore first N frames in stats
SHOW_VIDEO = False          # set False for pure benchmarking (faster / cleaner)
PLOT_EVERY = 5             # draw boxes every N frames (reduce overhead)
CONF = 0.4
IOU = 0.5

results_summary = {}

for mp in tqdm(MODELS, desc="Testing Models", unit="model"): 
    mp1 = f"./models/{mp}" 
    if mp1.endswith(".npy") or mp1.endswith(".py") or mp1.endswith("bench"):
        print(f"⚠️ Skipping non-model file: {mp1}")
        continue 
    print(f"\n⏳ Loading model from: {mp1} ...")
    model = YOLO(mp1) 
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

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        elapsed = time.perf_counter() - t_start
        if elapsed >= TIME_TO_TEST:
            break

        frame_count += 1

        # ---- System Latency  ----
        t0 = time.perf_counter()
        pred = model(frame, conf=CONF, iou=IOU, verbose=False)
        dt = time.perf_counter() - t0 #! in seconds 


        # ---- Model Latency (Ultralytics Method) ----
        speeds = pred[0].speed 
        model_inf = speeds['inference']
        model_pre = speeds['preprocess']
        model_post = speeds['postprocess']
        
        
        # ---- Hardware Monitoring ----
        ram_percent = psutil.virtual_memory().percent
        try:
            temps = psutil.sensors_temperatures()
            # 'cpu_thermal' is specific to Raspberry Pi architecture
            cpu_temp = temps['cpu_thermal'][0].current 
        except:
            cpu_temp = 0.0 # Fallback for testing on Windows/Mac

        # ignore warmup frames and Record metrics 
        if frame_count > WARMUP_FRAMES:
            sys_times.append(dt)
            model_infs.append(model_inf)
            model_pres.append(model_pre)
            model_posts.append(model_post)
            ram_usages.append(ram_percent)
            cpu_temps.append(cpu_temp)

        # Optional display (avoid slowing benchmark too much)
        if SHOW_VIDEO:
            if frame_count % PLOT_EVERY == 0:
                annotated = pred[0].plot()
            else:
                annotated = frame

            fps = (1.0 / dt) if dt > 0 else 0.0
            cv2.putText(
                annotated, f"YOLO FPS (inf): {fps:.1f} | Temp: {cpu_temp}C", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            cv2.putText(
                annotated, f"RAM: {ram_percent}%", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2
            )
            cv2.imshow("Benchmark", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    measured = len(sys_times)
    if measured < 5:
        print("❌ Not enough frames measured. Try increasing TIME_TO_TEST.")
        continue

    times_ms = [t * 1000 for t in sys_times]
    mean_sys_ms = statistics.mean(times_ms)
    median_sys_ms = statistics.median(times_ms)
    min_sys_ms = min(times_ms)
    max_sys_ms = max(times_ms)
    mean_fps = 1000.0 / mean_sys_ms

    # aggregate for model inference, preprocess, postprocess separately 
    mean_inf_ms = statistics.mean(model_infs)
    mean_pre_ms = statistics.mean(model_pres)
    mean_post_ms = statistics.mean(model_posts)

    # bottle neck 
    python_overhead = mean_sys_ms - (mean_inf_ms + mean_pre_ms + mean_post_ms) 

    results_summary[mp] = {
        "family": mp.split("_")[0][:-1],
        "frames_measured": measured,
        "system_fps_mean": round(mean_fps, 2), 
        "system_latency_mean_ms": round(mean_sys_ms, 2),
        "system_latency_median_ms": round(median_sys_ms, 2),
        "system_latency_min_ms": round(min_sys_ms, 2), 
        "system_latency_max_ms": round(max_sys_ms, 2), 
        "model_inference_ms": round(mean_inf_ms, 2), 
        "model_preprocess_ms": round(mean_pre_ms, 2), 
        "model_postprocess_ms": round(mean_post_ms, 2), 
        "python_overhead_ms": round(python_overhead, 2), 
        "max_ram_percent": round(max(ram_usages), 1), 
        "max_cpu_temp_c": round(max(cpu_temps), 1)
    }
    print("\n================ BENCHMARK RESULTS ================")
    print(f"Model: {mp}")
    print(f"Test duration: {elapsed:.2f} s")
    print(f"Warmup ignored: {WARMUP_FRAMES} frames")
    print(f"Frames measured: {measured}")
    print("-" * 40)
    for metric,value in results_summary[mp].items():
        print(f"{metric:<30}{value}") 
    print("===================================================")

df = pd.DataFrame(results_summary).T
print("\n================ SUMMARY TABLE ================")
print(df.sort_values("system_fps_mean", ascending=False))
df.to_csv("benchmark_results.csv", index=True) 