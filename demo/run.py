import cv2
import numpy as np
import time
import os
import statistics
from collections import Counter

from mobileNet_helpers import run_inference, load_model
from pick_model import return_latest_version_path

try:
    from monitoring import track_ram, track_temp, track_power, returns_latest_file_number
except ImportError:
    track_ram = lambda: 0.0
    track_temp = lambda: 0.0
    track_power = lambda: 0.0

# ---------------- Config ----------------
FRAME_W, FRAME_H = 640, 480
CAPTURE_BOX_SIZE = 320
MODEL_INPUT_SIZE = 224
QUANTIZED = True

BUFFER_THRESHOLD = 10
FREQ_THRESHOLD = 7
AVERAGE_CONF_THRESHOLD = 0.89

# ---------------- Load Model ----------------
latest_model_path = return_latest_version_path("mobilenet")
model, input_details, output_details, model_path = load_model(quantized=QUANTIZED, model_path=latest_model_path)

labels_path = os.path.join(latest_model_path, "labels.txt")
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded class names: {CLASS_NAMES}")

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("✅ Waiting for detection... (q to quit)\n")

buffer = []
frame_count = 0
t_start = time.perf_counter()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ---- Preprocess ----
        y, x, _ = frame.shape
        startx = x // 2 - CAPTURE_BOX_SIZE // 2
        starty = y // 2 - CAPTURE_BOX_SIZE // 2
        roi = frame[starty:starty+CAPTURE_BOX_SIZE, startx:startx+CAPTURE_BOX_SIZE]
        roi_resized = cv2.resize(roi, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        img = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

        # ---- Inference ----
        class_id, confidence, inf_time_ms, probabilities = run_inference(
            quantized=QUANTIZED,
            inference_engine=model,
            input_details=input_details,
            output_details=output_details,
            img_crop=img
        )
        #print (f"probabilities: {probabilities}")

        # ---- Buffer ----
        buffer.append((class_id, confidence, probabilities))
        if len(buffer) > BUFFER_THRESHOLD:
            buffer.pop(0)

        print(f"Frame {frame_count:4d} | {CLASS_NAMES[class_id]:<20} {confidence:.2f}", end="\r")

        # ---- Eval ----
        if len(buffer) == BUFFER_THRESHOLD:
            ids = [x[0] for x in buffer]
            counts = Counter(ids)
            most_common_cls, freq = counts.most_common(1)[0]
            scores_for_most_common = [x[1] for x in buffer if x[0] == most_common_cls]
            avg_confidence = sum(scores_for_most_common) / len(scores_for_most_common)

            if freq >= FREQ_THRESHOLD and avg_confidence >= AVERAGE_CONF_THRESHOLD:
                elapsed = time.perf_counter() - t_start

                print(f"\n\n{'='*50}")
                print(f"✅ DETECTED: {CLASS_NAMES[most_common_cls]}")
                print(f"{'='*50}")
                print(f"Frames to detection : {frame_count}")
                print(f"Time to detection   : {elapsed:.2f}s")
                print(f"Freq in buffer      : {freq}/{BUFFER_THRESHOLD}")
                print(f"Avg confidence      : {avg_confidence:.2%}")
                print(f"Min confidence      : {min(scores_for_most_common):.2%}")
                print(f"Max confidence      : {max(scores_for_most_common):.2%}")

                # median
                sorted_scores = sorted(scores_for_most_common)
                median_conf = sorted_scores[len(sorted_scores) // 2]
                print(f"Median confidence   : {median_conf:.2%}")

                print(f"\nFull buffer at trigger:")
                for i, (cid, conf, probs) in enumerate(buffer):
                    margin = sorted(probs, reverse=True)[0] - sorted(probs, reverse=True)[1]
                    print(f"  [{i+1:2d}] {CLASS_NAMES[cid]:<20} conf={conf:.2f}  margin={margin:.2f}")

                print(f"\nFinal probabilities (averaged across buffer):")
                avg_probs = np.mean([x[2] for x in buffer], axis=0)
                ranked = sorted(enumerate(avg_probs), key=lambda x: x[1], reverse=True)
                for cid, prob in ranked:
                    label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
                    bar = "█" * int(prob * 40)
                    print(f"  {label:<20} {prob:.2%}  {bar}")

                break

   
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted — computing stats from current buffer...\n")
    if len(buffer) == 0:
        print("Buffer is empty, nothing to compute.")
    else:
        ids = [x[0] for x in buffer]
        counts = Counter(ids)
        most_common_cls, freq = counts.most_common(1)[0]
        scores_for_most_common = [x[1] for x in buffer if x[0] == most_common_cls]
        avg_confidence = sum(scores_for_most_common) / len(scores_for_most_common)
        sorted_scores = sorted(scores_for_most_common)
        median_conf = sorted_scores[len(sorted_scores) // 2]
        elapsed = time.perf_counter() - t_start

        print(f"{'='*50}")
        print(f"Buffer size         : {len(buffer)}/{BUFFER_THRESHOLD}")
        print(f"Top class           : {CLASS_NAMES[most_common_cls]}")
        print(f"Freq in buffer      : {freq}/{len(buffer)}")
        print(f"Avg confidence      : {avg_confidence:.2%}")
        print(f"Min confidence      : {min(scores_for_most_common):.2%}")
        print(f"Max confidence      : {max(scores_for_most_common):.2%}")
        print(f"Median confidence   : {median_conf:.2%}")
        print(f"Time elapsed        : {elapsed:.2f}s")

        print(f"\nFull buffer:")
        for i, (cid, conf, probs) in enumerate(buffer):
            margin = sorted(probs, reverse=True)[0] - sorted(probs, reverse=True)[1]
            print(f"  [{i+1:2d}] {CLASS_NAMES[cid]:<20} conf={conf:.2f}  margin={margin:.2f}")

        print(f"\nFinal probabilities (averaged across buffer):")
        avg_probs = np.mean([x[2] for x in buffer], axis=0)
        ranked = sorted(enumerate(avg_probs), key=lambda x: x[1], reverse=True)
        for cid, prob in ranked:
            label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
            bar = "█" * int(prob * 40)
            print(f"  {label:<20} {prob:.2%}  {bar}")
finally:
    cap.release()
    cv2.destroyAllWindows()