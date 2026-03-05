import cv2
import numpy as np
import time
#import tflite_runtime.interpreter as tflite # Lightweight version for Pi
import tensorflow as tf 


# ---------------- Config ----------------
MODEL_PATH = "best_quantized.tflite"
# Mapping based on your 'train' folder screenshot
CLASS_NAMES = [
    "bottle", 
    "can", 
    "cardboard", 
    "carton_tetrapack", 
    "cup_mug", 
    "disposable_cup", 
    "paper_sheet", 
    "plastic-wrapper", 
    "wine_glass"
]
FRAME_W, FRAME_H = 640, 480
CROP_SIZE = 224 # Match your model input size

# ---------------- Load Model ----------------
print("🚀 Loading Quantized TFLite Model...")
# model = tf.keras.models.load_model("best.keras")
# Now you can use tf.lite.Interpreter instead of tflite.Interpreter
interpreter = tf.lite.Interpreter(model_path="best_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- Main Loop ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

print("✅ System Ready. Press 'q' to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Start timing for Section 10.2 (Tail Latency)
        t0 = time.perf_counter()

        # 1. Preprocessing (Crop to Center)
        y, x, _ = frame.shape
        startx = x//2 - CROP_SIZE//2
        starty = y//2 - CROP_SIZE//2
        roi = frame[starty:starty+CROP_SIZE, startx:startx+CROP_SIZE]
        
        # 2. Convert to RGB and UINT8
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0).astype(np.uint8) # Must be uint8 for INT8 model

        # 3. Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # 4. Post-processing
        class_id = np.argmax(output_data)
        confidence = output_data[0][class_id] / 255.0 # Convert back to 0-1 range for display

        # End timing
        dt = (time.perf_counter() - t0) * 1000 # Convert to ms
        fps = 1000.0 / dt

        # ---- Display Results ----
        cv2.rectangle(frame, (startx, starty), (startx+CROP_SIZE, starty+CROP_SIZE), (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {dt:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Class: {CLASS_NAMES[class_id]} (Conf: {confidence:.2f})", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Pi 5 Edge Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()