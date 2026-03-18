import sys
import time 
import numpy as np
import os

# ---------------- THE MAGIC TRICK ----------------
try:
    import ai_edge_litert
    import ai_edge_litert.interpreter
    sys.modules["tflite_runtime"] = ai_edge_litert
    sys.modules["tflite_runtime.interpreter"] = ai_edge_litert.interpreter
    import tflite_runtime.interpreter as tflite
    print("🚀 Using LiteRT Engine via alias")
except ImportError:
    print("❌ Critical Error: ai-edge-litert not found.")
    sys.exit(1)

def load_model(
    quantized: bool = True,
    model_path: str = None) : 
    if quantized: 
        if model_path is None:
            model_path = "best_quantized.tflite"
        else: 
            model_path = os.path.join(model_path, "best_quantized.tflite") 
        print("🚀 Loading Quantized LiteRT Model...")

    else: 
        if model_path is None:
            model_path = "best_fp32.tflite"
        else: 
            model_path = os.path.join(model_path, "best_fp32.tflite")
        print(f"🚀 Loading massive Keras model {model_path} (Watch your RAM)...")

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details, model_path
        
# def run_inference(
#     quantized = True,
#     inference_engine=None,
#     input_details=None,
#     output_details=None,
#     img_crop=None) -> tuple[int, float, float] :
    
#     inf_start = time.perf_counter()
    
#     if quantized: 
#         # INT8 requires uint8 input
#         input_data = np.expand_dims(img_crop, axis=0).astype(np.uint8)
#     else: 
#         # FP32 requires float32 input normalized to 0-1
#         input_data = np.expand_dims(img_crop, axis=0).astype(np.float32) / 255.0

#     # 2. Inference: Use the LiteRT C++ bindings
#     inference_engine.set_tensor(input_details[0]['index'], input_data)
#     inference_engine.invoke()
#     output_data = inference_engine.get_tensor(output_details[0]['index'])
    
#     # 3. Post-processing
#     class_id = np.argmax(output_data)
    
#     if quantized:
#         confidence = output_data[0][class_id] / 255.0       
#     else:
#         # FP32 is already a float percentage
#         confidence = float(output_data[0][class_id])
    
#     inf_time_ms = (time.perf_counter() - inf_start) * 1000 
    
#     return class_id, confidence, inf_time_ms



def run_inference(
    quantized=True,
    inference_engine=None,
    input_details=None,
    output_details=None,
    img_crop=None
) -> tuple[int, float, float, np.ndarray]:

    inf_start = time.perf_counter()

    # Preprocess input
    if quantized:
        input_data = np.expand_dims(img_crop, axis=0).astype(np.uint8)
    else:
        input_data = np.expand_dims(img_crop, axis=0).astype(np.float32) / 255.0

    # Run inference
    inference_engine.set_tensor(input_details[0]['index'], input_data)
    inference_engine.invoke()
    output_data = inference_engine.get_tensor(output_details[0]['index'])[0]


    # Convert raw output into probabilities
    if quantized:
        # For quantized output, dequantize first if possible
        scale, zero_point = output_details[0].get("quantization", (0.0, 0))
        probabilities = scale * (output_data.astype(np.float32) - zero_point)

    else:
        probabilities = output_data.astype(np.float32)  # already softmaxed by model

    # Softmax to convert logits -> probabilities
    #exp_scores = np.exp(output_data - np.max(output_data))
    #probabilities = exp_scores / np.sum(exp_scores)

    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id])

    inf_time_ms = (time.perf_counter() - inf_start) * 1000

    return class_id, confidence, inf_time_ms, probabilities