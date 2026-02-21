import tensorflow as tf

def inspect(path):
    itp = tf.lite.Interpreter(model_path=path)
    itp.allocate_tensors()
    in0 = itp.get_input_details()[0]
    out0 = itp.get_output_details()[0]
    print("\n===", path, "===")
    print("Input dtype:", in0["dtype"], "quant:", in0.get("quantization"))
    print("Output dtype:", out0["dtype"], "quant:", out0.get("quantization"))

inspect("models/yolo26n_int8.tflite")
inspect("models/yolo26n_integer_quant.tflite")
inspect("models/yolo26n_full_integer_quant.tflite")