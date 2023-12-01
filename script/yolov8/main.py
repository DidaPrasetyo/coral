import numpy as np
import tensorflow as tf
# import tflite_runtime.interpreter as tflite

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="yolov8n/best_int8.tflite")
# interpreter = tflite.Interpreter("yolov8n/best_integer_quant.tflite",
#   experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# best_full use float32 format, not int8

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)