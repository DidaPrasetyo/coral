import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="yolov8n/best_int8.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print the dimensions of the input tensor.
input_shape = input_details[0]['shape']
print("Input Tensor Shape:", input_shape)

# Print the dimensions of the output tensor (assuming there is only one output tensor).
output_shape = output_details[0]['shape']
print("Output Tensor Shape:", output_shape)
