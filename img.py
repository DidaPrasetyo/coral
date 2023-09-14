import numpy as np
import cv2
import tensorflow as tf
# import tflite_runtime.interpreter as tflite

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="yolov8n/best_int8.tflite")
# interpreter = tflite.Interpreter("yolov8n/best_int8_edgetpu.tflite",
#   experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print the dimensions of the input tensor.
input_shape = input_details[0]['shape']
print("Input Tensor Shape:", input_shape)

# Print the dimensions of the output tensor.
output_shape = output_details[0]['shape']
print("Output Tensor Shape:", output_shape)

# Load an image for object detection (replace 'image.jpg' with your image path).
image = cv2.imread('input/person.jpg')

# Resize the input image to match the expected input tensor shape.
input_shape = input_details[0]['shape'][1:3]  # Extract height and width from the input shape
input_data = cv2.resize(image, (input_shape[1], input_shape[0]))

# Ensure input data is in FLOAT32 format.
input_data = np.array(input_data, dtype=np.float32) / 255.0  # Normalize to [0, 1]

# Add a batch dimension to the input data.
input_data = np.expand_dims(input_data, axis=0)

# Set the input tensor.
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference.
interpreter.invoke()

# Get the output tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

# Define the confidence threshold for detection.
# confidence_threshold = 0.5

# Parse the output tensor to count detected objects.
def count_detected_objects(output_data, confidence_threshold=0.5):
    detections = output_data[0]

    num_detected_objects = 0
    for detection in detections:
        x, y, width, height, confidence = detection[:5]  # Assuming the format [x, y, width, height, confidence]

        if confidence > confidence_threshold:
            num_detected_objects += 1

    return num_detected_objects

num_objects_detected = count_detected_objects(output_data)

# Print the output data.
print("Output Data:")
print(output_data)

# Print the number of detected objects.
print(f"Number of objects detected: {num_objects_detected}")
