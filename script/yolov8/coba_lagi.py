from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Path to the TensorFlow Lite model file (.tflite)
MODEL_PATH = 'path/to/your/model.tflite'

# Load the model and allocate tensors.
interpreter = cv2.dnn.readNetFromTensorflow(MODEL_PATH)

# RTSP stream URL
RTSP_URL = 'rtsp://your_rtsp_stream_url_here'

def gen_frames():
    cap = cv2.VideoCapture(RTSP_URL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the input image for object detection
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True)
        interpreter.setInput(blob)

        # Perform inference
        detections = interpreter.forward()

        # Process the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Adjust the confidence threshold as needed
                class_id = int(detections[0, 0, i, 1])
                label = f'Class {class_id}'
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
