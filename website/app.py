# Import necessary libraries
from flask import Flask, render_template, Response, request
import cv2

app = Flask(__name__)

def video_capture(source):
    return cv2.VideoCapture(source)

def use_camera(cam):
    global camera1, camera2

    if cam == "camera1":
        camera1 = video_capture(0)
    elif cam == "camera2":
        camera2 = video_capture(2)

# Initialize counters for active clients for each camera
active_clients_camera1 = 0
active_clients_camera2 = 0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # Get the camera choice from the request
    camera_choice = request.args.get("camera")

    global active_clients_camera1
    global active_clients_camera2

    if camera_choice == "camera1":
        if active_clients_camera1 == 0:
            use_camera("camera1")
        active_clients_camera1 += 1
        print("Cam1 = ", active_clients_camera1)
        print("Cam2 = ", active_clients_camera2)
        return Response(generate_frames(camera1, camera_choice), mimetype="multipart/x-mixed-replace; boundary=frame")
    elif camera_choice == "camera2":
        if active_clients_camera2 == 0:
            use_camera("camera2")
        active_clients_camera2 += 1
        print("Cam1 = ", active_clients_camera1)
        print("Cam2 = ", active_clients_camera2)
        return Response(generate_frames(camera2, camera_choice), mimetype="multipart/x-mixed-replace; boundary=frame")

def generate_frames(camera, camera_choice):
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/disconnect")
def disconnect():
    camera_choice = request.args.get("camera")

    global active_clients_camera1
    global active_clients_camera2

    if camera_choice == "camera1":
        active_clients_camera1 -= 1
        if active_clients_camera1 == 0:
            camera1.release()
        else:
            use_camera("camera1")
    elif camera_choice == "camera2":
        active_clients_camera2 -= 1
        if active_clients_camera2 == 0:
            camera2.release()
        else:
            use_camera("camera2")

    print("Client Disconnected")
    print("Cam1 = ", active_clients_camera1)
    print("Cam2 = ", active_clients_camera2)
    return "Success!\n"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
