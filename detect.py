import argparse
import cv2
import os
import time

import mysql.connector
from mysql.connector import Error

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-tk', '--top_k', type=int, default=3,
    #                     help='number of categories with highest score to display')
    parser.add_argument('-m', '--model', required=True,
                        choices=["mobilenetv1", 
                                 "mobilenetv2", 
                                 "mobiledet", 
                                 "efficientdet0", 
                                 "efficientdet1", 
                                 "efficientdet2", 
                                 "efficientdet3"], 
                        help='Choose the available model')
    parser.add_argument('-i', '--input', required=True,
                        help='File path of video file or camera index (0 for default camera)')
    parser.add_argument('-t', '--threshold', type=float, default=0.3,
                        help='classifier score threshold')
    parser.add_argument('--host', required=True,
                        help='Host ip to connect to database')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='hide and show text information in video')
    parser.add_argument('-resw', '--width', default=1280,
                        help='set width for the video input')
    parser.add_argument('-resh', '--height', default=720,
                        help='set height for the video input')
    args = parser.parse_args()

    model_name = args.model
    label = 'label/coco_labels.txt'

    if model_name == 'mobilenetv1':
        model = 'model/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite'

    elif model_name == 'mobilenetv2':
        model = 'model/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'

    elif model_name == 'mobiledet':
        model = 'model/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'

    elif model_name == 'efficientdet0':
        model = 'model/efficientdet_lite0_320_ptq_edgetpu.tflite'

    elif model_name == 'efficientdet1':
        model = 'model/efficientdet_lite1_384_ptq_edgetpu.tflite'

    elif model_name == 'efficientdet2':
        model = 'model/efficientdet_lite2_448_ptq_edgetpu.tflite'

    elif model_name == 'efficientdet3':
        model = 'model/efficientdet_lite3_512_ptq_edgetpu.tflite'


    print('Loading {} with {} labels.'.format(model, label))
    interpreter = make_interpreter(model)
    interpreter.allocate_tensors()
    labels = read_label_file(label)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.input)

    start_time = time.time()
    time_elapsed = 0

    fps_start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame / End of the frame")
                break

            if time_elapsed >= 3600:
                print("1 hour elapsed. Program done.")
                break

            cv2_im = frame

            if (time.time() - fps_start_time) > 0 :
                elapsed_time = time.time() - fps_start_time
                fps = 1 / elapsed_time
                fps_start_time = time.time()

            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

            inference_start_time = time.perf_counter()
            run_inference(interpreter, cv2_im_rgb.tobytes())
            inference_time = time.perf_counter() - inference_start_time

            # objs = get_objects(interpreter, args.threshold)[:args.top_k] # limit number detected
            objs = get_objects(interpreter, args.threshold)

            detected_persons = 0

            if objs:
                print('Detected Objects:')
                for obj in objs:
                    if labels.get(obj.id, obj.id) == "person":
                        detected_persons += 1
                        # print(f"{labels.get(obj.id, obj.id)} - Score: {obj.score:.2f}")
                        
                        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels, args.debug)
                        
            if args.debug:
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
                cv2.putText(frame, f"Persons: {detected_persons}", (175, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
                cv2.putText(frame, f"inference TIme: {(inference_time * 1000):.4f} ms", (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            
            if detected_persons > 0:
                dim = (int(args.width), int(args.height))
                frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

                blob_img = convert_image_to_blob(frame)
                upload_image_to_mysql(args.host, time.strftime('%Y-%m-%d %H:%M:%S'), detected_persons, blob_img)

            print(f"FPS: {fps:.2f}")
            print(f"Detected Person: {detected_persons}")
            time_elapsed = time.time() - start_time
            print(f"Elapsed Time : {time_elapsed} seconds")

    except KeyboardInterrupt:
        print("Inference process interrupted.")

    finally:
        cap.release()

def append_objs_to_img(cv2_im, inference_size, objs, labels, debug, target_label="person"):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        if labels.get(obj.id, obj.id) == target_label:
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)

            label = '{:.2f} {}'.format(obj.score, labels.get(obj.id, obj.id))

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)

            if debug:
                cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return cv2_im

def convert_image_to_blob(frame):
    # Convert image to binary format
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_blob = img_encoded.tobytes()
    return img_blob

def upload_image_to_mysql(host, timestamp, count, blob_data):
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host=host,
            database='coral',
            user='coral',
            password=''
        )

        if connection.is_connected():
            cursor = connection.cursor()

            query = "INSERT INTO detection_log (detect_time, count, img) VALUES (%s, %s, %s)"
            cursor.execute(query, (timestamp, count, blob_data))

            connection.commit()
            print("Image uploaded to MySQL database")

    except Error as e:
        print(f"Error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed")

if __name__ == '__main__':
    program_start = time.time()
    main()
    program_finish = time.time() - program_start
    print(f'Program duration: {program_finish} s')
