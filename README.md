# Google Coral Dev Board Mini Human Detection

This repository contains the necessary code and resources to implement human detection on the Google Coral Dev Board Mini. Follow the instructions below to set it up.

## Prerequisites

- Prepare the database for saving the inference result and website to see the inference result by following setup provided [here](https://github.com/DidaPrasetyo/coral_log).
- Set up the Google Coral Dev Board Mini by following the steps in the [official documentation](https://coral.ai/docs/dev-board-mini/get-started).

## Detection Test Setup

1. Clone this repository

```
git clone https://github.com/DidaPrasetyo/coral
```

2. Install OpenCV and mysql-connector-python

```
cd coral/preparation
bash install_opencv.sh

pip3 install mysql-connector-python
```

3. Check the model and label in the `model`  and `label`  folders.

```
- model
  - efficientdet_lite0_320_ptq_edgetpu.tflite
  - efficientdet_lite1_384_ptq_edgetpu.tflite
  - efficientdet_lite2_448_ptq_edgetpu.tflite
  - efficientdet_lite3_512_ptq_edgetpu.tflite
  - efficientdet_lite3x_640_ptq_edgetpu.tflite
  - ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite
  - ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
  - ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite
  - tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_edgetpu.tflite
  - tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite
- label
  - coco_labels.txt
```

All this model and label can be downloaded manually from this [link](https://coral.ai/models/object-detection/).

## Usage

1. Run the inference using detection script from this following command.

```
python3 detect.py -m {model} -i {input source} --host {ip address}
```

| Argument | Required | Description |
| --- | --- | --- |
| -m, --model | Yes | Specifies the model to be used. There are 7 models that can be used, namely mobilenetv1, mobilenetv2, mobileedet, efficientdet0, efficientdet1, efficientdet2, efficientdet3 |
| -i, --input | Yes | Specifies the input sources that the model will interfere with. Input can be camera index, video file path, and IP camera url |
| -t, --threshold | Optional | Specifies the threshold value of the detection result score to be displayed, default: 0.3 |
| --host | Yes | Specifies the IP of the database server host used to store detection logs and images |
| --debug | Optional | Valued true or false which is used to display information on the detection results image such as FPS, inference time, person count, score and label |
| -resw | Optional | Specifies width of resolution input source, default: 1280 |
| -resh | Optional | Specifies heigh of resolution input source, default: 720 |

2. The program will transmit the inference result to the database, which can be viewed on the website deployed based in this [repository](https://github.com/DidaPrasetyo/coral_log/tree/master?tab=readme-ov-file#website-deployment).
