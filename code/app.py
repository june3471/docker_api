import json
import numpy as np
import requests
from flask import Flask, request, jsonify,escape
import os
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import decode_predictions,preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image


# from flask_cors import CORS
app = Flask(__name__)


# Testing URL
@app.route('/', methods=['GET']) # 호출 방식 get post 둘다 허용
def hello_world():
    uid = request.args.get('uid')
    return 'Hello, {escape(uid)} by GET!'


@app.route('/inception/predict/',methods=['POST'])
def image_classifier():
    model_ip = os.environ['inception_ip']
    #ni.ifaddresses('eth0')
    #ip = ni.ifaddresses('eth0')[ni.AF_INET][0]['addr']
    #ip = ip.split('.')
    #ip[-1] = '01'
    #gate_way = '.'.join(ip)
    address = 'http://%s:8501/v1/models/inception:predict'%(model_ip)
    
    content = request.get_json()
    # from json get img path
    img_path = content['instances']
    # img loading from path
    img = image.load_img(img_path, target_size=(224, 224))
    # img preprocessing
    x = image.img_to_array(img)
    
    x = preprocess_input(x)
    data = {
        "instances": [{'input_1': x.tolist()}]
    }
    
    # Making POST request (POST 방식으로 address에 requsets)
    result = requests.post(address, json=data)
    
    # Decoding results from TensorFlow Serving server
    pred = json.loads(result.content.decode('utf-8'))
    
    # Returning JSON response to the frontend
    return jsonify(decode_predictions(np.array(pred['predictions']))[0])

@app.route('/yolov3/predict/',methods=['POST'])
def object_detection():
    model_ip = os.environ['yolov3_ip']
    address = 'http://%s:8501/v1/models/yolov3:predict'%(model_ip)
    content = request.get_json()
    img_path = content['instances']
    classes_names = [c.strip() for c in open(content['class']).readlines()]
    size = int(content['size'])
    threshold = float(content['thresh'])
    
    
    img = tf.image.decode_image(open(img_path, 'rb').read(), channels=3)
    x = tf.image.resize(img, (size, size))
    x = x / 255
    
    data = {
        "instances": [{'input': x.numpy().tolist()}]
    }
    
    # Making POST request (POST 방식으로 address에 requsets)
    outputs = requests.post(address, json=data)
    
    # Decoding results from TensorFlow Serving server
    outputs = json.loads(outputs.content.decode('utf-8'))
    outputs = outputs["predictions"][0]
    boxes, scores, classes, nums = outputs["yolo_nms"], outputs[
        "yolo_nms_1"], outputs["yolo_nms_2"], outputs["yolo_nms_3"]
    
    results = {}
    obj = []
    score = []
    box = []
    for i in range(nums):
        if scores[i]>= threshold:
            obj.append(classes_names[int(classes[i])])
            score.append(scores[i])
            box.append(boxes[i])
    results['object'] = obj
    results['score'] = score
    results['box'] = box
        
    return results

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')