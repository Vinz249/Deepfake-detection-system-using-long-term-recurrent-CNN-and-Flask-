import cv2
import torch
from flask import Flask, render_template, request, jsonify
from torch_utils import frame_extract, create_face_video, get_data_transforms, preprocess, predict, output_predict

# define model class

from torch import nn
from torchvision import models


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'mp4','webm','avi','mov','wmv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def webpage():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        fileInput = request.files.get('fileInput')
        video_path = "./video_input/" + fileInput.filename
        fileInput.save(video_path)
        if fileInput is None or fileInput.filename=="":
            return jsonify({'error': 'no file'})
        if not allowed_file(fileInput.filename):
            return jsonify({'error': 'format is not supported'})
        try:
            

            input_video_path = " "
            output_video_path = " "
            create_face_video(input_video_path, output_video_path)
            video_tensor = preprocess(output_video_path)
            predicted_labels = predict(video_tensor)
            output_predict(predicted_labels)
        except:
            return jsonify({'error': 'error during prediction'})

    return render_template('index.html')


app.run(debug=True)
