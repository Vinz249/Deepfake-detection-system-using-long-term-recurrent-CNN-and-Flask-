import cv2
import torch
from flask import Flask, render_template, request, jsonify
from torch_utils import frame_extract, create_face_video, get_data_transforms, preprocess, predict, output_predict

# define model class

from torch import nn
from torchvision import models


class MyModel(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(MyModel, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)  # Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))


model = MyModel(2)

device = torch.device('cpu')
PATH = r"C:\\Users\\ANKUR SINGH\\Desktop\\PROJECTS\\Deepfake Detector\\Deepfake-detection-system-using-long-term-recurrent-CNN-and-Flask-\\app\\model(1).pth"
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

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
            def frame_extract(path):
                vidObj = cv2.VideoCapture(path)
                success = 1
                while success:
                    success, image = vidObj.read()
                    if success:
                        yield image

            input_video_path = "./video_input/"
            output_video_path = "./video_output/"
            create_face_video(input_video_path, output_video_path)
            video_tensor = preprocess(output_video_path)
            predicted_labels = predict(video_tensor, model)
            output_predict(predicted_labels)
        except:
            return jsonify({'error': 'error during prediction'})

    return render_template('index.html')


app.run(debug=True)
