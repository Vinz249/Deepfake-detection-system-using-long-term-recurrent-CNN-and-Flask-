import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import random
import pandas as pd
import glob
import matplotlib.pyplot as plt

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
PATH = "C:\\Users\\ANKUR SINGH\\Desktop\\PROJECTS\\Deepfake Detector\\Deepfake-detection-system-using-long-term-recurrent-CNN-and-Flask-\\app\\model(1).pth"
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()
