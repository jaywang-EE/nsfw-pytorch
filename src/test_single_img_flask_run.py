# -*- coding: utf-8 -*-

import argparse
import time

from model import resnet
from model.dpn import dpn92

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
from model.utils import load_filtered_state_dict, SaveBestModel, AverageMeter, accuracy
from PIL import Image
import glob
import os
import cv2
import numpy as np

# hyperparameters
default_class=['drawings', 'hentai', 'neutral', 'porn', 'sexy']

USE_GPU     = True
NUM_CLASSES = len(default_class)
IMAGE_SIZE  = 299
MODEL_PATH  = "./models/resnet50-19c8e357.pth"

# model loading
print("model loading start")
model = resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], NUM_CLASSES)    
model.eval()
if USE_GPU:        
    cudnn.enabled = True 
    softmax = nn.Softmax().cuda()
    model.cuda()
    saved_state_dict = torch.load(MODEL_PATH)
else:
    softmax = nn.Softmax()        
    saved_state_dict = torch.load(MODEL_PATH, map_location='cpu')
load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=False, gpu=cudnn.enabled)

transformations = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
print("model loading end")

from flask import Flask
import json
import os

app = Flask(__name__)

@app.route('/<path:image_path>')
def index(image_path):
    print(os.listdir())
    print("image_path:",image_path)

    imgs = torch.FloatTensor(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    if USE_GPU:
        imgs = imgs.cuda()

    #image loading
    imgs[0] = transformations(Image.open(image_path).convert("RGB"))

    pred = model(imgs)
    pred = softmax(pred)
    print(pred.cpu().detach().numpy())
    _, pred_1 = pred.topk(1, 1, True, True)
    c = default_class[pred_1.cpu().numpy()[0][0]]
    print("{} -- {}".format(image_path, c))
    result = {"filename": image_path, "result":c}
    return json.dumps(result)


if __name__ == '__main__':
    app.run(debug=True)
