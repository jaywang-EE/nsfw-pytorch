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

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--image_path', dest='image_path', help='Image path', type=str)
    args = parser.parse_args()
    return args

def classify_single_image(image_path):
    # model loading
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

if __name__ == '__main__':
    args = parse_args()
    classify_single_image(args.image_path)
    