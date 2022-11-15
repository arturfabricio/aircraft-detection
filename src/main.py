import os
import json
import pandas as pd
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from dataset import *

data_root = os.getcwd()
print(data_root)
train_imgs = os.path.join(data_root, 'data/train')
annot_dir = os.path.join(data_root, 'data/annot/rareplanes.json')
train_im_list = [z for z in os.listdir(train_imgs) if z.endswith('.png')]
f = open(annot_dir)
data = json.load(f)
print(len(train_im_list))
print(len(data['images']))
assert len(train_im_list) == len(data['images'])

final, img_name = imgs_annot_aggregator(1,train_im_list, data)
result = bbox_points(final,train_imgs)
result.head()

print("Total number of images: ",len(train_im_list))
final_data, img_name_data = imgs_annot_aggregator(len(train_im_list),train_im_list, data)
result_data = bbox_points(final_data, train_imgs)
print("Loaded all the data: ",np.shape(result_data))

class_dict = {'Small Civil Transport/Utility': 0, 
              'Medium Civil Transport/Utility': 1, 
              'Large Civil Transport/Utility': 2, 
              "Military Transport/Utility/AWAC": 3,
              "Military Fighter/Interceptor/Attack": 4,
              "Military Bomber": 5,
              "Military Trainer": 6
              }

result_data['class'] = result_data['class'].apply(lambda x: class_dict[x])
 
new_size = 128
ratio = 512/new_size

result_data = result_data.reset_index()
X = result_data[['file_path','bbox']]
Y = result_data['class']

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

train_ds = AircraftDataset(X_train['file_path'],X_train['bbox'] ,y_train, transforms=True)
valid_ds = AircraftDataset(X_val['file_path'],X_val['bbox'],y_val)

batch_size = 16
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)

class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.classifier = nn.Sequential(nn.BatchNorm1d(64), nn.Linear(64, 7))
        self.bb = nn.Sequential(nn.BatchNorm1d(64), nn.Linear(64, 4))
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

def val_metrics(model, valid_dl, C=1000):
    count = 0
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    for x, y_class, y_bb in valid_dl:
        #print('Inside loop (val): ',count)
        batch = y_class.shape[0]
        x = x.cuda().float()
        x = torch.permute(x, (0, 3, 1, 2))
        y_class = y_class.cuda()
        y_bb = y_bb.cuda().float()
        out_class, out_bb = model(x)
        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
        loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb/C
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
        count = count + 1 
    return sum_loss/total, correct/total

def train_epocs(model, optimizer, train_dl, val_dl, epochs=10,C=1000):
    idx = 0
    count = 0
    for i in range(epochs):
        #print('Epoch number (train)',i,' of ', epochs)
        model.train()
        total = 0
        sum_loss = 0
        for x, y_class, y_bb in train_dl:
           # print('Inside loop: ',count)
            batch = y_class.shape[0]
            x = x.cuda().float()
            y_class = y_class.cuda()
            y_bb = y_bb.cuda().float()
            x = torch.permute(x, (0, 3, 1, 2))
            out_class, out_bb = model(x)
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
            count = count + 1 
        train_loss = sum_loss/total
        val_loss, val_acc = val_metrics(model, valid_dl, C)
        print("train_loss %.3f val_loss %.3f val_acc %.3f epoch %.3f" % (train_loss, val_loss, val_acc, i))
    return sum_loss/total

use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")  

learning_rate = 0.006
model = BB_model().cuda()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=learning_rate)
print("Used learning rate: ", learning_rate)

print(model)

train_epocs(model, optimizer, train_dl, valid_dl, epochs=15)