import os
import json
import random
import pandas as pd
import skimage.io
import numpy as np
from sklearn import metrics
from torchvision import transforms
import time
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool1d
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, BatchNorm1d, BatchNorm2d
import bbox_utils
from bbox_utils import BBOX
import func_utils
import warnings
from func_utils import *
import importlib
from pathlib import Path
import datetime
import cv2
import matplotlib.patches as patches
import colorsys
import math
import datetime
from data_aug import *
from bbox_aug import *
import visualization
from typing import Union

warnings.simplefilter(action='ignore', category=FutureWarning)
dir_root = Path(__file__).parent.parent
train_imgs = Path(dir_root, './data/train')
annot_dir = Path(dir_root, './data/annot/rareplanes.json')
train_im_list = [z for z in os.listdir(train_imgs) if z.endswith('.png')]
data = json.load(open(annot_dir))
assert len(train_im_list) == len(data['images'])

### Hyperparameters #############


def loss_fn(output, target):
    loss = torch.mean((output-target)**2)
    sum_loss = torch.sum(target)
    if sum_loss == 0:
        return loss
    else:
        return torch.divide(loss, sum_loss)


s = 4
lr = 10e-4
batchsize = 64
num_epochs = 75
# Nr of images to load, set to False to load all
image_load_count: Union[int, bool] = False
train_model = True
print_logs = True
save_model = True
augment = True
wd = 10e-4
#################################

### Functions ###


def stack_to_numpy(row):
    out = np.array(row['bbox']).astype("float64")
    return out


def resize_im_rowwise(row):
    return bbox_utils.transformsImg(row['path'], new_size)


def resize_bbox_rowwise(row):
    return np.array(bbox_utils.transformsBbox(row['bbox'], ratio), dtype='float64')

### Processing ###


print("Started running!")

with open(annot_dir) as json_data:
    data = json.load(json_data)

annot_data = pd.DataFrame(data['categories'])
annot_data.drop(['loc_id', 'cat_id', 'location', 'role', 'role_id', 'is_plane', 'num_engines', 'propulsion', 'canards', 'num_tail_fins', 'wing_position', 'wing_type',
                'length', 'wingspan', 'area', 'faa_wingspan_class', 'Public_Train', 'Public_Test', 'partialDec', 'truncated', 'new_area', 'area_pixels', 'id'], axis=1, inplace=True)
annot_data.rename(columns={"image_fname": "name"}, inplace=True)

annot_data = annot_data.groupby(['image_id']).agg(
    tuple).applymap(np.array).reset_index()

if image_load_count != False:
    annot_data.drop(annot_data.index.to_list()[
                    image_load_count:], axis=0, inplace=True)

annot_data['path'] = annot_data.apply(
    lambda row: str(train_imgs) + "/"+row['name'][0], axis=1)
annot_data.drop(['name', 'image_id'], axis=1, inplace=True)

new_size = 128
ratio = int(512/new_size)

annot_data['image'] = annot_data.apply(resize_im_rowwise, axis=1)
annot_data['bbox'] = annot_data.apply(resize_bbox_rowwise, axis=1)
annot_data['np_bboxes'] = annot_data.apply(
    lambda row: stack_to_numpy(row), axis=1)

if augment == True:

    annot_data_rscale = annot_data.copy()
    annot_data_translate = annot_data.copy()
    annot_data_rotate = annot_data.copy()

    print("Init time: ", datetime.datetime.now())
    print("Initial amount of images: ", len(annot_data['image']))

    def rotate(row, angle):
        new_img, new_bboxs = RandomRotate(angle)(
            row['image'], row['np_bboxes'])
        return new_img, new_bboxs

    annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(
        *annot_data_rotate.apply(lambda row: rotate(row, 30), axis=1))
    annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,60), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,90), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,120), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,150), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,180), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,210), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,240), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,270), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,310), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)

    print("Final rotate time: ", datetime.datetime.now())

    def scale(row, ratio):
        new_img, new_bboxs = RandomScale(ratio, diff=True)(
            row['image'], row['np_bboxes'])
        return new_img, new_bboxs

    annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(
        *annot_data_rscale.apply(lambda row: scale(row, 0.2), axis=1))
    annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: scale(row,0.4), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: scale(row,0.6), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: scale(row,0.8), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)

    print("Final scale time: ", datetime.datetime.now())

    def translate(row, ratio):
        new_img, new_bboxs = RandomTranslate(
            ratio, diff=True)(row['image'], row['np_bboxes'])
        return new_img, new_bboxs

    annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(
        *annot_data_rscale.apply(lambda row: translate(row, 0.2), axis=1))
    annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: translate(row,0.4), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: translate(row,0.6), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: translate(row,0.8), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)

    print("Final translate time: ", datetime.datetime.now())

    annot_data.drop(['np_bboxes', 'path'], axis=1, inplace=True)
    # plotted_img = draw_rect(annot_data['image'][len(annot_data['bbox'])-1].copy(), annot_data['bbox'][len(annot_data['bbox'])-1].copy())
    # plt.imshow(plotted_img)
    # plt.show()

    print("Augmented amount of images: ", len(annot_data['image']))
    print("Final time: ", datetime.datetime.now())

# image_id = 0

# for i in range(image_load_count*2, len(annot_data['image'][image_id])):
#     print(annot_data['bbox'][i])
#     visualization.plot(annot_data['image'][i],
#                        annot_data['bbox'][i])

bboxs = bbox_utils.generate(s, 130//4, 10, (128, 128))

np_bboxs = np.asarray(list(
    map(lambda BBOX: [BBOX.arr[0], BBOX.arr[1], BBOX.arr[2], BBOX.arr[3]], bboxs)))


def calculate_iou_rowwise(row):
    target_vector = np.zeros(len(np_bboxs))
    for bbox in row['bbox']:
        xA = np.maximum(np_bboxs[:, 0], bbox[0])
        yA = np.maximum(np_bboxs[:, 1], bbox[1])
        xB = np.minimum(np_bboxs[:, 2], bbox[2])
        yB = np.minimum(np_bboxs[:, 3], bbox[3])
        interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
        boxAArea = (np_bboxs[:, 2] - np_bboxs[:, 0] + 1) * \
            (np_bboxs[:, 3] - np_bboxs[:, 1] + 1)
        boxBArea = (bbox[2] - bbox[0] + 1) * \
            (bbox[3] - bbox[1] + 1)
        iou = np.divide(interArea, (np.subtract(
            np.add(boxAArea, boxBArea), interArea)))
        b = np.zeros_like(iou)
        b[np.argmax(iou, axis=0)] = 1
        target_vector = target_vector + b
    # print(sum(target_vector))
    return target_vector


annot_data['target_vector'] = annot_data.apply(calculate_iou_rowwise, axis=1)
# display_bbox_target_vector(annot_data)

annot_data = annot_data.reset_index()
X = annot_data['image']
Y = annot_data['target_vector']
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.15, random_state=42)


class AircraftDataset(Dataset):
    def __init__(self, images, y):
        self.images = images.values
        self.y = y.values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        y = self.y[idx]
        return path, y


train_ds = AircraftDataset(X_train, y_train)
valid_ds = AircraftDataset(X_val, y_val)

batch_size = 64
train_dl = DataLoader(train_ds, batch_size=batch_size,
                      shuffle=True, num_workers=8, drop_last=False)
valid_dl = DataLoader(valid_ds, batch_size=batch_size,
                      shuffle=False, num_workers=8, drop_last=False)

start_time = str(time.time())

if train_model == False:
    exit(0)


class AircraftModel(nn.Module):
    def __init__(self):
        super(AircraftModel, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(3, 192, kernel_size=7, stride=2),
            nn.LeakyReLU(0.1),
            MaxPool2d(2, 2),
            Conv2d(192, 256, 3, 1),
            nn.LeakyReLU(0.1),
            MaxPool2d(2, 2),
            # Conv2d(256,128,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(128,256,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(256,256,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(256,512,1,1),
            # nn.LeakyReLU(0.1),
            # MaxPool2d(2,2),
            # Conv2d(512,256,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(256,512,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(512,256,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(256,512,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(512,256,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(256,512,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(512,256,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(256,512,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(512,1024,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(1024,512,1,1),
            # nn.LeakyReLU(0.1),
            # # MaxPool2d(2,2),
            # Conv2d(512,1024,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(1024,512,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(512,1024,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(1024,512,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(512,1024,1,1),
            # nn.LeakyReLU(0.1),
            # Conv2d(1024,1024,1,2),
            # nn.LeakyReLU(0.1),
            # Conv2d(1024,1024,1,1),
            # nn.LeakyReLU(0.1),
            Conv2d(256, 512, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5)
        )

        self.connected = nn.Sequential(
            # (128*128,out_features=1024, bias=False),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.Linear(64, out_features=len(bboxs), bias=False)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.connected(x)
        return x


model_directory = Path(dir_root,  f'./data/model/{start_time}/')
model_directory.mkdir(parents=True, exist_ok=True)

LOG_PATH = Path(
    model_directory, f'./logs.log')


def print_to_logs(to_print: str):
    with open(LOG_PATH, 'a') as file:
        file.write(f'{time.time()} - ' + to_print + '\n')


model = AircraftModel().double()
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')  # use cuda or cpu
print("Used device: ", device)
model.to(device)

# out = model(torch.randn(batchsize, 3, 128, 128, device=device))
# print("Output shape:", out.size())
# print(f"Output logits:\n{out.detach().cpu().numpy()}")
optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)

convert_tensor = transforms.ToTensor()
step = 0
model.train()

train_accuracies = []
valid_accuracies = []

if print_logs == True:
    titles = ['learning rate', 'batchsize', 'epochs',
              'train_images', 'val_images', 's', 'weigth decay', 'optimizer']
    hyper = [lr, batchsize, num_epochs, len(
        train_ds), len(valid_ds), s, wd, optimizer]
    PATH_HYPER = Path(
        model_directory, "hyper_parameters.csv")
    with open(PATH_HYPER, 'a', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(titles)
        wr.writerow(hyper)


for epoch in range(num_epochs):
    print_to_logs("Epoch number: " + str(epoch))

    train_losses = []
    val_losses = []

    for inputs, targets in train_dl:
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = torch.permute(inputs, (0, 3, 1, 2))
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)  # There's an error here
        loss.backward()
        optimizer.step()

        train_losses.append(loss.detach().cpu().numpy())

    with torch.no_grad():
        model.eval()
        for inputs, targets in valid_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = torch.permute(inputs, (0, 3, 1, 2))

            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)

            val_losses.append(loss.detach().cpu().numpy())
        model.train()

    print_to_logs('Training Loss: ' + str(np.mean(np.array(train_losses))))
    print_to_logs('Validation Loss: ' + str(np.mean(np.array(val_losses))))

    if save_model == True:
        DIR_PATH = Path(model_directory, './model/')
        DIR_PATH.mkdir(parents=True, exist_ok=True)
        FILE_PATH = Path(DIR_PATH, f"./epoch_{str(epoch)}.pth")
        torch.save(model.state_dict(), FILE_PATH)

print_to_logs("Finished training.")
