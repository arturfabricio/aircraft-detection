from time import localtime, strftime
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
import bbox_utils
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
import model
from training_utilities import loss_fn, calculate_iou

warnings.simplefilter(action='ignore', category=FutureWarning)
dir_root = Path(__file__).parent.parent
train_imgs = Path(dir_root, './data/train')
annot_dir = Path(dir_root, './data/annot/rareplanes.json')
train_im_list = [z for z in os.listdir(train_imgs) if z.endswith('.png')]
data = json.load(open(annot_dir))
assert len(train_im_list) == len(data['images'])

### Hyperparameters #############


# 10e-8 is probably too small
lr = 1e-1
batchsize = 4
num_epochs = 500

# Nr of images to load, set to False to load all
start_from_image: int = 0
image_load_count: Union[int, bool] = 200

train_model = True
print_logs = True
save_model = True
augment = False
wd = 0
#################################``

### Functions ###


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
    idxs = annot_data.index.to_list()
    delete_before = idxs[:(start_from_image)]
    delete_after = idxs[(start_from_image + image_load_count):]

    annot_data.drop(delete_after, axis=0, inplace=True)
    annot_data.drop(delete_before, axis=0, inplace=True)

annot_data['path'] = annot_data.apply(
    lambda row: str(train_imgs) + "/"+row['name'][0], axis=1)
annot_data.drop(['name', 'image_id'], axis=1, inplace=True)

new_size = 128
ratio = int(512/new_size)

annot_data['image'] = annot_data.apply(resize_im_rowwise, axis=1)
annot_data['bbox'] = annot_data.apply(resize_bbox_rowwise, axis=1)
annot_data['np_bboxes'] = annot_data.apply(
    lambda row: np.array(row['bbox']).astype("float64"), axis=1)

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
        *annot_data_rotate.apply(lambda row: rotate(row, 180.0), axis=1))
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


# Prints dataset with bounding boxes
# image_id = 0
# for i in range(0, len(annot_data['image'])):
#     print(annot_data['bbox'][i])
#     visualization.display_bboxs(annot_data['image'][i],
#                                 annot_data['bbox'][i])


print('nr of bboxes ', len(model.bboxs))
# print(model.np_bboxs)
# visualization.display_bboxs(annot_data['image'][0], model.np_bboxs)


annot_data['target_vector'] = annot_data.apply(lambda row: calculate_iou(row['bbox']), axis=1)

# # Prints the target vectors bounding boxes
# for i in range(0, len(annot_data['image'])):
#     visualization.display_bbox_target_vector(
#         annot_data['image'][i], annot_data['target_vector'][i], model.np_bboxs, 0.5)

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

train_dl = DataLoader(train_ds, batch_size=batchsize,
                      shuffle=True, drop_last=False)
valid_dl = DataLoader(valid_ds, batch_size=batchsize,
                      shuffle=True, drop_last=False)


def get_local_time() -> str:
    return strftime("%Y-%m-%d %H:%M:%S", localtime())


start_time: str = get_local_time().replace(':', '_')

if train_model == False:
    exit(0)


model_directory = Path(dir_root,  f'./data/model/{start_time}/')
model_directory.mkdir(parents=True, exist_ok=True)

LOG_PATH = Path(
    model_directory, f'./logs.log')


def print_to_logs(to_print: str):
    with open(LOG_PATH, 'a') as file:
        file.write(
            f'{get_local_time()} - ' + to_print + '\n')

aircraft_model = model.AircraftModel().double()

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')  # use cuda or cpu
print("Used device: ", device)
aircraft_model.to(device)

# out = aircraft_model(torch.randn(batchsize, 3, 128, 128, device=device))
# print("Output shape:", out.size())
# print(f"Output logits:\n{out.detach().cpu().numpy()}")
optimizer = optim.SGD(aircraft_model.parameters(), lr, weight_decay=wd, momentum=0)

convert_tensor = transforms.ToTensor()
step = 0
aircraft_model.train()

train_accuracies = []
valid_accuracies = []

if print_logs == True:
    titles = ['learning rate', 'batchsize', 'epochs',
              'train_images', 'val_images', 's', 'weigth decay', 'optimizer']
    hyper = [lr, batchsize, num_epochs, len(
        train_ds), len(valid_ds), model.s, wd, optimizer]
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

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = aircraft_model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    with torch.no_grad():
        aircraft_model.eval()
        for inputs, targets in valid_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = torch.permute(inputs, (0, 3, 1, 2))

            # forward + backward + optimize
            output = aircraft_model(inputs)
            loss = loss_fn(output, targets)

            val_losses.append(loss.item())

        aircraft_model.train()

    print_to_logs('Training Loss: ' + str(np.mean(np.array(train_losses))))
    print_to_logs('Validation Loss: ' + str(np.mean(np.array(val_losses))))

    if save_model == True:
        DIR_PATH = Path(model_directory, './model/')
        DIR_PATH.mkdir(parents=True, exist_ok=True)
        FILE_PATH = Path(DIR_PATH, f"./epoch_{str(epoch)}.pth")
        torch.save(aircraft_model.state_dict(), FILE_PATH)

print_to_logs("Finished training.")
