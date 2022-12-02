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
from data_aug import *
from bbox_aug import *

warnings.simplefilter(action='ignore', category=FutureWarning)
dir_root = Path(os.getcwd()).parent
train_imgs = Path(dir_root, './data/train')
annot_dir = Path(dir_root, './data/annot/rareplanes.json')
train_im_list = [z for z in os.listdir(train_imgs) if z.endswith('.png')]
f = open(annot_dir)
data = json.load(f)
assert len(train_im_list) == len(data['images'])

### Hyperparameters #############
def loss_fn(output, target):
    num_planes = torch.sum(target)
    return torch.where(condition=num_planes > 0,
                       x=torch.mean((output - target) ** 2) /
                       torch.sum(target),
                       y=torch.mean((output - target) ** 2))
s = 10
lr = 0.0001
batchsize = 64
num_epochs = 2
validation_every_steps = 1
#################################

### Functions ###

def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows, cols, *_ = x.shape
    Y = np.zeros([rows, cols])
    bb = bb.astype(int)
    Y[bb[1]:(bb[1]+bb[3]), bb[0]:(bb[0]+bb[2])] = 1.
    return Y

def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
    return np.array([x[2], x[3], x[4], x[5]])

def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols) == 0:
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def crop(im, r, c, target_r, target_c):
    return im[r:r+target_r, c:c+target_c]

def random_crop(x, r_pix=8):
    """ Returns a random crop"""
    r, c, *_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c, *_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], color=color,
                         fill=False, lw=1)

def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))

def image_merger(result_data):
    final_data_frame = pd.DataFrame()
    #print("Original length of result_data: ", len(result_data['file_path']))
    imgs_used = []

    for i in range(len(result_data['file_path'])):
        checker = True
        img_path_test = result_data['file_path'][i]

        for x in range(len(imgs_used)):
            if imgs_used[x] == img_path_test:
                checker = False

        if checker == True:
            imgs_used.append(img_path_test)
            indx_number = []

            for j in range(len(result_data['file_path'])):
                if img_path_test == result_data['file_path'][j]:
                    indx_number.append(j)

            final_bbxs = []
            for h in range(len(indx_number)):
                bbox = result_data['bbox'][indx_number[h]]
                final_bbxs.append(bbox_utils.BBOX(
                    bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]))

            final_data_frame = final_data_frame.append(
                {'path': img_path_test, 'final_bbx': final_bbxs}, ignore_index=True)

    return final_data_frame

### Processing ###
with open(annot_dir) as json_data:
    data = json.load(json_data)

annot_data = pd.DataFrame(data['categories'])
annot_data.drop(['loc_id', 'cat_id', 'location', 'role', 'role_id', 'is_plane', 'num_engines', 'propulsion', 'canards', 'num_tail_fins', 'wing_position', 'wing_type',
                'length', 'wingspan', 'area', 'faa_wingspan_class', 'Public_Train', 'Public_Test', 'partialDec', 'truncated', 'new_area', 'area_pixels', 'id'], axis=1, inplace=True)
annot_data.rename(columns={"image_fname": "name"}, inplace=True)

annot_data = annot_data.groupby(['image_id']).agg(
    tuple).applymap(np.array).reset_index()

# applymap(np.array)
annot_data.drop(annot_data.index.to_list()[3:], axis=0, inplace=True)

annot_data['path'] = annot_data.apply(
    lambda row: str(train_imgs) + "/"+row['name'][0], axis=1)
annot_data.drop(['name', 'image_id'], axis=1, inplace=True)

# annot_data['bbox'] = annot_data.apply(
#     lambda row: row['bbox'].squeeze(), axis=1)

print(annot_data.head())

new_size = 128
ratio = int(512/new_size)

# add to pandas df, reduced images, and reduced boudning boxes


def resize_im_rowwise(row):
    return bbox_utils.transformsImg(row['path'], new_size)


def resize_bbox_rowwise(row):
    return bbox_utils.transformsBbox(row['bbox'], ratio)


def rotate90img(row):
    return np.rot90(row['image'])


def rotate180img(row):
    return np.rot90(np.rot90(row['image']))


def rotate270img(row):
    return np.rot90((row['image']), k=-1)


def rotatebbox(bbox, rot):
    new_bbx = []
    for i in range(len(bbox)):
        ref = np.zeros([new_size, new_size])
        ox, oy = bbox[i][0], bbox[i][1]
        ref[oy, ox] = 1
        if rot == 90:
            newref = np.rot90(ref)
            new_ref_point = np.where(newref == 1)
            new_bbx.append([int(new_ref_point[1]), int(new_ref_point[0]), int(
                bbox[i][3]), int(new_ref_point[0])-int(bbox[i][2])])
        if rot == 180:
            newref = np.rot90(np.rot90(ref))
            new_ref_point = np.where(newref == 1)
            new_bbx.append([int(new_ref_point[1]), int(
                new_ref_point[0]), -int(bbox[i][2]), -int(bbox[i][3])])
        if rot == 270:
            newref = np.rot90(ref, k=-1)
            new_ref_point = np.where(newref == 1)
            new_bbx.append([int(new_ref_point[1]), int(
                new_ref_point[0]), -int(bbox[i][3]), int(bbox[i][2])])
    return new_bbx


def rotate90bbox(row):
    return rotatebbox(row['bbox'], 90)


def rotate180bbox(row):
    return rotatebbox(row['bbox'], 180)


def rotate270bbox(row):
    return rotatebbox(row['bbox'], 270)


# print(len(annot_data))
# print("Init time: ", datetime.datetime.now())
annot_data['image'] = annot_data.apply(resize_im_rowwise, axis=1)
annot_data['bbox'] = annot_data.apply(resize_bbox_rowwise, axis=1)

print(annot_data.head())


def stack_to_numpy(row):
    #result_arr = np.empty((1, 4))
   # result_arr = []

    out = np.array(row['bbox']).astype("float64")

    return out


annot_data['np_bboxes'] = annot_data.apply(lambda row: stack_to_numpy(row), axis=1)

print(annot_data.head())

# plotted_img = draw_rect(annot_data['image'][1],annot_data['np_bboxes'][1])
# plt.imshow(plotted_img)
# plt.show()

# img_, bboxes_ = RandomHorizontalFlip(1)(annot_data['image'][1].copy(), annot_data['np_bboxes'][1].copy())
# plotted_img = draw_rect(img_, bboxes_)
# plt.imshow(plotted_img)
# plt.show()
# img_, bboxes_ = RandomScale(0.3, diff = False)(annot_data['image'][1].copy(), annot_data['np_bboxes'][1].copy())
# plotted_img = draw_rect(img_, bboxes_)
# plt.imshow(plotted_img)
# plt.show()

# img_, bboxes_ = RandomTranslate(0.3, diff = True)(annot_data['image'][1].copy(), annot_data['np_bboxes'][1].copy())
# plotted_img = draw_rect(img_, bboxes_)
# plt.imshow(plotted_img)
# plt.show()

# img_, bboxes_ = RandomRotate(20)(annot_data['image'][1].copy(), annot_data['np_bboxes'][1].copy())
# plotted_img = draw_rect(img_, bboxes_)
# plt.imshow(plotted_img)
# plt.show()

# img_, bboxes_ = RandomShear(0.2)(annot_data['image'][1].copy(), annot_data['np_bboxes'][1].copy())
# plotted_img = draw_rect(img_, bboxes_)
# plt.imshow(plotted_img)
# plt.show()

img_, bboxes_ = RandomHSV(100, 100, 100)(annot_data['image'][1].copy(), annot_data['np_bboxes'][1].copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()



# annot_data_90 = annot_data.copy()
# annot_data_180 = annot_data.copy()
# annot_data_270 = annot_data.copy()

# annot_data_90['image'] = annot_data.apply(rotate90img, axis=1)
# annot_data_90['bbox'] = annot_data.apply(rotate90bbox, axis=1)

# annot_data_180['image'] = annot_data.apply(rotate180img, axis=1)
# annot_data_180['bbox'] = annot_data.apply(rotate180bbox, axis=1)

# annot_data_270['image'] = annot_data.apply(rotate270img, axis=1)
# annot_data_270['bbox'] = annot_data.apply(rotate270bbox, axis=1)

# annot_data = annot_data.append(annot_data_90, ignore_index=True)
# annot_data = annot_data.append(annot_data_180, ignore_index=True)
# annot_data = annot_data.append(annot_data_270, ignore_index=True)
# print("End time: ", datetime.datetime.now())
# print(len(annot_data))

# i = 0

# fig, ax = plt.subplots()
# ax.imshow(annot_data['image'][i])
# rect = patches.Rectangle((annot_data['bbox'][i][0][0], annot_data['bbox'][i][0][1]), \
#     annot_data['bbox'][i][0][2]-annot_data['bbox'][i][0][0], annot_data['bbox'][i][0][3]-annot_data['bbox'][i][0][1],linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# ax.scatter(annot_data['bbox'][i][0][0], annot_data['bbox'][i][0][1],c='r')

# fig, ax = plt.subplots()
# ax.imshow(annot_data['image'][i+3])
# rect = patches.Rectangle((annot_data['bbox'][i+3][0][0], annot_data['bbox'][i+3][0][1]), \
#     annot_data['bbox'][i+3][0][2]-annot_data['bbox'][i+3][0][0],annot_data['bbox'][i+3][0][3]-annot_data['bbox'][i+3][0][1],linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# ax.scatter(annot_data['bbox'][i+3][0][0], annot_data['bbox'][i+3][0][1],c='r')

# fig, ax = plt.subplots()
# ax.imshow(annot_data['image'][i+6])
# rect = patches.Rectangle((annot_data['bbox'][i+6][0][0], annot_data['bbox'][i+6][0][1]), \
#     annot_data['bbox'][i+6][0][2]-annot_data['bbox'][i+6][0][0], annot_data['bbox'][i+6][0][3]-annot_data['bbox'][i+6][0][1],linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# ax.legend('rot180')
# ax.scatter(annot_data['bbox'][i+6][0][0], annot_data['bbox'][i+6][0][1],c='r')

# fig, ax = plt.subplots()
# ax.imshow(annot_data['image'][i+9])
# rect = patches.Rectangle((annot_data['bbox'][i+9][0][0], annot_data['bbox'][i+9][0][1]), \
#     annot_data['bbox'][i+9][0][2]-annot_data['bbox'][i+9][0][0], annot_data['bbox'][i+9][0][3]+annot_data['bbox'][i+9][0][1],linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# ax.legend('rot270')
# ax.scatter(annot_data['bbox'][i+9][0][0], annot_data['bbox'][i+9][0][1],c='r')
# plt.show()


# bboxs = bbox_utils.generate(s, 130//4, 10, (128, 128))

# np_bboxs = np.asarray(list(
#     map(lambda BBOX: [BBOX.arr[0], BBOX.arr[1], BBOX.arr[2], BBOX.arr[3]], bboxs)))


# def calculate_iou_rowwise(row):
#     target_vector = np.zeros(len(np_bboxs))
#     for bbox in row['bbox']:
#         xA = np.maximum(np_bboxs[:, 0], bbox[0])
#         yA = np.maximum(np_bboxs[:, 1], bbox[1])
#         xB = np.minimum(np_bboxs[:, 2], bbox[2])
#         yB = np.minimum(np_bboxs[:, 3], bbox[3])
#         interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
#         boxAArea = (np_bboxs[:, 2] - np_bboxs[:, 0] + 1) * \
#             (np_bboxs[:, 3] - np_bboxs[:, 1] + 1)
#         boxBArea = (bbox[2] - bbox[0] + 1) * \
#             (bbox[3] - bbox[1] + 1)
#         iou = np.divide(interArea, (np.subtract(
#             np.add(boxAArea, boxBArea), interArea)))
#         b = np.zeros_like(iou)
#         b[np.argmax(iou, axis=0)] = 1
#         target_vector = target_vector + b
#     print(sum(target_vector))
#     return target_vector


# annot_data['target_vector'] = annot_data.apply(calculate_iou_rowwise, axis=1)


# def display_bbox_target_vector(data_frame):
#     fig, ax = plt.subplots()
#     to_draw = np_bboxs[np.array(data_frame['target_vector'][1], dtype=bool)]

#     ax.imshow(data_frame['image'][1])
#     for bbox in to_draw:
#         rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0],
#                                  bbox[3]-bbox[1], linewidth=1, edgecolor=bbox_utils.get_random_color(), facecolor='none')
#         ax.add_patch(rect)
#     plt.show()


# display_bbox_target_vector(annot_data)

# annot_data = annot_data.reset_index()
# X = annot_data['image']
# Y = annot_data['target_vector']
# X_train, X_val, y_train, y_val = train_test_split(
#     X, Y, test_size=0.15, random_state=42)


# class AircraftDataset(Dataset):
#     def __init__(self, images, y, transforms=False):
#         self.images = images.values
#         self.y = y.values

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         path = self.images[idx]
#         y = self.y[idx]
#         return path, y


# train_ds = AircraftDataset(X_train, y_train, transforms=True)
# valid_ds = AircraftDataset(X_val, y_val)

# batch_size = 64
# train_dl = DataLoader(train_ds, batch_size=batch_size,
#                       shuffle=True, num_workers=0, drop_last=False)
# valid_dl = DataLoader(valid_ds, batch_size=batch_size,
#                       shuffle=False, num_workers=0, drop_last=False)


# class AircraftModel(nn.Module):
#     def __init__(self):
#         super(AircraftModel, self).__init__()
#         self.conv = nn.Sequential(
#             Conv2d(3, 192, kernel_size=7, stride=2),
#             nn.LeakyReLU(0.1),
#             MaxPool2d(2, 2),
#             Conv2d(192, 256, 3, 1),
#             nn.LeakyReLU(0.1),
#             MaxPool2d(2, 2),
#             # Conv2d(256,128,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(128,256,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(256,256,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(256,512,1,1),
#             # nn.LeakyReLU(0.1),
#             # MaxPool2d(2,2),
#             # Conv2d(512,256,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(256,512,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(512,256,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(256,512,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(512,256,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(256,512,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(512,256,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(256,512,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(512,1024,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(1024,512,1,1),
#             # nn.LeakyReLU(0.1),
#             # # MaxPool2d(2,2),
#             # Conv2d(512,1024,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(1024,512,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(512,1024,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(1024,512,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(512,1024,1,1),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(1024,1024,1,2),
#             # nn.LeakyReLU(0.1),
#             # Conv2d(1024,1024,1,1),
#             # nn.LeakyReLU(0.1),
#             Conv2d(256, 512, 1, 1),
#             nn.LeakyReLU(0.1),
#             nn.Flatten(start_dim=1),
#             nn.Dropout(0.5)
#         )

#         self.connected = nn.Sequential(
#             # (128*128,out_features=1024, bias=False),
#             nn.LazyLinear(out_features=256),
#             nn.ReLU(),
#             nn.Linear(256, out_features=len(bboxs), bias=False)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.connected(x)
#         return x


# model = AircraftModel()
# device = torch.device('cuda')  # use cuda or cpu
# print("Used device: ", device)
# model.to(device)
# print(model)

# out = model(torch.randn(batchsize, 3, 128, 128, device=device))
# # print("Output shape:", out.size())
# # print(f"Output logits:\n{out.detach().cpu().numpy()}")
# optimizer = optim.Adam(model.parameters(), lr)

# convert_tensor = transforms.ToTensor()
# step = 0
# model.train()

# train_accuracies = []
# valid_accuracies = []

# start_time = str(time.time())

# titles = ['learning rate', 'batchsize', 'epochs',
#           'train_images', 'val_images', 's', 'loss_fn', 'optimizer']
# hyper = [lr, batchsize, num_epochs, len(
#     train_ds), len(valid_ds), s, loss_fn, optimizer]
# PATH_HYPER = Path(dir_root, './data/model/logs/hyper_{start_time}.csv')
# with open(PATH_HYPER, 'a', newline='') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(titles)
#     wr.writerow(hyper)

# for epoch in range(num_epochs):
#     print("Epoch number: ", epoch)

#     train_accuracies_batches = []
#     for inputs, targets in train_dl:
#         new_inputs = []
#         for i in range(len(inputs)):
#             im = bbox_utils.transformsXY_im(inputs[i], new_size)
#             tensor = convert_tensor(im)
#             new_inputs.append(tensor)

#         new_inputs = torch.stack(tuple(new_inputs), 0)
#         new_inputs, targets = new_inputs.to(device), targets.to(device)

#         # Forward pass, compute gradients, perform one training step.
#         optimizer.zero_grad()
#         output = model(new_inputs)
#         loss = loss_fn(output, targets)
#         loss.backward()
#         optimizer.step()

#         # Increment step counter
#         step += 1

#         # Compute accuracy.
#         # we use output & target

#         subtra = torch.subtract(output, targets)
#         squared = torch.square(subtra)
#         acc = torch.sum(squared)

#         print("targets: ", targets)
#         print("output: ", output)

#         correct_match = 0
#         correct_match += (output == targets).float().sum()
#         accuracy_train = 100 * correct_match / len(inputs)
#         print("accuracy_val: ", float(accuracy_train.numpy()))
#         # acc.cpu().detach().numpy())
#         train_accuracies_batches.append(float(accuracy_train.cpu().numpy()))

#         PATH_TRAIN = Path(
#             dir_root, './data/model/logs/logs_train_{start_time}.csv')
#         with open(PATH_TRAIN, 'a', newline='') as myfile:
#             wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#             wr.writerow([accuracy_train.cpu().detach().numpy(),
#                         loss.cpu().detach().numpy()])  # [int(acc)])

#             if step % validation_every_steps == 0:

#                 with torch.no_grad():
#                     model.eval()
#                     for inputs, targets in valid_dl:
#                         new_inputs = []

#                         for i in range(len(inputs)):
#                             im = bbox_utils.transformsXY_im(
#                                 inputs[i], new_size)
#                             tensor = convert_tensor(im)
#                             new_inputs.append(tensor)

#                         new_inputs = torch.stack(tuple(new_inputs), 0)
#                         new_inputs, targets = new_inputs.to(
#                             device), targets.to(device)

#                         output = model(new_inputs)
#                         loss = loss_fn(output, targets)

#                         print("targets: ", targets)
#                         print("output: ", output)

#                         correct_match = 0
#                         correct_match += (output == targets).float().sum()
#                         accuracy_val = 100 * correct_match / len(inputs)
#                         print("accuracy_val: ", float(accuracy_val.numpy()))
#                         # acc.cpu().detach().numpy())
#                         train_accuracies_batches.append(
#                             float(accuracy_val.cpu().detach().numpy()))

#                         PATH_TRAIN = Path(
#                             dir_root, './data/model/logs/logs_val_{start_time}.csv')
#                         with open(PATH_TRAIN, 'a', newline='') as myfile:
#                             wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#                             # [int(acc)])
#                             wr.writerow(
#                                 [accuracy_val.cpu().detach().numpy(), loss.cpu().detach().numpy()])
#                     model.train()

# print("Finished training.")

# # #PATH = os.path.join(dir_root, f'../AIRCRAFT/data/model/{start_time}.pth')
# # #torch.save(model, PATH)
