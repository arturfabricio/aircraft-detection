from time import localtime, strftime
import os
import json
import pandas as pd
import numpy as np
from torchvision import transforms
import csv
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import bbox_utils
from func_utils import *
from pathlib import Path

from typing import Union
import model
from training_utilities import loss_fn, calculate_iou
from augmentation import add_augment_to_df

# warnings.simplefilter(action='ignore', category=FutureWarning)
dir_root = Path(__file__).parent.parent
train_imgs = Path(dir_root, './data/train')
annot_dir = Path(dir_root, './data/annot/rareplanes.json')
train_im_list = [z for z in os.listdir(train_imgs) if z.endswith('.png')]


### Hyperparameters #############
# 10e-8 is probably too small
learning_rate = 1e-1
momentum = 0.9
batchsize = 64
num_epochs = 2000

start_from_image: int = 0
# Nr of images to load, set to False to load all
image_load_count: Union[int, bool] = False
# Saves the model every number of epochs. Set to False to never save, or True for always
save_every_epochs: Union[int, bool] = 20


train_model = True
print_logs = True
weight_decay = 1e-4
augment = True
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
    add_augment_to_df(annot_data)
# Prints dataset with bounding boxes
# image_id = 0
# for i in range(0, len(annot_data['image'])):
#     print(annot_data['bbox'][i])
#     visualization.display_bboxs(annot_data['image'][i],
#                                 annot_data['bbox'][i])

annot_data['target_vector'] = annot_data.apply(lambda row: calculate_iou(row['bbox']), axis=1)

# Prints the target vectors bounding boxes
# for i in range(0, len(annot_data['image'])):
#     visualization.display_bbox_target_vector(
#         annot_data['image'][i], annot_data['target_vector'][i], model.np_bboxs, 0.5)

print(annot_data.head())

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
                      shuffle=False, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=batchsize,
                      shuffle=False, drop_last=True)


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
optimizer = optim.SGD(aircraft_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

convert_tensor = transforms.ToTensor()
aircraft_model.train()

train_accuracies = []
valid_accuracies = []

if print_logs == True:
    titles = ['learning rate', 'batchsize', 'epochs',
              'train_images', 'val_images', 's', 'weigth decay', 'name', 'optimizer']
    hyper = [learning_rate, batchsize, num_epochs, len(
        train_ds), len(valid_ds), model.s, weight_decay, aircraft_model.name, optimizer]
    PATH_HYPER = Path(
        model_directory, "hyper_parameters.csv")
    with open(PATH_HYPER, 'a', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(titles)
        wr.writerow(hyper)


import torch.nn as nn

is_set = False
unique_x = None

for epoch in range(num_epochs):
    print_to_logs("Epoch number: " + str(epoch))

    train_losses = []
    val_losses = []

    for inputs, targets in train_dl:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = torch.permute(inputs, (0, 3, 1, 2))

        # zero the parameter gradients
        optimizer.zero_grad()

        # if not is_set:
        #     unique_x = inputs.clone()
        #     is_set = True
        # lets_print = torch.equal(unique_x, inputs)

        # if lets_print:
        #     print('input', inputs)
        #     print('targets', targets)

        # forward + backward + optimize
        output = aircraft_model(inputs)
        # if lets_print:
        #     print('output', output)
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

    #print("train_losses", train_losses)

    print_to_logs('Training Loss: ' + str(np.mean(np.array(train_losses))))
    print_to_logs('Validation Loss: ' + str(np.mean(np.array(val_losses))))

    if save_every_epochs == True or (save_every_epochs != False and (epoch % save_every_epochs == save_every_epochs-1 or epoch == 0) ):
        DIR_PATH = Path(model_directory, './model/')
        DIR_PATH.mkdir(parents=True, exist_ok=True)
        FILE_PATH = Path(DIR_PATH, f"./epoch_{str(epoch)}.pth")
        torch.save(aircraft_model.state_dict(), FILE_PATH)

print_to_logs("Finished training.")
