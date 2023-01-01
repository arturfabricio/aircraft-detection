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
from pathlib import Path
import datetime
from data_aug import *
from bbox_aug import *
import visualization
import warnings

from typing import Union
import model
from training_utilities import loss_fn, calculate_target_vector

warnings.simplefilter(action='ignore', category=FutureWarning)
dir_root = Path(__file__).parent.parent
train_imgs = Path(dir_root, './data/train')
annot_dir = Path(dir_root, './data/annot/rareplanes_train.json')
train_im_list = [z for z in os.listdir(train_imgs) if z.endswith('.png')]

### Hyperparameters #############
# 10e-8 is probably too small
learning_rate = 10e-6
momentum = 0.9
batchsize = 2
num_epochs = 1500

start_from_image: int = 0
# Nr of images to load, set to False to load all
image_load_count: Union[int, bool] = 100
# Saves the model every number of epochs. Set to False to never save, or True for always
save_every_epochs: Union[int, bool] = False

path_start_from_weights: Union[Path, bool] = False #Path(dir_root, './data/model/2022-12-09 10_04_12/model/epoch_3.pth')


train_model = True
print_logs = True
weight_decay = 0.0005
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

annot_data = annot_data.groupby(['image_id']).agg(tuple).applymap(np.array).reset_index()

if image_load_count != False:
    idxs = annot_data.index.to_list()
    delete_before = idxs[:(start_from_image)]
    delete_after = idxs[(start_from_image + image_load_count):]

    annot_data.drop(delete_after, axis=0, inplace=True)
    annot_data.drop(delete_before, axis=0, inplace=True)
    annot_data.reset_index(inplace=True)

annot_data['path'] = annot_data.apply( lambda row: str(train_imgs) + "/"+row['name'][0], axis=1)
annot_data.drop(['name', 'image_id'], axis=1, inplace=True)

new_size = model.IMAGE_SIZE[0]
ratio = int(512/new_size)

annot_data['image'] = annot_data.apply(resize_im_rowwise, axis=1)
annot_data['bbox'] = annot_data.apply(resize_bbox_rowwise, axis=1)
annot_data['np_bboxes'] = annot_data.apply(
    lambda row: np.array(row['bbox']).astype("float64"), axis=1)


inital_amount_of_datapoints = len(annot_data['image'])


seq = Sequence([RandomHSV(25, 25, 25), RandomRotate(180), RandomScale(0.2), RandomTranslate(0.2), RandomShear(0.2)])

def get_augmented_image(idx, image_column_df, bbox_column_df):
    img, bbox = image_column_df[idx].copy(), bbox_column_df[idx].copy()
    if random.random() > 0.2:
        img, bbox = seq(img, bbox)
    img = img / 255
    target_vector = calculate_target_vector(bbox)
    return img, target_vector

#Prints dataset with bounding boxes
# image_id = 0
# for i in range(0, len(annot_data['image'])):
#     print(annot_data['bbox'][i])
#     visualization.display_bboxs(annot_data['image'][i],
#                                 annot_data['bbox'][i])

# #Prints the target vectors bounding boxes
# for i in range(0, len(annot_data['image'])):
#     visualization.display_bbox_target_vector(
#         annot_data['image'][i], annot_data['target_vector'][i], model.np_bboxs, 0.5)



X_train, X_val, y_train, y_val = train_test_split(annot_data['image'], annot_data['np_bboxes'], test_size=0.15, random_state=42)

class AircraftDataset(Dataset):
    def __init__(self, images, np_bboxes):
        self.images = images.values
        self.np_bboxes = np_bboxes.values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return get_augmented_image(idx, self.images, self.np_bboxes)


train_ds = AircraftDataset(X_train, y_train)
valid_ds = AircraftDataset(X_val, y_val)

train_dl = DataLoader(train_ds, batch_size=batchsize,
                      shuffle=True, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=batchsize,
                      shuffle=True, drop_last=True)

def get_local_time() -> str:
    return strftime("%Y-%m-%d %H:%M:%S", localtime())


start_time: str = get_local_time().replace(':', '_')

if train_model == False:
    exit(0)


model_directory = Path(dir_root,  f'./data/model/{start_time}/')
model_directory.mkdir(parents=True, exist_ok=True)

LOG_PATH = Path(
    model_directory, f'./logs.log')

# Create logging file
with open(LOG_PATH, 'w') as fp:
    pass

def print_to_logs(to_print: str):
    with open(LOG_PATH, 'a') as file:
        file.write(
            f'{get_local_time()} - ' + to_print + '\n')



aircraft_model = model.AircraftModel()
if (path_start_from_weights != False):
    aircraft_model.load_weights(path_start_from_weights)

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')  # use cuda or cpu
print("Used device: ", device)
aircraft_model.to(device)

# out = aircraft_model(torch.randn(batchsize, 3, model.IMAGE_SIZE[0], model.IMAGE_SIZE[1], device=device))
# print("Output shape:", out.size())
# print(f"Output logits:\n{out.detach().cpu().numpy()}")
optimizer = optim.SGD(aircraft_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

convert_tensor = transforms.ToTensor()

train_accuracies = []
valid_accuracies = []

import json
if print_logs == True:

    data_to_file = {
        'learning rate': learning_rate,
        'batchsize': batchsize,
        'epochs': num_epochs,
        'inital_datapoints': inital_amount_of_datapoints,
        'total_datapoints': len(annot_data),
        's': model.s,
        'weigth decay': weight_decay,
        'name': aircraft_model.name
    }
    PATH_HYPER = Path(
        model_directory, "hyper_parameters.csv")

    with open(PATH_HYPER, 'w') as f:
        json.dump(data_to_file, f, sort_keys=True, indent=4)

is_set = False
unique_x = None

for epoch in range(num_epochs):
    
    aircraft_model.train()
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

    #print("train_losses", train_losses)

    print_to_logs('Training Loss: ' + str(np.mean(np.array(train_losses))))
    print_to_logs('Validation Loss: ' + str(np.mean(np.array(val_losses))))

    if save_every_epochs == True or (save_every_epochs != False and (epoch % save_every_epochs == save_every_epochs-1 or epoch == 0) ):
        DIR_PATH = Path(model_directory, './model/')
        DIR_PATH.mkdir(parents=True, exist_ok=True)
        FILE_PATH = Path(DIR_PATH, f"./epoch_{str(epoch)}.pth")
        torch.save(aircraft_model.state_dict(), FILE_PATH)

print_to_logs("Finished training.")
