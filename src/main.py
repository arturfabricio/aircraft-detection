print('Yolo Implementation, v1 (22/12/2022)')

from time import localtime, strftime
import os
import json
import pandas as pd
import numpy as np
from torchvision import transforms
import csv
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import bbox_utils
from pathlib import Path
import datetime
from data_aug import *
from bbox_aug import *
import visualization
import warnings
from PIL import Image
from torch.optim import SGD
from model_yolo import YOLOv1
import time

from typing import Union
import model
from training_utilities import *

warnings.simplefilter(action='ignore', category=FutureWarning)
dir_root = Path(__file__).parent.parent
train_imgs = Path(dir_root, './data/train')
annot_dir = Path(dir_root, './data/annot/rareplanes_train.json')
train_im_list = [z for z in os.listdir(train_imgs) if z.endswith('.png')]
output_dir = Path(dir_root, '/outputs')

######### Hyperparameters #############
learning_rate = 10e-5
momentum = 0.9
weight_decay = 0.0005
batchsize = 64
num_epochs = 1
num_classes = 6
image_size = 512
save_freq = 40
S = 7
B = 2

train_proportion = 0.8
val_proportion = 0.2

start_from_image: int = 0
# Nr of images to load, set to False to load all
image_load_count: Union[int, bool] = 100
# Saves the model every number of epochs. Set to False to never save, or True for always
save_every_epochs: Union[int, bool] = False
path_start_from_weights: Union[Path, bool] = False #Path(dir_root, './data/model/2022-12-09 10_04_12/model/epoch_3.pth')
train_model = True
print_logs = True

#################################
print("Initiating...")
print("Hyperparameters: ")
print("Epochs: ", num_epochs)
print("Learning rate: ", learning_rate)
print("Batch Size: ", batchsize)
#################################

#### Functions ######
def resize_im_rowwise(row):
    return bbox_utils.transformsImg(row['path'], 512)

def resize_bbox_rowwise(row):
    return np.array(bbox_utils.transformsBbox(row['bbox'], 1), dtype='float64')

### Processing ###
print("Started running!")

with open(annot_dir) as json_data:
    data = json.load(json_data)

annot_data = pd.DataFrame(data['categories'])
annot_data.drop(['loc_id', 'cat_id', 'location', 'role', 'role_id', 'is_plane', 'num_engines', 'propulsion', 'canards', 'num_tail_fins', 'wing_position', 'wing_type',
                'length', 'wingspan', 'area', 'Public_Train', 'Public_Test', 'partialDec', 'truncated', 'new_area', 'area_pixels', 'id'], axis=1, inplace=True)
annot_data.rename(columns={"image_fname": "name"}, inplace=True)
annot_data['faa_wingspan_class'] = annot_data.apply( lambda row: int(row['faa_wingspan_class']-1), axis=1)
annot_data = annot_data.groupby(['image_id']).agg(tuple).applymap(np.array).reset_index()

print("Loaded data....")

if image_load_count != False:
    idxs = annot_data.index.to_list()
    delete_before = idxs[:(start_from_image)]
    delete_after = idxs[(start_from_image + image_load_count):]

    annot_data.drop(delete_after, axis=0, inplace=True)
    annot_data.drop(delete_before, axis=0, inplace=True)
    annot_data.reset_index(inplace=True)

annot_data['path'] = annot_data.apply( lambda row: str(train_imgs) + "/"+row['name'][0], axis=1)
annot_data.drop(['name', 'image_id'], axis=1, inplace=True)

annot_data['image'] = annot_data.apply(resize_im_rowwise, axis=1)
annot_data['bbox'] = annot_data.apply(resize_bbox_rowwise, axis=1)
annot_data['np_bboxes'] = annot_data.apply(
    lambda row: np.array(row['bbox']).astype("float64"), axis=1)

# print(annot_data.head())

inital_amount_of_datapoints = len(annot_data['image'])
print("Amount init data: ", inital_amount_of_datapoints)

## Data Augmentation
print("Starting Augmentation!")
for i in range(len(annot_data['image'])):
    print("image number: ", i)
    img, bboxes = RandomHorizontalFlip(1)(annot_data['image'][i].copy(), annot_data['np_bboxes'][i].copy())
    df_append = pd.DataFrame({"index":[annot_data['index'][i]],"faa_wingspan_class":[annot_data['faa_wingspan_class'][i]],"image":[img],"np_bboxes":[bboxes]})
    annot_data = annot_data.append(df_append, ignore_index = True)

    img, bboxes = RandomRotate(180)(annot_data['image'][i].copy(), annot_data['np_bboxes'][i].copy())
    df_append = pd.DataFrame({"index":[annot_data['index'][i]],"faa_wingspan_class":[annot_data['faa_wingspan_class'][i]],"image":[img],"np_bboxes":[bboxes]})
    annot_data = annot_data.append(df_append, ignore_index = True)

    img, bboxes = RandomRotate(270)(annot_data['image'][i].copy(), annot_data['np_bboxes'][i].copy())
    df_append = pd.DataFrame({"index":[annot_data['index'][i]],"faa_wingspan_class":[annot_data['faa_wingspan_class'][i]],"image":[img],"np_bboxes":[bboxes]})
    annot_data = annot_data.append(df_append, ignore_index = True)

    img, bboxes = RandomShear(0.2)(annot_data['image'][i].copy(), annot_data['np_bboxes'][i].copy())
    df_append = pd.DataFrame({"index":[annot_data['index'][i]],"faa_wingspan_class":[annot_data['faa_wingspan_class'][i]],"image":[img],"np_bboxes":[bboxes]})
    annot_data = annot_data.append(df_append, ignore_index = True)

    img, bboxes = RandomHSV(100, 100, 100)(annot_data['image'][i].copy(), annot_data['np_bboxes'][i].copy())
    df_append = pd.DataFrame({"index":[annot_data['index'][i]],"faa_wingspan_class":[annot_data['faa_wingspan_class'][i]],"image":[img],"np_bboxes":[bboxes]})
    annot_data = annot_data.append(df_append, ignore_index = True)

    img, bboxes = RandomRotate(90)(annot_data['image'][i].copy(), annot_data['np_bboxes'][i].copy())
    df_append = pd.DataFrame({"index":[annot_data['index'][i]],"faa_wingspan_class":[annot_data['faa_wingspan_class'][i]],"image":[img],"np_bboxes":[bboxes]})
    annot_data = annot_data.append(df_append, ignore_index = True)
print("Augmentation done!")
# print(annot_data.head())
print("New data size: ", len(annot_data['index']))

# plotted_img = draw_rect(annot_data['image'][0], annot_data['np_bboxes'][0])
# plt.imshow(plotted_img)
# plt.show()

def tlbr_to_xywh(bbox,classe,size):
    return_bbox = []
    for i in range(len(bbox)):
        x = (bbox[i][0] + ((bbox[i][2]-bbox[i][0])/2))/size
        y = (bbox[i][1] + ((bbox[i][3]-bbox[i][1])/2))/size
        w = ((bbox[i][2]-bbox[i][0]))/size
        h = ((bbox[i][3]-bbox[i][1]))/size
        return_bbox.append([x,y,w,h,classe[i]])
    return return_bbox

annot_data['yolo_bboxes'] = annot_data.apply(lambda row: tlbr_to_xywh(row['np_bboxes'],row['faa_wingspan_class'],image_size),axis=1)
# print(annot_data.head())
# print(annot_data['yolo_bboxes'][0])

#### Test to see if yolo_bboxes are ok
# row_val = 2
# xplot = annot_data['yolo_bboxes'][row_val][0][0]*image_size
# yplot = annot_data['yolo_bboxes'][row_val][0][1]*image_size
# wplot = annot_data['yolo_bboxes'][row_val][0][2]*image_size
# hplot = annot_data['yolo_bboxes'][row_val][0][3]*image_size
# print("plot values: ", xplot,yplot,wplot,hplot)

# xv_values = [xplot,xplot]
# yv_values = [yplot + (hplot/2), yplot - (hplot/2)]

# xh_values = [xplot + (wplot/2), xplot - (wplot/2)]
# yh_values = [yplot, yplot]

# plt.figure()
# plt.title("Checking if the labels are correct.")
# plt.xlim([0,512])
# plt.ylim([0,512])
# plt.imshow(annot_data['image'][row_val])
# plt.plot(xv_values, yv_values, 'ro', linestyle="--")
# plt.plot(xh_values, yh_values, 'bo', linestyle="--")
# plt.scatter(xplot,yplot,c='green',s=20,label='New generated center')
# plt.gca().invert_yaxis()
# plt.show()

# compression_opts = dict(method='zip',archive_name='out.csv')
# annot_data.to_csv('out.zip', index=False,compression=compression_opts) 

# plotted_img = draw_rect(annot_data['image'][0], annot_data['np_bboxes'][0])
# plt.imshow(plotted_img)
# plt.show()

# plotted_img = draw_rect(annot_data['image'][1], annot_data['np_bboxes'][1])
# plt.imshow(plotted_img)
# plt.show()

# plotted_img = draw_rect(annot_data['image'][2], annot_data['np_bboxes'][2])
# plt.imshow(plotted_img)
# plt.show()

# plotted_img = draw_rect(annot_data['image'][3], annot_data['np_bboxes'][3])
# plt.imshow(plotted_img)
# plt.show()

# plotted_img = draw_rect(annot_data['image'][4], annot_data['np_bboxes'][4])
# plt.imshow(plotted_img)
# plt.show()

# plotted_img = draw_rect(annot_data['image'][5], annot_data['np_bboxes'][5])
# plt.imshow(plotted_img)
# plt.show()

# plotted_img = draw_rect(annot_data['image'][6], annot_data['np_bboxes'][6])
# plt.imshow(plotted_img)
# plt.show()

#     annot_data['image'].append(img)
#     annot_data['np_bboxes'].append(bboxes)

    # img_.append(img)
    # bboxes_.append(img)

# plotted_img = draw_rect(img_, bboxes_)
# plt.imshow(plotted_img)
# plt.show()

# seq = Sequence([RandomHSV(25, 25, 25), RandomRotate(180), RandomScale(0.2), RandomTranslate(0.2), RandomShear(0.2)])

# def get_augmented_image(idx, image_column_df, bbox_column_df):
#     img, bbox = image_column_df[idx].copy(), bbox_column_df[idx].copy()
#     # if random.random() > 0.2:
#     img, bbox = seq(img, bbox)
#     img = img / 255
#     target_vector = calculate_target_vector(bbox)
#     print(np.size(img))
#     return img, target_vector

#Prints dataset with bounding boxes
# for i in range(0, len(annot_data['image'])):
#     print(annot_data['bbox'][i])
#     visualization.display_bboxs(annot_data['image'][i],
#                                 annot_data['bbox'][i])

# X_train, X_val, y_train, y_val = train_test_split(annot_data['image'], annot_data['yolo_bboxes'], test_size=0.15, random_state=42)

class YOLODataset(Dataset):
    def __init__(self, img, bbox, S, B, num_classes, transforms=None):
        self.img = img 
        self.bbox = bbox 
        self.transforms = transforms
        self.S = S
        self.B = B
        self.num_classes = num_classes

    def __len__(self):
        return len(self.img)

    def __getitem__(self,idx):
        
        img = Image.fromarray(self.img[idx])
        img = self.transforms(img)  #to tensor

        xywhc = []
   
        for box in self.bbox:
            x, y, w, h, c = float(box[0][0]), float(box[0][1]), float(box[0][2]), float(box[0][3]), int(box[0][4])
            xywhc.append((x, y, w, h, c))

        label = xywhc2label(xywhc, self.S, self.B, self.num_classes)  # convert xywhc list to label
        label = torch.Tensor(label)
        return img, label

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

dataset = YOLODataset(annot_data['image'],annot_data['yolo_bboxes'],S,B,num_classes,transforms=transform)

dataset_size = len(dataset)
print("dataset_size: ", dataset_size)

train_size = int(dataset_size * train_proportion)
val_size = int(dataset_size * val_proportion)

# split dataset to train set, val set and test set three parts
train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, 0])
print("train_dataset: ", len(train_dataset))

train_dl = DataLoader(train_dataset, batch_size=batchsize, shuffle=False)
print("train_loader: ", len(train_dl))

valid_dl = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
print("val_loader: ", len(valid_dl))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

aircraft_model = YOLOv1(S,B,num_classes).to(device)

optimizer = SGD(aircraft_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
train_loss_lst, val_loss_lst = [], []

# # aircraft_model = model.AircraftModel()
# # if (path_start_from_weights != False):
# #     aircraft_model.load_weights(path_start_from_weights)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    
    aircraft_model.train()
    train_loss = 0

    for inputs, labels in train_dl:
        t_start = time.time()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = aircraft_model(inputs)

        criterion = YOLOv1Loss(S, B)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        t_batch = time.time() - t_start

        train_losses.append(loss.item())

        print('Train Epoch: {} [{}/{} ({:.1f}%)]  Time: {:.4f}s  Loss: {:.6f}'
                .format(epoch, 64 * len(inputs), len(train_dl.dataset),
                        100. * 64 / len(train_dl), t_batch, loss.item()))
    
    train_loss /= len(train_dl)
    train_loss_lst.append(train_loss)
    
    aircraft_model.eval()
    with torch.no_grad():
        val_loss = 0

        for data, target in valid_dl:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add one batch loss
            criterion = YOLOv1Loss(S, B)
            val_loss += criterion(output, target).item()

    val_loss /= len(valid_dl)
    print('Val set: Average loss: {:.4f}\n'.format(val_loss))
    val_loss_lst.append(val_loss)

    if epoch % save_freq == 0 and epoch >= num_epochs / 2:
        torch.save(model.state_dict(), os.path.join(output_dir, f'epoch_{num_epochs}_' + str(epoch) + '.pth'))
    # lr_scheduler.step()

    #print("train_losses", train_losses)

    # print_to_logs('Training Loss: ' + str(np.mean(np.array(train_losses))))
    # print_to_logs('Validation Loss: ' + str(np.mean(np.array(val_losses))))

    # if save_every_epochs == True or (save_every_epochs != False and (epoch % save_every_epochs == save_every_epochs-1 or epoch == 0) ):
    #     DIR_PATH = Path(model_directory, './model/')
    #     DIR_PATH.mkdir(parents=True, exist_ok=True)
    #     FILE_PATH = Path(DIR_PATH, f"./epoch_{str(epoch)}.pth")
    #     torch.save(aircraft_model.state_dict(), FILE_PATH)

torch.save(aircraft_model.state_dict(), os.path.join(output_dir, f'model_{num_epochs}.pth'))

# plot loss, save params change
fig = plt.figure()
plt.plot(range(num_epochs), train_loss_lst, 'g', label='train loss')
plt.plot(range(num_epochs), val_loss_lst, 'k', label='val loss')
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('acc-loss')
plt.legend(loc="upper right")
plt.savefig(os.path.join(output_dir, f'loss_curve_epochs:{num_epochs}.jpg'))
# plt.show()
plt.close(fig)

print("Finished training... Yay!")
