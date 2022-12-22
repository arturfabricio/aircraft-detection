print('Yolo Implementation')

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
learning_rate = 10e-5
momentum = 0.9
batchsize = 64
num_epochs = 1

start_from_image: int = 0
# Nr of images to load, set to False to load all
image_load_count: Union[int, bool] = 2
# Saves the model every number of epochs. Set to False to never save, or True for always
save_every_epochs: Union[int, bool] = False
path_start_from_weights: Union[Path, bool] = False #Path(dir_root, './data/model/2022-12-09 10_04_12/model/epoch_3.pth')

train_model = True
print_logs = True
weight_decay = 0.0005
augment = True
#################################``

# ### Functions ###
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
                'length', 'wingspan', 'area', 'Public_Train', 'Public_Test', 'partialDec', 'truncated', 'new_area', 'area_pixels', 'id'], axis=1, inplace=True)
annot_data.rename(columns={"image_fname": "name"}, inplace=True)
annot_data['faa_wingspan_class'] = annot_data.apply( lambda row: int(row['faa_wingspan_class']-1), axis=1)



# df_cols = ['filename','x','y','w','h','class']
# yolov1 = pd.DataFrame(columns=df_cols)

# yolov1['filename'] = annot_data.apply( lambda row: row['name'], axis=1)
# yolov1['x'] = annot_data.apply( lambda row: int(row['bbox'][0]+(row['bbox'][2]/2)), axis=1)
# yolov1['y'] = annot_data.apply( lambda row: int(row['bbox'][1]+(row['bbox'][3]/2)), axis=1)
# yolov1['w'] = annot_data.apply( lambda row: int(row['bbox'][2]), axis=1)
# yolov1['h'] = annot_data.apply( lambda row: int(row['bbox'][3]), axis=1)
# yolov1['class'] = annot_data.apply( lambda row: int(row['faa_wingspan_class'])-1, axis=1)

# # print(yolov1.head())

# def normalize(data):
#     return (data - 0) / (512 - 0)

# yolov1['x'] = yolov1.apply( lambda row: normalize(row['x']), axis=1)
# yolov1['y'] = yolov1.apply( lambda row: normalize(row['y']), axis=1)
# yolov1['w'] = yolov1.apply( lambda row: normalize(row['w']), axis=1)
# yolov1['h'] = yolov1.apply( lambda row: normalize(row['h']), axis=1)

# yolov1['filename'] = yolov1['filename'].str.replace(r'.png$', '')

# yolov1['txt'] = yolov1.apply( lambda row: str([row['x'], row['y'], row['w'], row['h'], row['class']]) , axis=1)

# # print(yolov1.head())
# yolov1['txt'] = yolov1['txt'].str.replace(r']', '')
# yolov1['txt'] = yolov1['txt'].str.replace(r'[', '')
# yolov1['txt'] = yolov1['txt'].str.replace(r',', '')




# print(yolov1['txt'][0])

# print(yolov1['class'].unique())

# i = 0
# for index, row in yolov1.iterrows():
#     if i > len(yolov1):
#         break
#     else:
#         f = open(yolov1['filename'][i] + '.txt', 'a')
#         f.write(yolov1['txt'][i] + '\n')
#         f.close()
#         i+=1

# img_dir = os.path.join(train_imgs,yolov1['filename'][0])
# print("Img dir: ", img_dir)
# new_image = cv2.imread(img_dir,cv2.IMREAD_COLOR)

# row_val = 2


# xplot = yolov1['x'][row_val]*512
# yplot = yolov1['y'][row_val]*512
# wplot = yolov1['w'][row_val]*512
# hplot = yolov1['h'][row_val]*512

# print("plot values: ", xplot,yplot,wplot,hplot)
# print("Real values: ", annot_data['bbox'][row_val])

# xcal = int((annot_data['bbox'][row_val][0]+annot_data['bbox'][row_val][2]/2))
# print("xcal: ", xcal)
# ycal = int((annot_data['bbox'][row_val][1]+annot_data['bbox'][row_val][3]/2))
# print("ycal: ", ycal)

# xv_values = [xcal,xcal ]
# yv_values = [ycal + (annot_data['bbox'][row_val][3]/2), ycal - (annot_data['bbox'][row_val][3]/2)]

# xh_values = [xcal + (annot_data['bbox'][row_val][2]/2), xcal - (annot_data['bbox'][row_val][2]/2)]
# yh_values = [ycal, ycal]


# plt.figure()
# plt.title("Checking if the labels are correct.")
# plt.xlim([0,512])
# plt.ylim([0,512])
# plt.imshow(new_image)
# plt.scatter(xplot,yplot,c='r',label='New generated center')
# plt.scatter(xcal,ycal,c='g',label='original')
# plt.plot(xv_values, yv_values, 'ro', linestyle="--")
# plt.plot(xh_values, yh_values, 'bo', linestyle="--")
# plt.gca().invert_yaxis()

# plt.show()

# yolov1.to_csv('train_labels.csv')

# annot_data = annot_data.groupby(['image_id']).agg(tuple).applymap(np.array).reset_index()

# if image_load_count != False:
#     idxs = annot_data.index.to_list()
#     delete_before = idxs[:(start_from_image)]
#     delete_after = idxs[(start_from_image + image_load_count):]

#     annot_data.drop(delete_after, axis=0, inplace=True)
#     annot_data.drop(delete_before, axis=0, inplace=True)
#     annot_data.reset_index(inplace=True)

# annot_data['path'] = annot_data.apply( lambda row: str(train_imgs) + "/"+row['name'][0], axis=1)
# annot_data.drop(['name', 'image_id'], axis=1, inplace=True)



# new_size = model.IMAGE_SIZE[0]
# ratio = int(512/new_size)

# annot_data['image'] = annot_data.apply(resize_im_rowwise, axis=1)
# annot_data['bbox'] = annot_data.apply(resize_bbox_rowwise, axis=1)
# annot_data['np_bboxes'] = annot_data.apply(
#     lambda row: np.array(row['bbox']).astype("float64"), axis=1)

# # print(annot_data['image'][0])
# # print(annot_data['np_bboxes'][0])

# from PIL import Image
# import os

# image_path = ".\data\augmented"


# annot_data['path'] = annot_data['path'].str.replace(r'.png$', '')
# annot_data['path'] = annot_data['path'].str.replace(r"\", "//")


# for i in range(len(annot_data['image'])):
#     img_, bboxes_ = RandomHorizontalFlip(1)(annot_data['image'][i].copy(), annot_data['np_bboxes'][i].copy())
#     name = f"{str(annot_data['path'][i])}_RHF.jpg"
#     print(name)
#     cv2.imwrite(name,img_)    
#     # image = img_.save(f"{image_path}/{annot_data['image'][i]}_RHF.png")
#     # plotted_img = draw_rect(img_, bboxes_)
#     # plt.imshow(plotted_img)
#     # plt.show()

#     img_, bboxes_ = RandomRotate(180)(annot_data['image'][i].copy(), annot_data['np_bboxes'][i].copy())
#     cv2.imwrite(f"{annot_data['image'][i]}_RR1.jpg",img_)

#     # plotted_img = draw_rect(img_, bboxes_)
#     # plt.imshow(plotted_img)
#     # plt.show()

#     img_, bboxes_ = RandomRotate(270)(annot_data['image'][i].copy(), annot_data['np_bboxes'][i].copy())
#     cv2.imwrite(f"{annot_data['image'][i]}_RR2.jpg",img_)
#     # plotted_img = draw_rect(img_, bboxes_)
#     # plt.imshow(plotted_img)
#     # plt.show()

#     img_, bboxes_ = RandomShear(0.2)(annot_data['image'][i].copy(), annot_data['np_bboxes'][i].copy())
#     cv2.imwrite(f"{annot_data['image'][i]}_RS.jpg",img_)
#     # plotted_img = draw_rect(img_, bboxes_)
#     # plt.imshow(plotted_img)
#     # plt.show()

#     img_, bboxes_ = RandomHSV(100, 100, 100)(annot_data['image'][i].copy(), annot_data['np_bboxes'][i].copy())
#     cv2.imwrite(f"{annot_data['image'][i]}_HSV.jpg",img_)
#     # plotted_img = draw_rect(img_, bboxes_)
#     # # plt.imshow(plotted_img)
#     # # plt.show()
#     print("loop end: ", i)

# inital_amount_of_datapoints = len(annot_data['image'])

# seq = Sequence([RandomHSV(25, 25, 25), RandomRotate(180), RandomScale(0.2), RandomTranslate(0.2), RandomShear(0.2)])

# def get_augmented_image(idx, image_column_df, bbox_column_df):
#     img, bbox = image_column_df[idx].copy(), bbox_column_df[idx].copy()
#     if random.random() > 0.2:
#         img, bbox = seq(img, bbox)
#     img = img / 255
#     target_vector = calculate_target_vector(bbox)
#     return img, target_vector

# # #Prints dataset with bounding boxes
# # image_id = 0
# # for i in range(0, len(annot_data['image'])):
# #     print(annot_data['bbox'][i])
# #     visualization.display_bboxs(annot_data['image'][i],
# #                                 annot_data['bbox'][i])

# # #Prints the target vectors bounding boxes
# # for i in range(0, len(annot_data['image'])):
# #     visualization.display_bbox_target_vector(
# #         annot_data['image'][i], annot_data['target_vector'][i], model.np_bboxs, 0.5)



# X_train, X_val, y_train, y_val = train_test_split(annot_data['image'], annot_data['np_bboxes'], test_size=0.15, random_state=42)


# class AircraftDataset(Dataset):
#     def __init__(self, images, np_bboxes):
#         self.images = images.values
#         self.np_bboxes = np_bboxes.values

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         return get_augmented_image(idx, self.images, self.np_bboxes)


# train_ds = AircraftDataset(X_train, y_train)
# valid_ds = AircraftDataset(X_val, y_val)

# print(len(train_ds))

# train_dl = DataLoader(train_ds, batch_size=batchsize,
#                       shuffle=True, drop_last=True)
# valid_dl = DataLoader(valid_ds, batch_size=batchsize,
#                       shuffle=True, drop_last=True)


# # def get_local_time() -> str:
# #     return strftime("%Y-%m-%d %H:%M:%S", localtime())


# # start_time: str = get_local_time().replace(':', '_')

# # if train_model == False:
# #     exit(0)


# # model_directory = Path(dir_root,  f'./data/model/{start_time}/')
# # model_directory.mkdir(parents=True, exist_ok=True)

# # LOG_PATH = Path(
# #     model_directory, f'./logs.log')

# # # Create logging file
# # with open(LOG_PATH, 'w') as fp:
# #     pass

# # def print_to_logs(to_print: str):
# #     with open(LOG_PATH, 'a') as file:
# #         file.write(
# #             f'{get_local_time()} - ' + to_print + '\n')



# # aircraft_model = model.AircraftModel()
# # if (path_start_from_weights != False):
# #     aircraft_model.load_weights(path_start_from_weights)

# # device = torch.device('cuda' if torch.cuda.is_available()
# #                       else 'cpu')  # use cuda or cpu
# # print("Used device: ", device)
# # aircraft_model.to(device)

# # # out = aircraft_model(torch.randn(batchsize, 3, model.IMAGE_SIZE[0], model.IMAGE_SIZE[1], device=device))
# # # print("Output shape:", out.size())
# # # print(f"Output logits:\n{out.detach().cpu().numpy()}")
# # optimizer = optim.SGD(aircraft_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# # convert_tensor = transforms.ToTensor()

# # train_accuracies = []
# # valid_accuracies = []

# # import json
# # if print_logs == True:

# #     data_to_file = {
# #         'learning rate': learning_rate,
# #         'batchsize': batchsize,
# #         'epochs': num_epochs,
# #         'inital_datapoints': inital_amount_of_datapoints,
# #         'total_datapoints': len(annot_data),
# #         's': model.s,
# #         'weigth decay': weight_decay,
# #         'name': aircraft_model.name
# #     }
# #     PATH_HYPER = Path(
# #         model_directory, "hyper_parameters.csv")

# #     with open(PATH_HYPER, 'w') as f:
# #         json.dump(data_to_file, f, sort_keys=True, indent=4)

# # is_set = False
# # unique_x = None

# # for epoch in range(num_epochs):
    
# #     aircraft_model.train()
# #     print_to_logs("Epoch number: " + str(epoch))

# #     train_losses = []
# #     val_losses = []

# #     for inputs, targets in train_dl:
# #         inputs, targets = inputs.to(device), targets.to(device)
# #         inputs = torch.permute(inputs, (0, 3, 1, 2))

# #         # zero the parameter gradients
# #         optimizer.zero_grad()

# #         # forward + backward + optimize
# #         output = aircraft_model(inputs)
# #         loss = loss_fn(output, targets)
# #         loss.backward()
# #         optimizer.step()

# #         train_losses.append(loss.item())

# #     with torch.no_grad():
# #         aircraft_model.eval()
# #         for inputs, targets in valid_dl:
# #             inputs, targets = inputs.to(device), targets.to(device)
# #             inputs = torch.permute(inputs, (0, 3, 1, 2))

# #             # forward + backward + optimize
# #             output = aircraft_model(inputs)
# #             loss = loss_fn(output, targets)

# #             val_losses.append(loss.item())

# #     #print("train_losses", train_losses)

# #     print_to_logs('Training Loss: ' + str(np.mean(np.array(train_losses))))
# #     print_to_logs('Validation Loss: ' + str(np.mean(np.array(val_losses))))

# #     if save_every_epochs == True or (save_every_epochs != False and (epoch % save_every_epochs == save_every_epochs-1 or epoch == 0) ):
# #         DIR_PATH = Path(model_directory, './model/')
# #         DIR_PATH.mkdir(parents=True, exist_ok=True)
# #         FILE_PATH = Path(DIR_PATH, f"./epoch_{str(epoch)}.pth")
# #         torch.save(aircraft_model.state_dict(), FILE_PATH)

# # print_to_logs("Finished training.")
