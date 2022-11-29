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

warnings.simplefilter(action='ignore', category=FutureWarning)
data_root = os.getcwd()
train_imgs = os.path.join(data_root, '../AIRCRAFT/data/train')
annot_dir = os.path.join(data_root, '../AIRCRAFT/data/annot/rareplanes.json')
train_im_list = [z for z in os.listdir(train_imgs) if z.endswith('.png')]
f = open(annot_dir)
data = json.load(f)
assert len(train_im_list) == len(data['images'])

### Hyperparameters #############
s=15
lr = 0.0001
loss_fn = nn.CrossEntropyLoss()  
batchsize = 64
num_epochs = 500
validation_every_steps = 20
#################################

### Functions ###

#change

def imgs_annot_aggregator(iter):
    print("Running imgs_annot_aggregator...")
    final = np.zeros((iter,4))
    bounding_boxes = []
    image_names = []
    instances_img = []
    amount_matches = []
    class_type = []
    for j in range(iter): 
        instances_img = []                        
        img = random.sample(train_im_list,1)
        image_names.append(img[0])
        for i in range(len(data['categories'])):
            if [data['categories'][i]['image_fname']] == img:
                instances_img.append(data['categories'][i]['id'])
        for l in range(len(instances_img)):
            for i in range(len(data['categories'])):
                if data['categories'][i]['id'] == instances_img[l]:
                    bounding_boxes.append(data['categories'][i]['bbox'])
                    class_type.append(data['categories'][i]['role'])
        amount_matches.append(len(instances_img))
    final = amount_matches, image_names, bounding_boxes, class_type
    return final, image_names

def bbox_points(data_annot):
    print("Running bbox_points...")
    name = []
    x_org = []
    y_org = []
    x_dist = []
    y_dist = []
    bbox = []
    class_type = []
    file_path = []
    itr = 0
    for i in range(len(data_annot[0])):
        for j in range(data_annot[0][i]):
            name.append(data_annot[1][i])
            class_type.append(data_annot[3][i])
            x_org.append(data_annot[2][j+itr][0])
            y_org.append(data_annot[2][j+itr][1])
            x_dist.append(data_annot[2][j+itr][2])
            y_dist.append(data_annot[2][j+itr][3])
            bbox.append([data_annot[2][j+itr][0],data_annot[2][j+itr][1],data_annot[2][j+itr][0]+data_annot[2][j+itr][2],data_annot[2][j+itr][1]+data_annot[2][j+itr][3]])
            file_path.append(os.path.join(train_imgs, data_annot[1][i]))
        itr = itr + data_annot[0][i]
    df = pd.DataFrame(
    {'name': name,
     'class': class_type,
     'x_org': x_org,
     'y_org': y_org,
     'x_dist': x_dist,
     'y_dist': y_dist,
     'bbox': bbox,
     'file_path': file_path
    })
    return df

def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    Y = np.zeros([rows,cols])
    bb = bb.astype(int)
    Y[bb[1]:(bb[1]+bb[3]),bb[0]:(bb[0]+bb[2])] = 1.
    return Y

def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
    return np.array([x[2],x[3],x[4],x[5]])

def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
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
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
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
                final_bbxs.append(bbox_utils.BBOX(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]))
            
            final_data_frame = final_data_frame.append({'path': img_path_test, 'final_bbx': final_bbxs}, ignore_index=True)

    return final_data_frame

def iou(boxA, boxB):
    xA = max(boxA.arr[0], boxB.arr[0])
    yA = max(boxA.arr[1], boxB.arr[1])
    xB = min(boxA.arr[2], boxB.arr[2])
    yB = min(boxA.arr[3], boxB.arr[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA.arr[2] - boxA.arr[0] + 1) * (boxA.arr[3] - boxA.arr[1] + 1)
    boxBArea = (boxB.arr[2] - boxB.arr[0] + 1) * (boxB.arr[3] - boxB.arr[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def app_flat(my_array):
    out = np.zeros_like(my_array)
    idx  = my_array.argmax()
    out.flat[idx] = 1
    return out

def get_vectors_mask_wise(df_final):
    for_real_tho = pd.DataFrame()

    for l in range(len(df_final['path'])):
        img_test_path_curr = df_final['path'][l]
        masks = df_final['final_bbx'][l]
        vector_f = np.zeros([np.shape(bboxs)[0],])

        for i in range(len(masks)):
            temp_vector = []
            for j in range(len(bboxs)):
                value = iou(masks[i],bboxs[j])
                temp_vector.append(value)
            temp_vector = np.array(temp_vector)
            return_temp_vector = app_flat(temp_vector) 
            vector_f = vector_f + return_temp_vector
        for_real_tho = for_real_tho.append({'path': img_test_path_curr, 'vector': vector_f}, ignore_index=True)

    return for_real_tho

def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

### Processing ###
final, img_name = imgs_annot_aggregator(1)
result = bbox_points(final)
final_data, img_name_data = imgs_annot_aggregator(len(train_im_list))
result_data = bbox_points(final_data)
result_data.head()

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
ratio = int(512/new_size)

result_data = result_data.reset_index()
X = result_data[['file_path','bbox']]
Y = result_data['class']
data_frame_int = pd.DataFrame()

for i in range(len(result_data['file_path'])):
    print(result_data['file_path'][i])
    im, bb = bbox_utils.transformsXY(str(result_data['file_path'][i]), np.array(result_data['bbox'][i]),new_size,ratio)
    data_frame_int = data_frame_int.append({'file_path': result_data['file_path'][i], 'bbox': bb}, ignore_index=True)

df_final = image_merger(data_frame_int)
df_final.head()
img_test_path_curr = df_final['path'][0]
im, bb = bbox_utils.transformsXY(str(result_data['file_path'][i]), np.array(result_data['bbox'][i]),new_size,ratio)

bboxs = bbox_utils.generate(s, 130//4, 10, im.shape)
for_real_tho = get_vectors_mask_wise(df_final)

for_real_tho = for_real_tho.reset_index()
X = for_real_tho['path']
Y = for_real_tho['vector']
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

class AircraftDataset(Dataset):
    def __init__(self, paths, y, transforms=False):
        self.paths = paths.values
        self.y = y.values
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.y[idx]   
        return path, y

train_ds = AircraftDataset(X_train,y_train, transforms=True)
valid_ds = AircraftDataset(X_val,y_val)

batch_size = 64
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

class AircraftModel(nn.Module):
    def __init__(self):
        super(AircraftModel, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(3,192,kernel_size=7,stride=2),
            nn.LeakyReLU(0.1),
            MaxPool2d(2,2),
            Conv2d(192,256,3,1),
            nn.LeakyReLU(0.1),
            MaxPool2d(2,2),
            Conv2d(256,128,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(128,256,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(256,256,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(256,512,1,1),
            nn.LeakyReLU(0.1),
            MaxPool2d(2,2),
            Conv2d(512,256,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(256,512,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(512,256,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(256,512,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(512,256,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(256,512,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(512,256,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(256,512,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(512,1024,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(1024,512,1,1),
            nn.LeakyReLU(0.1),
            # MaxPool2d(2,2),
            Conv2d(512,1024,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(1024,512,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(512,1024,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(1024,512,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(512,1024,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(1024,1024,1,2),
            nn.LeakyReLU(0.1),
            Conv2d(1024,1024,1,1),
            nn.LeakyReLU(0.1),
            Conv2d(1024,1024,1,1),
            nn.LeakyReLU(0.1),
            nn.Flatten(start_dim=1)
        )

        self.connected = nn.Sequential(
            nn.Linear(128*128,out_features=1024, bias=False),
            nn.ReLU(), 
            nn.Linear(1024,out_features=len(bboxs), bias=False)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.connected(x)
        return x

model = AircraftModel()
device = torch.device('cuda')  # use cuda or cpu
print("Used device: ", device)
model.to(device)
print(model)

out = model(torch.randn(batchsize,3, 128, 128, device=device))
# print("Output shape:", out.size())
# print(f"Output logits:\n{out.detach().cpu().numpy()}")
optimizer = optim.Adam(model.parameters(), lr)  

convert_tensor = transforms.ToTensor()
step = 0
model.train()

train_accuracies = []
valid_accuracies = []
        
start_time = str(time.time()) 

titles = ['learning rate','batchsize', 'epochs', 'train_images','val_images', 's', 'loss_fn', 'optimizer']
hyper = [lr, batchsize,num_epochs,len(train_ds),len(valid_ds),s,loss_fn,optimizer]

PATH_HYPER = os.path.join(data_root, f'../AIRCRAFT/data/model/logs/hyper_{start_time}.csv')
with open(PATH_HYPER, 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(titles)
            wr.writerow(hyper)

for epoch in range(num_epochs):
    print("Epoch number: ", epoch)

    train_accuracies_batches = []
    for inputs, targets in train_dl:
        new_inputs = []
        for i in range(len(inputs)):
            im = bbox_utils.transformsXY_im(inputs[i],new_size)
            tensor = convert_tensor(im)
            new_inputs.append(tensor)
            
        new_inputs = torch.stack(tuple(new_inputs),0)
        new_inputs, targets = new_inputs.to(device), targets.to(device) 

        # Forward pass, compute gradients, perform one training step.
        optimizer.zero_grad()
        output = model(new_inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        
        # Increment step counter
        step += 1
        
        # Compute accuracy.
        # we use output & target

        # subtra = torch.subtract(output,targets)
        # squared = torch.square(subtra)
        # acc = torch.sum(squared)

        print("targets: ",targets)
        print("output: ",output)

        correct_match = 0
        correct_match += (output == targets).float().sum()
        accuracy_train = 100 * correct_match / len(inputs)
        print("accuracy_val: ", float(accuracy_train.numpy()))
        train_accuracies_batches.append(float(accuracy_train.cpu().numpy()))#acc.cpu().detach().numpy())

        PATH_TRAIN = os.path.join(data_root, f'../AIRCRAFT/data/model/logs/logs_train_{start_time}.csv')
        with open(PATH_TRAIN, 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow([accuracy_train.cpu().detach().numpy(), loss.cpu().detach().numpy()])#[int(acc)])

        if step % validation_every_steps == 0:
            
            # Append average training accuracy to list.
            train_accuracies.append(np.mean(train_accuracies_batches))
            
            train_accuracies_batches = []
        
            # Compute accuracies on validation set.
            valid_accuracies_batches = []
            with torch.no_grad():
                model.eval()
                for inputs, targets in valid_dl:
                    new_inputs = []

                    for i in range(len(inputs)):
                        im = bbox_utils.transformsXY_im(inputs[i],new_size)
                        tensor = convert_tensor(im)
                        new_inputs.append(tensor)
                        
                    new_inputs = torch.stack(tuple(new_inputs),0)
                    new_inputs, targets = new_inputs.to(device), targets.to(device) 

                    output = model(new_inputs)
                    loss = loss_fn(output, targets)

                    print("targets: ",targets)
                    print("output: ",output)

                    correct_match = 0
                    correct_match += (output == targets).float().sum()
                    accuracy_val = 100 * correct_match / len(inputs)
                    print("accuracy_val: ", float(accuracy_val.numpy()))
                    train_accuracies_batches.append(float(accuracy_val.cpu().detach().numpy()))#acc.cpu().detach().numpy())

                    PATH_TRAIN = os.path.join(data_root, f'../AIRCRAFT/data/model/logs/logs_val_{start_time}.csv')
                    with open(PATH_TRAIN, 'a', newline='') as myfile:
                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                        wr.writerow([accuracy_val.cpu().detach().numpy(), loss.cpu().detach().numpy()])#[int(acc)])

                    # subtra = torch.subtract(output,targets)
                    # squared = torch.square(subtra)
                    # acc = torch.sum(squared)/len(bboxs)

                    # valid_accuracies_batches.append(acc.cpu().detach().numpy()* len(inputs))

                    # PATH_VAL = os.path.join(data_root, f'../AIRCRAFT/data/model/logs/logs_val_{start_time}.csv')
                    # with open(PATH_VAL, 'a', newline='') as myfile:
                    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    #     wr.writerow([int(acc)])
                    # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                model.train()
                
            # Append average validation accuracy to list.
            valid_accuracies.append(np.sum(valid_accuracies_batches) / len(X_train))
     
            print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
            print(f"             test accuracy: {valid_accuracies[-1]}")

print("Finished training.")

#PATH = os.path.join(data_root, f'../AIRCRAFT/data/model/{start_time}.pth')
#torch.save(model, PATH)