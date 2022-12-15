from __future__ import print_function, division
from torchmetrics.classification import BinaryConfusionMatrix
from model import AircraftModel
import torch.nn as nn
from dataset import get_mean_and_std
from torchvision import datasets as data, transforms
from globals import root_directory
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from dataset_airplane import RareplanesDataset, ToTensor, Resize, ToClassificationTask, RandomTranslate, Normalize, RandomHSV, RandomRotate, RandomScale, RandomShear, RandomHorizontalFlip, CenterCrop
from visualize import show_image
import torchvision


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

normalization = ((0.0725, 0.1074, 0.0950), (0.8838, 0.9833, 0.9969))
# mean, std = get_mean_and_std(dataloader)
# print("mean and std: \n", mean, std)
# trues 3625
# falses 2190


rareplanes_dataset = RareplanesDataset(json_file=Path(root_directory, 'data/rareplanes/train.json'),
                                       root_dir=Path(root_directory, 'data/rareplanes/train/'),
                                       transform=transforms.Compose([
                                           RandomRotate(180),
                                           RandomShear(0.2),
                                           RandomScale(0.2),
                                           RandomTranslate(0.2, diff=True),
                                           RandomHSV(25, 25, 25),
                                           RandomHorizontalFlip(0.5),
                                           ToTensor(),
                                           CenterCrop(256),
                                           ToClassificationTask(),
                                           Normalize(*normalization)
                                       ]))


dataloader = DataLoader(rareplanes_dataset, batch_size=256, shuffle=True, num_workers=0, collate_fn=RareplanesDataset.custom_collate)

print_every_mini_batch = 5

metric = BinaryConfusionMatrix()

net = AircraftModel()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    running_confusion_matrix = torch.tensor([[0, 0], [0, 0]])
    for i, sample in enumerate(dataloader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(sample['image'])
        loss = criterion(outputs, sample['has_plane'])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_confusion_matrix += metric(outputs, sample['has_plane'])

        if i % print_every_mini_batch == print_every_mini_batch - 1:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_every_mini_batch:.3f} | confusion: {running_confusion_matrix}")
            # tensor([[  TN, FP],
            #         [  FN, TP]])
            running_loss = 0.0
            running_confusion_matrix = torch.tensor([[0, 0], [0, 0]])

print("Finished Training")

PATH = "./convulutionals.pth"
torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    pass
    # for i_batch, sample_batched in enumerate(dataloader):

    #     print(i_batch)
    # for i in range(len(sample_batched['image'])):
    #     print(sample_batched['has_plane'][i])
    #     show_image(sample_batched['image'][i], sample_batched['bboxs'][i], normalization_used=normalization)

    # # observe 4th batch and stop.
    # if i_batch == 3:
    #     break
