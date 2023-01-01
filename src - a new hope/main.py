from __future__ import print_function, division
import pandas as pd
import json
from torchmetrics.classification import BinaryConfusionMatrix
from model import AircraftModel
import torch.nn as nn
from dataset import get_mean_and_std
from torchvision import datasets as transforms
from globals import root_directory
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from dataset_airplane import RareplanesDataset, ToTensor, Resize, ToClassificationTask, RandomTranslate, Normalize, RandomHSV, RandomRotate, RandomScale, RandomShear, RandomHorizontalFlip, CenterCrop
from visualize import show_image
from version_logger import VersionLogger, get_local_time


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Estimated
normalization = ((0.2851, 0.2534, 0.2222), (0.2009, 0.2023, 0.2001))
# mean, std = get_mean_and_std(dataloader)
# print("mean and std: \n", mean, std)
# trues 3625
# falses 2190

transform = transforms.Compose([
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
])


rareplanes_trainset = RareplanesDataset(json_file=Path(root_directory, 'data/annot/rareplanes_train.json'),
                                        root_dir=Path(root_directory, 'data/train/'),
                                        transform=transform)
rareplanes_testset = RareplanesDataset(json_file=Path(root_directory, 'data/annot/rareplanes_test.json'),
                                       root_dir=Path(root_directory, 'data/test/'),
                                       transform=transform)

test_loader = DataLoader(rareplanes_testset, batch_size=512, shuffle=True, num_workers=4, collate_fn=RareplanesDataset.collate_fn)
train_loader = DataLoader(rareplanes_trainset, batch_size=512, shuffle=True, drop_last=True, num_workers=4, collate_fn=RareplanesDataset.collate_fn)


logger = VersionLogger(Path(root_directory, f'./data/runs/{get_local_time().replace(":","_")}/'), print_to_console=True)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.print("Used device: ", device)

net = AircraftModel()
#Add below line to use weights from another model 
#net.load_weights(Path(root_directory, './data/runs/2022-12-17 17_03_59/models/convulutionals.pth'), device)
net.to(device)

criterion = nn.BCELoss()
metric = BinaryConfusionMatrix().to(device)
# no batch normalization lr=10e-2 works
# with batch normalization lr=10e-5 works might need even lower
# 10e-2 is to high, 10e-3 seems too high, 10e-4 works

# With batch norms and SILU
# started at 10e-3, then 10e-4

optimizer = torch.optim.SGD(net.parameters(), lr=10e-6, momentum=0.4)

validate_every_epochs = 10
print_every_mini_batch = 8

for epoch in range(10000):  # loop over the dataset multiple times
    running_loss = 0.0
    running_confusion_matrix = torch.tensor([[0, 0], [0, 0]])
    for i, sample in enumerate(train_loader):
        inputs, targets = sample['image'].to(device), sample['has_plane'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_confusion_matrix += metric(outputs, targets).cpu()

        if i % print_every_mini_batch == print_every_mini_batch - 1:
            logger.print(f"[{epoch + 1}, {i + 1:5d}] training loss: {running_loss / print_every_mini_batch:.3f} | confusion: {running_confusion_matrix}")
            # tensor([[  TN, FP],
            #         [  FN, TP]])
            running_loss = 0.0
            running_confusion_matrix = torch.tensor([[0, 0], [0, 0]])

    if epoch % validate_every_epochs == validate_every_epochs - 1:
        with torch.no_grad():
            running_validation_loss = 0.0
            running_validation_confusion_matrix = torch.tensor([[0, 0], [0, 0]])
            batches = 0
            net.eval()
            for sample in test_loader:
                inputs, targets = sample['image'].to(device), sample['has_plane'].to(device)

                outputs = net(inputs)
                running_validation_loss += criterion(outputs, targets).item()
                running_validation_confusion_matrix += metric(outputs, targets).cpu()
                batches += 1

            logger.print(f"[{epoch + 1}] validation loss: {running_validation_loss / batches:.3f} | confusion: {running_validation_confusion_matrix}")
            net.train()

    logger.save_model(net, 'convulutionals.pth')

logger.print("Finished Training")


if __name__ == '__main__':
    pass
