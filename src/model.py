import torch.nn as nn
import torch
import bbox_utils
import numpy as np

s = 2
bboxs = bbox_utils.generate(s, 3, 64, (128, 128))
np_bboxs = np.asarray(list(map(lambda BBOX: [BBOX.arr[0], BBOX.arr[1], BBOX.arr[2], BBOX.arr[3]], bboxs)))

class AircraftModel(nn.Module):
    def __init__(self):
        super(AircraftModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.10),
            #nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32,3,1,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        self.connected = nn.Sequential(
            # (128*128,out_features=1024, bias=False),
            nn.Linear(in_features=1, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=len(bboxs), bias=False),
            # nn.Linear(in_features=64, out_features=len(bboxs), bias=True),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=len(bboxs), bias=False)
        )

    def forward(self, x):
        # print("x ", x.size())
        x = torch.flatten(x, start_dim=1)
        x = torch.mean(x,1)
        x = torch.unsqueeze(x, 1)
        # print("x ", x.size())
        # x = self.conv(x)
        x = self.connected(x)
        x = torch.clamp(x, min=0, max=1)
        return x

