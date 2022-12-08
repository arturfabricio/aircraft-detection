import torch.nn as nn
import bbox_utils
import numpy as np

s = 9
bboxs = bbox_utils.generate(s, 3, 64, (128, 128))
np_bboxs = np.asarray(list(map(lambda BBOX: [BBOX.arr[0], BBOX.arr[1], BBOX.arr[2], BBOX.arr[3]], bboxs)))

class AircraftModel(nn.Module):
    def __init__(self):
        super(AircraftModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=7, stride=2),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 256, 3, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 128, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 256, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 1, 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Flatten(start_dim=1),
            #nn.Dropout(0.5)
        )

        self.connected = nn.Sequential(
            # (128*128,out_features=1024, bias=False),
            nn.Linear(in_features=2048, out_features=len(bboxs), bias=False),
            nn.Sigmoid(),
            # nn.Linear(in_features=64, out_features=len(bboxs), bias=False)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.connected(x)
        return x

