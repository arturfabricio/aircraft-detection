import torch.nn as nn
import torch
from pathlib import Path


class AircraftModel(nn.Module):
    def __init__(self):
        super().__init__()

        # image_size = floor(((image_dim_x-kernal_size+1)/conv_stride)/max_pool_kernal)
        # channels = image_size*conv_out*2

        self.name = "Our Yolo v10 Convolutionals only"
        self.conv = nn.Sequential(
            # 3 | 256
            nn.Conv2d(3, 32, kernel_size=7),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            # 32 | 125

            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            # 64 | 60

            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            # # 128 | 29

            nn.Conv2d(128, 256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.SiLU(),

            # # 256 | 14

            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.SiLU(),

            # # 512 | 12

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.SiLU(),

            # # 512 | 10

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.SiLU(),

            # # 512 | 8

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.SiLU(),

            # # 512 | 6

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.SiLU(),

            nn.Flatten(start_dim=1),
        )

        self.connected = nn.Sequential(
            nn.Linear(in_features=4608, out_features=1),
            nn.Sigmoid(),
        )

    def load_weights(self, path: Path, device):

        DEBUG_LAYERS = False

        loaded_model = torch.load(path, device)

        if DEBUG_LAYERS:
            print('------------ MODEL FORMAT -------------')
            for layer_name, tensor in self.named_parameters():
                print(layer_name, tensor.size())
            print('----------- LOADED MODEL --------------')
            for value in loaded_model:
                print(value)

        print(self.load_state_dict(loaded_model, strict=False))

    def forward(self, x):
        x = self.conv(x)
        print("Output", x)
        x = self.connected(x)
        return x
