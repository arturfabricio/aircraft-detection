import torch.nn as nn
import torch
import bbox_utils
import numpy as np

torch.set_printoptions(precision=1, sci_mode=False)
IMAGE_SIZE = bbox_utils.IMAGE_SIZE
s = 9
bboxs = bbox_utils.generate(s, 3, 64, IMAGE_SIZE)
print('Target vector length', len(bboxs))
np_bboxs = np.asarray(list(map(lambda BBOX: [BBOX.arr[0], BBOX.arr[1], BBOX.arr[2], BBOX.arr[3]], bboxs)))

class AircraftModel(nn.Module):
    def __init__(self):
        super(AircraftModel, self).__init__()

        # image_size = floor(((image_dim_x-kernal_size+1)/conv_stride)/max_pool_kernal)
        # channels = image_size*conv_out*2

        self.name = "Our Yolo v2"
        self.conv = nn.Sequential(
            # 3 | 256
            nn.Conv2d(3, 32, kernel_size=7),
            nn.BatchNorm2d(32), 
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(0.1),

            # 32 | 125

            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(0.1),

            # 64 | 60

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            # # 128 | 58

            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            # # 256 | 56

            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # # 512 | 54

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(0.1),

            # # 512 | 26

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # # 512 | 24

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # # 512 | 22

            nn.Conv2d(512, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            # # 1024 | 20

            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            # # 1024 | 18

            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(0.1),

            # 1024 | 8

            nn.Flatten(start_dim=1),
            # nn.Dropout(0.5)
        )

        self.connected = nn.Sequential(
            nn.Linear(in_features=65536, out_features=4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=len(bboxs), bias=False),
        )

        

        def get_xaviar_gain(layer):
            class_name = layer.__class__.__name__
            if class_name == 'Conv2d':
                return 1
            elif class_name == 'Linear':
                return 1
            elif class_name.find('BatchNorm') != -1: # BatchNorm1d and BatchNorm2d
                return 1
            elif class_name == 'LeakyReLU':
                return nn.init.calculate_gain('leaky_relu', layer.negative_slope)
            elif class_name == 'MaxPool2d':
                return 1
            else:
                raise Exception(f'Unsuported layer "{class_name}"')

        def initialize_weights(sequential):
            for idx in range(len(sequential)):

                layer = sequential[idx]
                gain = 1
                if idx - 1 >= 0:
                    gain = get_xaviar_gain(sequential[idx - 1])

                class_name = layer.__class__.__name__

                if class_name == 'Conv2d':
                    nn.init.xavier_uniform_(layer.weight, gain)
                elif class_name == 'BatchNorm2d':
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif class_name == 'BatchNorm1d':
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif class_name == 'Linear':
                    nn.init.xavier_uniform_(layer.weight, gain)
                    # Our linear layer has no bias
                    # torch.nn.init.zeros_(m.bias)

        initialize_weights(self.conv)
        initialize_weights(self.connected)



    def forward(self, x):
        x = self.conv(x)
        x = self.connected(x)
        return x
