import torch.nn as nn
import torch
import bbox_utils
import numpy as np
from pathlib import Path

torch.set_printoptions(precision=1, sci_mode=False)
IMAGE_SIZE = bbox_utils.IMAGE_SIZE
s = 7
bboxs = bbox_utils.generate(s, 3, IMAGE_SIZE[0]/2, IMAGE_SIZE)
print('Target vector length', len(bboxs))
np_bboxs = np.asarray(list(map(lambda BBOX: [BBOX.arr[0], BBOX.arr[1], BBOX.arr[2], BBOX.arr[3]], bboxs)))

class AircraftModel(nn.Module):
    def __init__(self):
        super(AircraftModel, self).__init__()

        # image_size = floor(((image_dim_x-kernal_size+1)/conv_stride)/max_pool_kernal)
        # channels = image_size*conv_out*2

        self.name = "Our Yolo v4"
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
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(0.1),

            # # 128 | 29

            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(0.1),

            # # 256 | 14

            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # # 512 | 12

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # # 512 | 10

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # # 512 | 8

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # # 512 | 6

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Flatten(start_dim=1),
        )

        self.connected = nn.Sequential(
            nn.Linear(in_features=4608, out_features=4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
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
            elif class_name == 'Dropout':
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

        self.double()

    def load_weights(self, path : Path):
        
        DEBUG_LAYERS = False
        
        loaded_model = torch.load(path, map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        if DEBUG_LAYERS:
            print('------------ MODEL FORMAT -------------')
            for layer_name, tensor in self.named_parameters():
                print(layer_name, tensor.size())
            print('----------- LOADED MODEL --------------')
            for layer_name, tensor in loaded_model.named_parameters():
                print(layer_name, tensor.size())


        #model.load_state_dict(loaded_model.state_dict())
        self.load_state_dict(loaded_model)


    def forward(self, x):
        x = self.conv(x)
        x = self.connected(x)
        return x
