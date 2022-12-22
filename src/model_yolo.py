from torch import nn
from torchvision.models import resnet34

class YOLOv1(nn.Module):
    """YOLOv1 model structure
    yolo-v1 = conv + fc
    """

    def __init__(self, S, B, num_classes):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes

        # conv part
        self.conv_layers = nn.Sequential(
            # 448*448*3 -> 112*112*192
            nn.Conv2d(3, 192, 7, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 112*112*192 ->56*56*256
            nn.Conv2d(192, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 56*56*256 -> 28*28*512
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 28*28*512 -> 14*14*1024
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 14*14*1024 -> 7*7*1024
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            # 7*7*1024 -> 7*7*1024
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # full connection part
        self.fc_layers = nn.Sequential(
            nn.LazyLinear(4096), #7 * 7 * 1024, 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.num_classes)),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()  # normalized to 0~1
        )

    def forward(self, x):
        out = self.conv_layers(x)  # b*1024*7*7
        out = out.view(out.size()[0], -1)  # b*50176
        out = self.fc_layers(out)
        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return out

class YOLOv1ResNet(nn.Module):
    """YOLOv1-Resnet model structure
    yolo-v1 resnet = resnet(backbone) + conv + fc
    """

    def __init__(self, S, B, num_classes):
        super(YOLOv1ResNet, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        # self.resnet = resnet18()
        self.resnet = resnet34(weights='DEFAULT')
        # print(self.resnet.fc.in_features)
        # print(*list(self.resnet.children())[-2:])  # show last two layers

        # backbone part, (cut resnet's last two layers)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])

        # conv part
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            # nn.Conv2d(1024, 1024, 3, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(0.1, inplace=True),
        )

        # full connection part
        self.fc_layers = nn.Sequential(
            nn.LazyLinear(4096), #7 * 7 * 1024,
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.num_classes)),
            nn.Sigmoid()  # normalized to 0~1
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.conv_layers(out)
        out = out.view(out.size()[0], -1)
        out = self.fc_layers(out)
        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return out
