import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        # x : (batch_size, channels=1, height=224, width=224)
        x = self.conv_1(x)
        # x : (batch_size, channels=96, height=27, width=27)
        x = self.conv_2(x)
        # x : (batch_size, channels=256, height=13, width=13)
        x = self.conv_3(x)
        # x : (batch_size, channels=384, height=13, width=13)
        x = self.conv_4(x)
        # x : (batch_size, channels=384, height=13, width=13)
        x = self.conv_5(x)
        # x : (batch_size, channels=256, height=6, width=6)
        x = self.classifier(x)
        # x : (batch_size, num_classes=10)
        return x
