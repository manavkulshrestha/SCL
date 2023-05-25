import torch
from torch import nn


class BaseMLP(nn.Module):
    def __init__(self, N, D):
        super(BaseMLP, self).__init__()
        self.N = N
        self.D = D

        self.inp_size = 2*N*D
        self.hidden_size = 128

        self.features = nn.Sequential(
            nn.Linear(self.inp_size, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_size),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, N),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x.view(-1, self.inp_size))
        x = self.classifier(x)
        return x


class BaseConvNet(nn.Module):
    def __init__(self):
        super(BaseConvNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # Output size: (16, 10, 511)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Output size: (16, 5, 255)
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # Output size: (32, 5, 255)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Output size: (32, 2, 127)
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # Output size: (64, 2, 127)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # Output size: (64, 1, 63)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 1 * 63, 128),  # Input size: (64 * 1 * 63)
            nn.ReLU(),
            nn.Linear(128, 10),  # Output size: (10,)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
