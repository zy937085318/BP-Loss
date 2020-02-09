import torch
from torch import nn


class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.relu1(x1))
        x3 = self.conv3(self.relu2(x2))
        x4 = self.conv4(self.relu2(x3))

        return [x4, x3, x2, x1]


class CycleNet(nn.Module):
    def __init__(self):
        super(CycleNet, self).__init__()
        self.b_block = block()
        self.m_block = block()
    def forward(self, x):
        t1 = self.b_block(x)
        t2 = self.m_block(t1[0])
        return t1 + t2
