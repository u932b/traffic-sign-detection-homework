# import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43  # GTSRB as 43 classes


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv1_5 = nn.Conv2d(10, 10, kernel_size=5, padding=(2, 2))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_5 = nn.Conv2d(20, 20, kernel_size=5, padding=(2, 2))
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5, padding=(2,2))
        self.conv3_5 = nn.Conv2d(40, 40, kernel_size=5, padding=(5, 5))
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(40*5*5, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        # print(x.data.shape)
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv1(x)
        # x = F.relu(F.max_pool2d(self.conv1_5(self.conv1(x)), 2))
        # print(x.data.shape)
        x = self.conv_drop(F.relu(F.max_pool2d(self.conv1_5(x), 2)))
        # print(x.data.shape)
        x = self.conv2(x)
        # print(x.data.shape)
        x = self.conv_drop(F.relu(F.max_pool2d(self.conv2_5(x), 2)))
        # print(x.data.shape)
        x = self.conv3(x)
        # print(x.data.shape)
        x = self.conv_drop(F.relu(F.max_pool2d(self.conv3_5(x), 2)))
        # print(x.data.shape)
        x = x.view(-1, 40*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
