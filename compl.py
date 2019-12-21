import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.autograd import Variable
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import optim
import numpy as np
from ptflops import get_model_complexity_info


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(50*4*4, 500)
        self.fc2 = nn.Linear(500, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu1(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# summary(LeNet().cuda(), input_size=(1, 28, 28))

class Mini1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 25, 5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(4*4*25, 250)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#summary(Mini1().cuda(), input_size=(1, 28, 28))

class Mini2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 7, 5)
        self.conv2 = nn.Conv2d(7, 15, 5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(4*4*15, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#summary(Mini2().cuda(), input_size=(1, 28, 28))

class Mini3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.conv2 = nn.Conv2d(2, 5, 5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(4*4*5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#summary(Mini3().cuda(), input_size=(1, 28, 28))

class Mini4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 2)
        self.conv2 = nn.Conv2d(2, 5, 2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(6*6*5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#summary(Mini4().cuda(), input_size=(1, 28, 28))

class Mini5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 20, 5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(12*12*20, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.maxpool1(x) # 12*12*2
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#summary(Mini5().cuda(), input_size=(1, 28, 28))

class Mini6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(12*12*10, 250)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

summary(Mini6().cuda(), input_size=(1, 28, 28))

class Mini7(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 5,  stride=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(6*6*10, 250)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#summary(Mini7().cuda(), input_size=(1, 28, 28))

class Mini8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 5, stride=4)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(3*3*10, 250)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # x = nn.MaxPool2d(nn.ReLU(self.conv(x)), (2, 2)) # 3*3*10
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

lp = list()

net1 = Mini1()
flops, params = get_model_complexity_info(net1, (1, 28, 28), as_strings=False, print_per_layer_stat=False)
lp.append((flops, params))
net2 = Mini2()
flops, params = get_model_complexity_info(net2, (1, 28, 28), as_strings=False, print_per_layer_stat=False)
lp.append((flops, params))
net3 = Mini3()
flops, params = get_model_complexity_info(net3, (1, 28, 28), as_strings=False, print_per_layer_stat=False)
lp.append((flops, params))
net4 = Mini4()
flops, params = get_model_complexity_info(net4, (1, 28, 28), as_strings=False, print_per_layer_stat=False)
lp.append((flops, params))
net5 = Mini5()
flops, params = get_model_complexity_info(net5, (1, 28, 28), as_strings=False, print_per_layer_stat=False)
lp.append((flops, params))
net6 = Mini6()
flops, params = get_model_complexity_info(net6, (1, 28, 28), as_strings=False, print_per_layer_stat=False)
lp.append((flops, params))
net7 = Mini7()
flops, params = get_model_complexity_info(net7, (1, 28, 28), as_strings=False, print_per_layer_stat=False)
lp.append((flops, params))
net8 = Mini8()
flops, params = get_model_complexity_info(net8, (1, 28, 28), as_strings=False, print_per_layer_stat=False)
lp.append((flops, params))
print(lp)