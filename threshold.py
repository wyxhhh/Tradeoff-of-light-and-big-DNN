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
import joblib
from sklearn import ensemble
from sklearn.metrics import mean_squared_error 
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import sys
sys.path.append('model')
from smallnet import *


batch_size = 30
learning_rate = 1e-1
num_epoches = 10
complexi = [(668830.0, 109295), (321426.0, 40482), (53766.0, 4867), (27676.0, 9615), (1768060.0, 1446030), (524030.0, 363020), (133070.0, 93020), (35330.0, 25520)]

train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)
# print(train_dataset)

# decide the dataset
train_dataset = torch.utils.data.random_split(train_dataset, [10000, len(train_dataset)-10000])[0]
# print(len(train_dataset))

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

# test_dataset = torch.utils.data.random_split(test_dataset, [100, len(test_dataset)-100])[0]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# decide the small network
net = Mini8()
n = 8
m = 60000
net.load_state_dict(torch.load('weights/MINI8_60000.pth'))
net = net.cuda()

net1 = LeNet()
net1.load_state_dict(torch.load('weights/LeNet.pth'))
net1 = net1.cuda()


# Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

net.eval()
net1.eval()

clf = joblib.load("gra_boost_1.model")
lamda = 0.1
# f = open(name, 'w')


def  test(threshold):
    fre = 0
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        with torch.no_grad():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        out = net(img)
        inp = list()
        for i in range(10):
            inp.append(float(out[0][i]))
        inp.append(int(complexi[n-1][0])/668830.0)
        inp.append(int(complexi[n-1][1])/109295.0)
        inp.append(60000/60000)

        score = clf.predict([inp])[0]
        # print(score)
        loss = criterion(out, label)
        _, pr = torch.max(out, 1)

        eval_loss += loss.data.item() * label.size(0)
        if (score < threshold):
            fre += 1
            out1 = net1(img)
            _, pred = torch.max(out1, 1)
        else:
            _, pred = torch.max(out, 1)

        num_correct = (pred == label).sum()
        eval_acc += num_correct.data.item()

    return eval_acc/len(test_dataset) - lamda * fre/len(test_dataset)

depth = 13
step_threshold = 0.01
original_threshold = 0
step = (2 ** depth) * step_threshold
now_value = 0
next_value = 0
direction = 1
threshold = original_threshold
best_threshold = [0,0]

val = list()
while step >= step_threshold:
    now_value = next_value
    next_value = test(threshold)
    print(threshold, next_value)
    if next_value > best_threshold[0]:
        best_threshold[0] = next_value
        best_threshold[1] = threshold
    if next_value < now_value:
        direction = 1 - direction
        step = step / 2.0
    if direction:
        threshold = threshold + step
    else:
        threshold = threshold - step
    val.append(next_value)
print(best_threshold)
