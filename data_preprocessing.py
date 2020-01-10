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
import sys
sys.path.append('model')
from smallnet import *
from ptflops import get_model_complexity_info

batch_size = 30
learning_rate = 1e-1
num_epoches = 10

# complexity of mini1-8
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
m = 10000
net.load_state_dict(torch.load('weights/MINI8_10000.pth'))
net = net.cuda()

net1 = LeNet()
net1.load_state_dict(torch.load('weights/LeNet.pth'))
net1 = net1.cuda()


# Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

net.eval()
net1.eval()
name = "training_data_Mini"+str(n)+'_'+str(m)+".txt"

f = open(name, 'w')
lamda = 0.1
ita = [0.02*i for i in range(1, 51)]
ac = list()

# [out--10D, complexity-2D, data-1D, small_net_result-1D, Acc-1D, frequency-1D]
for z in ita:
    fre = 0
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        # wr = list()
        img, label = data
        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
        out = net(img)
        loss = criterion(out, label)
        _, pr = torch.max(out, 1)
    
        eval_loss += loss.data.item() * label.size(0)
        sorted, indices = torch.sort(out, descending=True, dim=-1)
        m1 = float(sorted[0][0])
        m2 = float(sorted[0][1])
        if (m1 - m2 < z):
            fre += 1
            out1 = net1(img)
            _, pred = torch.max(out1, 1)
        else:
            _, pred = torch.max(out, 1)
        
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data.item()
        # w.append(wr)
    print(eval_acc/len(test_dataset) - lamda * fre/len(test_dataset))
    ac.append((eval_acc/len(test_dataset) - lamda * fre/len(test_dataset)))
print(ac, ac.index(max(ac)))
y = 0.02*(ac.index(max(ac))+1)
w = list()
fre = 0
eval_loss = 0
eval_acc = 0

# use trained small network to generate train set for determiner
for data in test_loader:
    wr = list()
    img, label = data
    img = Variable(img, volatile=True).cuda()
    label = Variable(label, volatile=True).cuda()
    out = net(img)
    for i in range(10):
        wr.append(float(out[0][i]))
    wr.append(int(complexi[n-1][0]))
    wr.append(int(complexi[n-1][1]))
    wr.append(m)
    loss = criterion(out, label)
    _, pr = torch.max(out, 1)
    if (pr != label):
        wr.append(-1)
    else:
        wr.append(1)
    eval_loss += loss.data.item() * label.size(0)
    sorted, indices = torch.sort(out, descending=True, dim=-1)
    m1 = float(sorted[0][0])
    m2 = float(sorted[0][1])
    if (m1 - m2 < y):
        fre += 1
        out1 = net1(img)
        _, pred = torch.max(out1, 1)
    else:
         _, pred = torch.max(out, 1)
       
    num_correct = (pred == label).sum()
    eval_acc += num_correct.data.item()
    w.append(wr)
for item in w:
    item.append(eval_acc / len(test_dataset))
    item.append(fre/len(test_dataset))
    # print(len(item))
    f.write(str(item))
    f.write('\n')
