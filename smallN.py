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

test_dataset = torch.utils.data.random_split(test_dataset, [100, len(test_dataset)-100])[0]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50*4*4, 500)
        self.fc2 = nn.Linear(500, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 12
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) # 4
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#summary(LeNet().cuda(), input_size=(1, 28, 28))

class Mini1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 25, 5)
        self.fc1 = nn.Linear(4*4*25, 250)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 12*12*10
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) # 4*4*25
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
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
        self.fc1 = nn.Linear(4*4*15, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 12*12*7
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) # 4*4*15
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
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
        self.fc1 = nn.Linear(4*4*5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 12*12*2
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) # 4*4*5
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
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
        self.fc1 = nn.Linear(6*6*5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 13*13*2
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) # 4*4*5
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
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
        self.fc1 = nn.Linear(12*12*20, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), (2, 2)) # 12*12*2
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
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
        self.fc1 = nn.Linear(12*12*10, 250)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), (2, 2)) # 12*12*2
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#summary(Mini6().cuda(), input_size=(1, 28, 28))

class Mini7(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 5,  stride=2)
        self.fc1 = nn.Linear(6*6*10, 250)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), (2, 2)) # 6*6*10
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
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
        self.fc1 = nn.Linear(3*3*10, 250)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), (2, 2)) # 3*3*10
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
		
# decide the small network
net = Mini8()
n = 8
net.load_state_dict(torch.load('D:\Overwatch\weights/MINI8_10000.pth'))
net = net.cuda()

net1 = LeNet()
net1.load_state_dict(torch.load('D:\Overwatch\weights/LeNet.pth'))
net1 = net1.cuda()


# Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
# summary(Mini8().cuda(), input_size=(1, 28, 28))

# for epoch in range(num_epoches):
#     print('epoch {}'.format(epoch + 1))
#     print('*' * 10)
#     running_loss = 0.0
#     running_acc = 0.0
#     for i, data in enumerate(train_loader, 1):
#         # print(i)
#         img, label = data
#         img = img.cuda()
#         label = label.cuda()
#         img = Variable(img)
#         label = Variable(label)
#         # 向前传播
#         out = net(img)
#         loss = criterion(out, label)
#         # print(loss.data, label)
#         running_loss += loss.data.item() * label.size(0)
#         _, pred = torch.max(out, 1)
#         num_correct = (pred == label).sum()
#         accuracy = (pred == label).float().mean()
#         running_acc += num_correct.data.item()
#         # 向后传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if i % 300 == 0:
#             print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
#                 epoch + 1, num_epoches, running_loss / (batch_size * i),
#                 running_acc / (batch_size * i)))
net.eval()
net1.eval()
name = "training_data_Mini"+str(n)+".txt"

f = open(name, 'w')
lamda = 0.1
ita = [0.05*i for i in range(1, 21)]
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
        # for i in range(10):
        #     wr.append(float(out[0][i]))
        # wr.append(int(complexi[n-1][0]))
        # wr.append(int(complexi[n-1][1]))
        # wr.append(10000)
        loss = criterion(out, label)
        _, pr = torch.max(out, 1)
        # if (pr != label):
        #     wr.append(-1)
        # else:
        #     wr.append(1)
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
        # if (pred != label):
        #     print(m1, m2)
        #     sorted1, indices1 = torch.sort(out1, descending=True, dim=-1)
        #     print(float(sorted1[0][0]), float(sorted1[0][1]))
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data.item()
        # w.append(wr)
    ac.append((eval_acc/len(test_dataset) - lamda * fre/len(test_dataset)))
print(ac, ac.index(max(ac)))
y = 0.05*(ac.index(max(ac))+1)
w = list()
fre = 0
eval_loss = 0
eval_acc = 0
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
    wr.append(10000)
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
        # if (pred != label):
        #     print(m1, m2)
        #     sorted1, indices1 = torch.sort(out1, descending=True, dim=-1)
        #     print(float(sorted1[0][0]), float(sorted1[0][1]))
    num_correct = (pred == label).sum()
    eval_acc += num_correct.data.item()
    w.append(wr)
for item in w:
    item.append(eval_acc / len(test_dataset))
    item.append(fre/len(test_dataset))
    # print(len(item))
    f.write(str(item))
    f.write('\n')

# 保存模型
# torch.save(net.state_dict(), 'D:\Overwatch\weights/MINI6_10000.pth')