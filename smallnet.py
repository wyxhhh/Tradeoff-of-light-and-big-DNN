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

batch_size = 199
learning_rate = 1e-1
num_epoches = 20

train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)
# print(train_dataset)

# train_dataset = torch.utils.data.random_split(train_dataset, [5000, len(train_dataset)-5000])[0]
# print(len(train_dataset))

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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
net = LeNet()
# net.load_state_dict(torch.load('LeNet.pth'))
# print(net)

net = net.cuda()


# Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
#summary(Mini8().cuda(), input_size=(1, 28, 28))

for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.cuda()
        label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        # 向前传播
        out = net(img)
        loss = criterion(out, label)
        # print(loss.data, label)
        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.data.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
net.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    img = Variable(img, volatile=True).cuda()
    label = Variable(label, volatile=True).cuda()
    out = net(img)
    # print(out)
    loss = criterion(out, label)
    eval_loss += loss.data.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.data.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))
print()

# 保存模型
torch.save(net.state_dict(), './LeNet.pth')