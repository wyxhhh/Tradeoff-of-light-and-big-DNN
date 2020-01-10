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

batch_size = 200
learning_rate = 1e-1
num_epoches = 40

def trainval(batch_size, learning_rate, num_epoches):

    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transforms.ToTensor(), download=True)
    
    # use different size of train set for training
    train_dataset = torch.utils.data.random_split(train_dataset, [60000, len(train_dataset)-60000])[0]

    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    net = Mini5()
    # net.load_state_dict(torch.load('weights/Mini5_60000.pth'))
    # print(net)

    net = net.cuda()
    # Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # train
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
            # Forward propagation
            out = net(img)
            loss = criterion(out, label)
            # print(loss.data, label)
            running_loss += loss.data.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            accuracy = (pred == label).float().mean()
            running_acc += num_correct.data.item()
            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 300 == 0:
                print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch + 1, num_epoches, running_loss / (batch_size * i),
                    running_acc / (batch_size * i)))
    # eval
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

    # save model
    torch.save(net.state_dict(), 'weights/Mini5_60000.pth')


if __name__=='__main__':
    trainval(batch_size, learning_rate, num_epoches)
