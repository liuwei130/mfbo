from __future__ import print_function
import os
import sys
import signal
import time
import pickle
import certifi

import numpy as np
import numpy.matlib
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

signal.signal(signal.SIGINT, signal.SIG_DFL)


'''
6: learning late {10^-4, 10^-3, 10^-2, 10^-1, 10^0}
5: batch size {32, 64, 128, 256, 512}
5: (2 ×) number of channels: {4, 8, 16, 32, 64}
4: drop out rate {1/16, 1/8, 1/4, 1/2}

cf.
https://github.com/pytorch/examples/blob/master/mnist/main.py
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''

class Net_MNIST(nn.Module):
    def __init__(self, ch1, ch2, drop_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(1, ch1, 3, 1)
        self.conv2 = nn.Conv2d(ch1, ch2, 3, 1)
        self.dropout1 = nn.Dropout(drop_rate)
        self.fc1 = nn.Linear(int(ch2 * 12**2), 10)

        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        # (batch_size, ch1, 26, 26)
        x = F.relu(x)
        x = self.conv2(x)
        # (batch_size, ch2, 24, 24)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # (batch_size, ch2, 12, 12)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # (batch_size, 64 * 12 * 12)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Net_CIFAR10(nn.Module):
    def __init__(self, ch1, ch2, drop_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch1, 5)
        self.dropout1 = nn.Dropout(drop_rate)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(ch1, ch2, 5)
        self.fc1 = nn.Linear(ch2 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / len(test_loader.dataset)


def main():
    os.environ['SSL_CERT_FILE'] = certifi.where()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')

    parser.add_argument('--ch1', type=int, default=16, metavar='CH1',
                        help='number of channel on first layer(default: 32)')
    parser.add_argument('--ch2', type=int, default=32, metavar='CH2',
                        help='number of channel on second layer(default: 32)')
    parser.add_argument('--drop-rate', type=float, default=0.25, metavar='DR',
                        help='drop rate (default: 0.25)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data-name', default='MNIST',
                        help='Data name (MNIST or CIFAR10)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    print('./cnn_{}_data/X_{}_{}_{}_{}_{}.pickle'.format(args.data_name, args.lr, args.batch_size, args.ch1, args.ch2, args.drop_rate))

    if args.data_name=='MNIST':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)

        model = Net_MNIST(args.ch1, args.ch2, args.drop_rate).to(device)
    elif args.data_name=='CIFAR10':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.CIFAR10('../data', train=False,
                        transform=transform)

        model = Net_CIFAR10(args.ch1, args.ch2, args.drop_rate).to(device)


    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


    data_list = list()
    X = np.array([args.lr, args.batch_size, args.ch1, args.ch2, args.drop_rate])

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        scheduler.step()
        print('accuracy:', accuracy)
        elapsed_time = time.time() - start_time
        print('elaplsed time:', elapsed_time)

        data_list.append(np.r_[X, np.array([epoch, elapsed_time, accuracy])])

    data = np.vstack(data_list)

    # with open('./cnn_'+args.data_name+'_data/X_'+str(round(log_C, 1))+'_'+str(round(log_gamma, 1))+'_'+str(data_size)+'.pickle', 'wb') as f:
    with open('./cnn_{}_data/X_{}_{}_{}_{}_{}.pickle'.format(args.data_name, args.lr, args.batch_size, args.ch1, args.ch2, args.drop_rate), 'wb') as f:
        pickle.dump(data, f)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()