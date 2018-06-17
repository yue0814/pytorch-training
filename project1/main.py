# -*- coding: utf-8 -*- 
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torchvision import datasets, transforms


DOWNLOAD_MNIST = False
BATCH_SZ = 512
EPOCH = 1500
LR = 0.0015

# Device configuration
device = torch.Tensor.device('cuda' if torch.cuda.is_available() else 'cpu')

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='./mnist',
        train=True,
        transform=transforms.ToTensor(),
        download=DOWNLOAD_MNIST),
    batch_size=BATCH_SZ,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='./mnist',
        train=False,
        transform=transforms.ToTensor(),
        download=DOWNLOAD_MNIST),
    batch_size=BATCH_SZ,
    shuffle=True
)

del train_loader
del test_loader

print('===>Loading data')
with open('./mnist/processed/training.pt', 'rb') as f:
    training_set = torch.load(f)

with open('./mnist/processed/test.pt', 'rb') as f:
    test_set = torch.load(f)
print('<===Done')

# reshape image to 60000*784
training_data = training_set[0].view(-1, 784)
training_data = training_data.float()
training_labels = training_set[1]

test_data = test_set[0].view(-1, 784)
test_data = test_data.float()
test_labels = test_set[1]

train_dataset = torch.utils.data.TensorDataset(training_data, training_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SZ, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SZ, shuffle=True)

del training_set
del test_set


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        init.normal_(self.fc1.weight, mean=0, std=0.1)
        self.fc2 = nn.Linear(n_hidden, n_output)
        init.normal_(self.fc2.weight, mean=0, std=0.1)
        self.dropout = nn.Dropout(0.9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


net = Net(n_feature=784, n_hidden=625, n_output=10)
optimizer = optim.Adam(net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# Train the model
total_loss = 0
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = Variable(b_x), Variable(b_y).long()
        output = net(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    if epoch % 100 == 0:
        print("After {0} epoch, loss is {1}".format(epoch + 1, total_loss / (epoch + 1)))

# test the model
correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images)
    outputs = net(images)
    _, predicted = torch.Tensor.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('accuracy of the model %.2f' % (100 * correct / total))