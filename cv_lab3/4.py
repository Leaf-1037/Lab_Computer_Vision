import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#用于加载图片对数据
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __getitem__(self, index):
        img1, label1 = self.mnist_dataset[index]
        img2, label2 = self.mnist_dataset[np.random.choice(len(self.mnist_dataset))]

        # 生成相同/不同标签
        is_same = 1 if label1 == label2 else 0

        return img1, img2, torch.tensor(is_same, dtype=torch.float32)

    def __len__(self):
        return len(self.mnist_dataset)


# 定义孪生卷积神经网络模型
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1600, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        
    def forward_one(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        return x
    
    def forward(self, x1, x2):
        x1 = self.forward_one(x1)
        x2 = self.forward_one(x2)
        # 计算其几何距离
        x = torch.abs(x1 - x2)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# 使用MNISTDataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = MNIST(root='/data/', train=True, download=True, transform=transform)
mnist_test = MNIST(root='/data/', train=False, download=True, transform=transform)

train_dataset = MNISTDataset(mnist_train)
test_dataset = MNISTDataset(mnist_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义损失函数和优化器
model = SiameseNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# 存储训练过程中的损失和准确率
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
loss_log = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_train = 0
    batch_idx = 0

    for input1, input2, labels in train_loader:
        batch_idx += 1
        optimizer.zero_grad()
        outputs = model(input1, input2)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_train += (torch.round(torch.sigmoid(outputs)) == labels.view(-1, 1)).sum().item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            loss_log.append(loss.item())

    avg_loss = total_loss / len(train_loader)
    accuracy_train = correct_train / len(train_dataset)

    # 测试模型
    model.eval()
    total_loss = 0.0
    correct_test = 0

    with torch.no_grad():
        for input1, input2, labels in test_loader:
            outputs = model(input1, input2)
            loss = criterion(outputs, labels.view(-1, 1))
            total_loss += loss.item()
            correct_test += (torch.round(torch.sigmoid(outputs)) == labels.view(-1, 1)).sum().item()

    avg_loss_test = total_loss / len(test_loader)
    accuracy_test = correct_test / len(test_dataset)

    # 存储训练过程中的损失和准确率
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy_train)
    test_losses.append(avg_loss_test)
    test_accuracies.append(accuracy_test)

    # 输出训练和测试结果
    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy_train:.4f}, '
          f'Test Loss: {avg_loss_test:.4f}, Test Accuracy: {accuracy_test:.4f}')

import matplotlib.pyplot as plt

# 绘制训练和测试过程中的损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和测试过程中的准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
