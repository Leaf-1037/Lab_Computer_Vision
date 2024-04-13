import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F
import numpy as np

num_epochs = 20

# 测试结果列表
log_loss_4_mini_batch = []
log_train_losses = []
log_train_acc = []
log_test_losses = []
log_test_acc = []

#使用cuda训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#用于加载图片对数据
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx]
        img2, label2 = self.dataset[np.random.choice(len(self.dataset))]
        labels = (label1 == label2)  # 标签是否相同
        return img1, img2, torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)


# 定义孪生卷积神经网络模型
# 定义卷积神经网络模型
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(4*4*64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward_shared(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 4*4*64)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return x

    def forward(self, x1, x2):
        x1 = self.forward_shared(x1)
        x2 = self.forward_shared(x2)
        x = torch.abs(x1 - x2)
        out = self.fc3(x)
        return out


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
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_train = 0
    batch_idx = 0

    for input1, input2, labels in train_loader:
        input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
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
            log_loss_4_mini_batch.append(loss.item())

    loss_train = total_loss / len(train_loader)
    acc_train = correct_train / len(train_dataset)
    
    model.eval()
    total_loss = 0.0
    correct_test = 0

    with torch.no_grad():
        for input1, input2, labels in test_loader:
            input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
            outputs = model(input1, input2)
            loss = criterion(outputs, labels.view(-1, 1))
            total_loss += loss.item()
            correct_test += (torch.round(torch.sigmoid(outputs)) == labels.view(-1, 1)).sum().item()

    loss_test = total_loss / len(test_loader)
    acc_test = correct_test / len(test_dataset)

    # 存储训练过程中的损失和准确率
    log_train_losses.append(loss_train)
    log_train_acc.append(acc_train)
    log_test_losses.append(loss_test)
    log_test_acc.append(acc_test)

    # 输出训练和测试结果
    print(f'Epoch {epoch + 1}/{num_epochs}, '
        f'Training Loss: {loss_train:.4f}, Training Accuracy: {acc_train:.4f}, '
        f'Test Loss: {loss_test:.4f}, Test Accuracy: {acc_test:.4f}')

if __name__ == '__main__':

    # 打印结果，以便绘图
    print(log_loss_4_mini_batch)
    print(log_train_losses)
    print(log_test_losses)

    print(log_train_acc)
    print(log_test_acc)


