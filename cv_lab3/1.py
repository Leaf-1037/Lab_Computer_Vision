import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# 设置随机种子
torch.manual_seed(42)

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 从训练集中选择10%的样本作为训练集，剩余的作为验证集
train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.9)
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

# 创建训练集、验证集和测试集的数据加载器
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 定义孪生神经网络模型
# class SiameseNet(nn.Module):
#     def __init__(self):
#         super(SiameseNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
#         self.fc1 = nn.Linear(64*7*7, 1024)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(1024, 128)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x1, x2):
#         x1 = self.conv1(x1)
#         x1 = self.pool(x1)
#         x1 = torch.relu(x1)
#         x1 = self.conv2(x1)
#         x1 = self.pool(x1)
#         x1 = torch.relu(x1)
#         x1 = x1.view(-1, 64*7*7)
#         x1 = self.fc1(x1)
#         x1 = torch.relu(x1)
#         x1 = self.dropout(x1)
#         x1 = self.fc2(x1)

#         x2 = self.conv1(x2)
#         x2 = self.pool(x2)
#         x2 = torch.relu(x2)
#         x2 = self.conv2(x2)
#         x2 = self.pool(x2)
#         x2 = torch.relu(x2)
#         x2 = x2.view(-1, 64*7*7)
#         x2 = self.fc1(x2)
#         x2 = torch.relu(x2)
#         x2 = self.dropout(x2)
#         x2 = self.fc2(x2)

#         # 计算两个输入的特征向量的欧氏距离
#         distance = torch.sqrt(torch.sum(torch.pow(x1 - x2, 2), dim=1))
#         similarity = self.sigmoid(distance)

#         return similarity


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  #默认padding = 0 步长为1   ((data_size-kernel_size)+2*padding)ride+1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256,1)

    def forward_once(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) #1x28x28 -> 32x24x24 ->32x12x12
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #32x12x12 -> 64x8x8 -> 64x4x4
        x = x.view(x.size(0), -1) #sizex1024
        x = F.relu(self.fc1(x)) #sizex256
        return x

    def forward(self, input1, input2):
        # 分别处理两个输入
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # 计算两个特征的L2距离
        distance = (output1 - output2)**2
        output = self.fc2(distance)
        return output

# 创建孪生神经网络实例
model = SiameseNet()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0

    model.train()
    for (inputs1, labels1), (inputs2, labels2) in zip(train_loader, train_loader):
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
        labels1, labels2 = labels1.to(device), labels2.to(device)

        optimizer.zero_grad()
        outputs = model(inputs1, inputs2)
        # print(outputs)
        
        labels = (labels1 == labels2).float()  # 正样本对应标签为1，负样本对应标签为0
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs1.size(0)
        predicted = (outputs > 0.5).float()
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100.0 * train_correct / len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0

        for (inputs1, labels1), (inputs2, labels2) in zip(val_loader, val_loader):
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            labels1, labels2 = labels1.to(device), labels2.to(device)

            outputs = model(inputs1, inputs2)
            labels = (labels1 == labels2).float()  # 正样本对应标签为1，负样本对应标签为0
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs1.size(0)
            predicted = (outputs > 0.5).float()
            val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100.0 * val_correct / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# 在测试集上评估模型
model.eval()
test_correct = 0

with torch.no_grad():
    for (inputs1, labels1), (inputs2, labels2) in zip(test_loader, test_loader):
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
        labels1, labels2 = labels1.to(device), labels2.to(device)

        outputs = model(inputs1, inputs2)
        labels = (labels1 == labels2).float()  # 正样本对应标签为1，负样本对应标签为0
        predicted = (outputs > 0.5).float()

        test_correct += (predicted == labels).sum().item()

test_accuracy = 100.0 * test_correct / len(test_loader.dataset)
print(f"Test Accuracy: {test_accuracy:.2f}%")
