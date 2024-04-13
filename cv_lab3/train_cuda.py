import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# # 设置随机种子
# torch.manual_seed(2022)

# # 设置设备
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# # 数据预处理
# transform = transforms.Compose([
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# # 加载训练集和测试集
# train_dataset = datasets.MNIST(root='./data1', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data1', train=False, download=True, transform=transform)

# # 从训练集中随机选取10%作为本实验的训练集
# train_size = int(0.1 * len(train_dataset))
# train_subset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])

# # 从测试集中随机选取10%作为本实验的测试集
# test_size = int(0.1 * len(test_dataset))
# test_subset, _ = torch.utils.data.random_split(test_dataset, [test_size, len(test_dataset) - test_size])

# # 创建数据加载器
# train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
# print(train_loader)
# test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

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

# 将两张图片组合成一个样本
def combine_images(img1, img2):
    combined_img = torch.cat((img1, img2), dim=1)
    return combined_img

# 自定义数据集类
class CombinedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img1, target1 = self.dataset[index]
        img2, target2 = self.dataset[index]
        return combine_images(img1, img2), int(target1 == target2)

    def __len__(self):
        return len(self.dataset)

# 使用自定义的数据集类重新定义训练集和测试集
combined_train_dataset = CombinedMNISTDataset(train_dataset)
combined_test_dataset = CombinedMNISTDataset(test_dataset)

# 创建新的数据加载器
combined_train_loader = torch.utils.data.DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
combined_test_loader = torch.utils.data.DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=True)

# 打印训练集和测试集的样本数量
print("训练集样本数:", len(combined_train_dataset))
print("测试集样本数:", len(combined_test_dataset))

# 定义卷积神经网络模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = x1.view(x1.size(0), -1)  # 展平
        x1 = self.fc1(x1)

        x2 = self.conv1(x2)  # 共享参数的关键步骤
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.conv2(x2)  # 共享参数的关键步骤
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = x2.view(x2.size(0), -1)  # 展平
        x2 = self.fc1(x2)  # 共享参数的关键步骤

        x = torch.abs(x1 - x2)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# 创建模型实例并将其移动到设备上
model = ConvNet().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (data1, data2) in enumerate(train_loader):
        data1, data2 = data1.to(device), data2.to(device)
        
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = criterion(output, torch.ones_like(output))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    return train_loss / len(train_loader)

# 测试模型
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data1, data2) in enumerate(test_loader):
            data1, data2 = data1.to(device), data2.to(device)
            
            output = model(data1, data2)
            test_loss += criterion(output, torch.ones_like(output)).item()
            pred = (output >= 0.5).float()
            correct += pred.eq(torch.ones_like(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy

# 开始训练和测试
epochs = 10
for epoch in range(1, epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, criterion)
    test_loss, accuracy = test(model, device, test_loader, criterion)
    
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
