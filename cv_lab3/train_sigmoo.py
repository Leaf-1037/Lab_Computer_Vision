import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# 加载MNIST数据集，仅使用10%的样本
trainset = MNIST(root='./data', train=True, download=True)#, transform=ToTensor())
trainset = torch.utils.data.Subset(trainset, list(range(6000)))
testset = MNIST(root='./data', train=False, download=True)#, transform=ToTensor())
testset = torch.utils.data.Subset(testset, list(range(1000)))

# 创建包含每个数字类别图像的索引列表
train_indices = [torch.where(trainset.dataset.targets == i)[0] for i in range(10)]
test_indices = [torch.where(testset.dataset.targets == i)[0] for i in range(10)]

# 将每个数字类别的图像分组成一对同类或异类输入
train_pairs, train_labels = [], []
for i in range(10):
    for j in range(i+1, 10):
        idx_i = train_indices[i]
        idx_j = train_indices[j]
        img_i = trainset.dataset.data[idx_i][0]
        img_j = trainset.dataset.data[idx_j][0]
        train_pairs.append([img_i, img_j])
        train_labels.append(1)
        img_i = trainset.dataset.data[idx_i][1]
        img_j = trainset.dataset.data[idx_j][1]
        train_pairs.append([img_i, img_j])
        train_labels.append(0)

test_pairs, test_labels = [], []
for i in range(10):
    for j in range(i+1, 10):
        idx_i = test_indices[i]
        idx_j = test_indices[j]
        img_i = testset.dataset.data[idx_i][0]
        img_j = testset.dataset.data[idx_j][0]
        test_pairs.append([img_i, img_j])
        test_labels.append(1)
        img_i = testset.dataset.data[idx_i][1]
        img_j = testset.dataset.data[idx_j][1]
        test_pairs.append([img_i, img_j])
        test_labels.append(0)

# 定义数据集和数据加载器
class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        pair = self.transform(self.pairs[index])
        label = self.labels[index]
        return pair, label
    
    def __len__(self):
        return len(self.pairs)

train_dataset = SiameseDataset(train_pairs, train_labels, transform=ToTensor())
test_dataset = SiameseDataset(test_pairs, test_labels, transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 定义模型
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
        x = torch.abs(x1 - x2)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

model = SiameseNet()
print(model)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
model.to(device)

for epoch in range(10):
    running_loss = 0.0
    print(train_loader)
    print(enumerate(train_loader, 0))
    for i, data in enumerate(train_loader, 0):
        # print(data)
        inputs, labels = data
        input1, input2 = inputs[:, 0], inputs[:, 1]
        input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input1, input2)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d, loss: %.3f' % (epoch+1, running_loss/len(train_loader)))

# 在测试集上评估模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        input1, input2 = inputs[:, 0], inputs[:, 1]
        input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
        outputs = model(input1, input2)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.unsqueeze(1)).sum().item()

print('Test accuracy: %.3f' % (correct / total))
