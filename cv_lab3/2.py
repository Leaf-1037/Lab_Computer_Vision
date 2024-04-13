import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

Epoch_size = 10
lr = 0.001

loss_log = []
acc_log = []
class_correct_log = [[] for i in range(10)]

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

#定义孪生网络
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
    

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            loss_log.append(loss.item())

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = [0 for i in range(10)]
    class_total = [0 for i in range(10)]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            label = np.array(target.cpu()).tolist()
            prediction = np.array(pred.cpu()).T.tolist()[0]

            for i in range(len(label)):
                class_total[label[i]] += 1
                if prediction[i] == label[i]:
                    # acc += 1
                    class_correct[prediction[i]] += 1

    for i in range(10):
        class_correct[i] /= class_total[i]
        class_correct_log[i].append(class_correct[i])
        
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    acc_log.append(accuracy)
    
    
def main():
    # 设置随机数种子和设备
    torch.manual_seed(42)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Cuda_available {}".format(torch.cuda.is_available()))
    
    
    # 加载MNIST数据集，仅使用10%的样本
    trainset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
    trainset = torch.utils.data.Subset(trainset, list(range(6000)))
    testset = MNIST(root='./data', train=False, download=True, transform=ToTensor())
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
    
    # 加载MNIST数据集并进行预处理
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    # train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60, shuffle=True, num_workers=2)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    train_dataset = SiameseDataset(train_pairs, train_labels, transform=ToTensor())
    test_dataset = SiameseDataset(test_pairs, test_labels, transform=ToTensor())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    # 创建ResNet模型并将其移动到设备上
    model = SiameseNet().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 训练和测试模型
    for epoch in range(1, Epoch_size+1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        test(model, device, test_loader)
    
    print(loss_log)
    print(acc_log)
    # 打印每个类别的正确率演化
    # print()
    # for i in range(10):
    #     print(class_correct_log[i])

if __name__ == '__main__':
    main()
    