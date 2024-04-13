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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 3x3卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 归一
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        #再卷一次
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        residual = x
        # 卷积操作
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.stride != 1 or residual.shape[1] != out.shape[1]:
            residual = nn.Conv2d(residual.shape[1], out.shape[1], kernel_size=1, stride=self.stride, bias=False)(residual)

        out += residual
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self.make_layer(64, 64, 2)
        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        # out = self.layer0(out)
        # out = self.layer0(out)
        # out = self.layer0(out)
        out = self.layer0(out)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    

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
    
    # 加载MNIST数据集并进行预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    # 创建ResNet模型并将其移动到设备上
    model = ResNet().to(device)

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
    