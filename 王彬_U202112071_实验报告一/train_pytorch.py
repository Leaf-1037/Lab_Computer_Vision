import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 超参数
batch_size = 32
learning_rate = 0.001
num_epochs = 10000
input_size = 2
hidden_size = 16
output_size = 4

from torch.utils.data import Dataset, DataLoader

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # print(self.data[idx])
        return self.data[idx], self.labels[idx]

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

if __name__=="__main__":
    # Load data
    data = pd.read_csv('dataset.csv')
    # shuffle
    data = data.sample(frac=1).reset_index(drop=True)

    # 将特征和标签分离
    X = data[['data1', 'data2']].values
    y = data['label'].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # 初始化模型和损失函数
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train_loss = []
    test_loss = []
    
    train_acc = []
    test_acc = []

    # 开始训练
    for epoch in range(num_epochs):
        running_loss = 0.0
        acc = 0.0
        model.train()
        for datum, labels in train_loader:
            # print(i.shape)
            # print(data.shape)
            # forward
            # inputs, labels = i
            outputs = model(datum.float())
            
            # compute loss
            optimizer.zero_grad()
            loss = criterion(outputs, labels.long())
            
            # backward
            loss.backward()
            
            # update weights
            optimizer.step()
            
            # sum up loss
            running_loss += loss.item()
        
        # 打印训练过程中的损失值
        if epoch % 100 == 99:
            print(f"Epoch {epoch+1}, Loss {running_loss/len(train_loader):.4f}")
            train_loss.append(float(running_loss/len(train_loader)))
            
            model.eval()
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs.float())
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
            
            accuracy = float(total_correct / total_samples)
            print(f"Test Accuracy: {accuracy:.4f}")
            test_acc.append(accuracy)
            
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")
    
    with open("./loss_data/loss.txt","w") as train_los:
        train_los.write(str(train_loss))
    with open("./loss_data/acc.txt","w") as test_ac:
        test_ac.write(str(test_acc))

