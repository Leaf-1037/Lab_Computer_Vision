import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 导入数据集
data = pd.read_csv('dataset.csv')

# 随机排序数据集
data = data.sample(frac=1).reset_index(drop=True)

# 将特征和标签分离
X = data[['data1', 'data2']]
y = data['label']

# 标签编码
print(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 定义模型
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))

# 编译模型
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 评估模型
_, accuracy = model.evaluate(X_test, y_test)
print('测试集准确率：', accuracy)
