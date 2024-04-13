./Results 文件夹中为各个测试数据结果，其中以acc开头的是准确率测试结果，以loss开头的是损失值测试结果。
每个文件名后面的即注释，使用了多少层残差块，使用了什么激活函数。

特别地，文件“acc_for_each_classification.txt”为对于不同label的测试结果。

代码文件见./train_cuda.py和./matplot.py