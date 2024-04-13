import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def read_data(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")
    return np.asfarray(data, float)

def draw_curve(folder_path, f_names):
    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epochs')    # x轴标签
    plt.ylabel('loss')     # y轴标签
    
    color = ['g','b','r']
    lss = ['--','-.',':']
    marker_ = ['*','o','+']
    lbl = ['layer=10','layer=3']
    
    idx = 0

    for filename in f_names:
        file_path = os.path.join(folder_path, filename)
        y_loss = read_data(file_path)
        x_val = range(1, len(y_loss)+1)

        plt.plot(x_val, y_loss,label=lbl[idx],c=color[idx],ls=lss[idx],marker=marker_[idx], alpha=0.5)
        plt.legend(loc='center', bbox_to_anchor=(0.7, 0.85))
        
        idx = idx + 1
    
    plt.title('Loss Curve')
    plt.show()

if __name__ == '__main__':
    
    folder_path = "./loss_data/"
    file_names = ["Loss_batch_10000_layer_10.txt","loss.txt"]

    draw_curve(folder_path, file_names)