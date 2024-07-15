import math
import os
import pandas as pd
import torch
import numpy as np
from wifilib import *
import torch.nn.utils.rnn as rnn_utils
from standardization_fn import *
if __name__ == "__main__":
    # dataset
    # path
    total_folder_path = './Spinal_problems_data'
    data_list = []
    labels = []
    len_of_everyClass = []
    window_size = 400  # 窗口大小
    step_size = 200  # 滑动步长
    k = 0
    for folder_name in os.listdir(total_folder_path):
        folder_path = os.path.join(total_folder_path, folder_name)
        # 检查当前路径是否为目录，不是目录则跳过
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            # 跳过.DS_Store文件
            if filename == '.DS_Store':
                continue
            file_path = os.path.join(folder_path, filename)
            # 读取CSI数据，并转换为tensor
            tensor = torch.Tensor(process_file(file_path))
            n_samples = tensor.size(0)  # 数据的总长度
            # 400的滑动窗口，并且每次滑动200步来切割数据
            for start in range(0, n_samples - window_size + 1, step_size):
                end = start + window_size
                slice = tensor[start:end, :, :]
                slice = slice.permute(2, 0, 1)
                data_list.append(slice)
                print("Tensor size using shape attribute:", slice.size())
            # 未应用滑窗的数据分割
            # for i in range(0, tensor.size(0), 400):
            #     block = tensor[i:i + 400]  # 从第 i 个元素到第 i+500 个元素
            #     block = block.permute(2, 0, 1)
            #     print(block.shape)
            #     data_list.append(block)
                # print("Tensor size using shape attribute:", block.size())
            # len_of_everyClass.append(tensor.size(0)/400)
            len_of_everyClass.append(1 + math.floor((n_samples - window_size) / step_size))
            print(k,len_of_everyClass[k])
            # # choose label
            # # [0, 2]==[normal, kyphosis, scoliosis]
            for i in range(0,int(len_of_everyClass[k])):
                if folder_name.startswith('normal'):
                    labels.append(0)
                elif folder_name.startswith('kyphosis'):
                    labels.append(1)
                elif folder_name.startswith('scoliosis'):
                    labels.append(2)
            k = k + 1

    data = torch.stack(data_list, dim=0)
    # 全局标准化
    normalized_data = global_standardization(data)
    # 分通道标准化
    # normalized_data = channel_standardization(data)
    # label, one-hot vectors
    label_onehot = torch.zeros(size=(int(sum(len_of_everyClass)), 3))   #int(sum(len_of_everyClass))=750*6  labels=(750*6) label_onehot=(750*6)*3
    for i in range(0, len(labels)):
        label_onehot[i, labels[i]] = 1
    print(normalized_data.size(), labels, len_of_everyClass)
    torch.save(normalized_data, 'dataset/data_s_gs_angle.pt')
    torch.save(label_onehot, 'dataset/label_s_gs_angle.pt')



