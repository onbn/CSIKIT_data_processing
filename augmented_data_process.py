from CSIKit.visualization.plot_scenario import *
import torch
import math
import os
import pandas as pd
import torch
import numpy as np
from wifilib import *
import torch.nn.utils.rnn as rnn_utils
from standardization_fn import *
from sklearn.model_selection import train_test_split
import gc  # 引入垃圾回收模块


def add_mask(data):               # 加掩码
    start = np.random.randint(0, data.shape[0] - 20)
    end = start + 20
    masked_data = np.copy(data)
    masked_data[start:end] = 0
    return masked_data

def add_noise(data):               # 加噪声
    scale_factor = np.random.uniform(0.8, 1.2)
    noise_data = (data * scale_factor)
    return noise_data

def data_window(data, label_t, window_size, step_size):              # 滑窗分割
    data_list = []
    labels = []
    len_of_labels = []
    k = 0
    for i in range(data.size(0)):
        for start in range(0, data.size(1) - window_size + 1, step_size):
            end = start + window_size
            slice = data[i, start:end, :, :]
            slice = slice.permute(2, 0, 1)
            data_list.append(slice)
        len_of_labels.append(1 + math.floor((data.size(1) - window_size) / step_size))
        for m in range(0, int(len_of_labels[k])):
            if label_t[i] == 0:
                labels.append(0)
            elif label_t[i] == 1:
                labels.append(1)
            elif label_t[i] == 2:
                labels.append(2)
        k = k + 1
    dataset = torch.stack(data_list, dim=0)
    label_onehot = torch.zeros(size=(
    int(sum(len_of_labels)), 3))  # int(sum(len_of_everyClass))=750*6  labels=(750*6) label_onehot=(750*6)*3
    for i in range(0, len(labels)):
        label_onehot[i, labels[i]] = 1
    return dataset, label_onehot


if __name__ == "__main__":
    # dataset
    # path
    total_folder_path = './spinal_dataset_n5'      #更换自己的文件路径
    data_list = []
    labels = []
    len_of_everyClass = []
    window_size = 400  # 窗口大小
    step_size = 200  # 滑动步长
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
            data_list.append(process_file(file_path))
            if folder_name.startswith('normal'):
                labels.append(0)
            elif folder_name.startswith('kyphosis'):
                labels.append(1)
            elif folder_name.startswith('scoliosis'):
                labels.append(2)
    data = np.stack(data_list, axis=0)
    # 全局标准化
    # 先滑窗后分割
    data = torch.Tensor(data)
    normalized_data = global_standardization(data)
    normalized_data_t, labels_t = data_window(normalized_data, labels, window_size, step_size)
    train_valid_data, test_data,  train_vallid_label, test_label = train_test_split(
         normalized_data_t, labels_t, test_size=0.2, random_state=42)
    train_data, valid_data, train_label, valid_label = train_test_split(
         train_valid_data, train_vallid_label, test_size=0.25, random_state=42)
   # 根据自己的需求存，.pt文件存储数据
    torch.save(valid_data, 'dataset/au_valid_data_n5_w4_onma_23.pt')
    torch.save(valid_label, 'dataset/au_valid_label_n5_w4_onma_23.pt')
    torch.save(test_data, 'dataset/au_test_data_n5_w4_onma_23.pt')
    torch.save(test_label, 'dataset/au_test_label_n5_w4_onma_23.pt')
    del valid_data, test_data, test_label, valid_label, data  # 删除不再需要的数据以释放内存
    gc.collect()  # 再次清理内存
    # 数据增强之加噪声、掩码
    train_data = np.array(train_data)
    train_data = np.transpose(train_data, (0, 2, 3, 1))
    train_data_mask_list = []
    # train_data_noise_list = []
    # train_data_flip_list = []
    for i in range(train_data.shape[0]):
        train_data_mask_list.append(add_mask(train_data[i]))
        # train_data_noise_list.append(add_noise(train_data[i]))
        # train_data_flip_list.append(np.flip(train_data[i], axis=0))
    #     gc.collect()  # 手动触发垃圾回收
    train_data_mask = np.stack(train_data_mask_list, axis=0)
    # train_data_noise = np.stack(train_data_noise_list, axis=0)
    # train_data_flip = np.stack(train_data_flip_list, axis=0)
    train_data = np.transpose(train_data, (0, 3, 1, 2))
    train_data_mask  = np.transpose(train_data_mask , (0, 3, 1, 2))
    # train_data_noise = np.transpose(train_data_noise, (0, 3, 1, 2))
    # train_data_flip = np.transpose(train_data_flip, (0, 3, 1, 2))
    # 转换数据类型
    train_data_t = torch.Tensor(train_data)
    train_data_mask_t = torch.Tensor(train_data_mask)
    # train_data_noise_t = torch.Tensor(train_data_noise)
    # train_data_flip_t = torch.Tensor(train_data_flip)
    # print('train_data_t:', train_data_t.size(), 'train_data_mask_t:', train_data_mask_t.size(), 'train_data_noise_t.size():', train_data_noise_t.size(),'train_data_flip_t.size():', train_data_flip_t.size() )
    # del train_data, train_data_mask, train_data_noise, train_data_flip  # 删除不再需要的数据以释放内存
    del train_data, train_data_mask  # 删除不再需要的数据以释放内存
    gc.collect()  # 再次清理内存
    # train_dataset = torch.cat((train_data_t, train_data_mask_t, train_data_noise_t, train_data_flip_t), dim=0)
    # del train_data_t, train_data_mask_t, train_data_noise_t, train_data_flip_t
    train_dataset = torch.cat((train_data_t, train_data_mask_t ), dim=0)
    del train_data_t, train_data_mask_t
    gc.collect()
    # train_label_onehot = torch.cat((train_label, train_label, train_label, train_label), dim=0)
    train_label_onehot = torch.cat((train_label, train_label), dim=0)
    del train_label
    gc.collect()
    # 生成随机打乱的索引
    indices = torch.randperm(train_dataset.size(0))
    # 使用打乱的索引重新排列数据集和标签集
    train_dataset_shuffled = train_dataset[indices]
    train_label_onehot_shuffled = train_label_onehot[indices]
    del train_dataset, train_label_onehot
    gc.collect()
    print('train_dataset:', train_dataset_shuffled.size(), '|train_label_onehot:', train_label_onehot_shuffled.size())
    # print('train_label_onehot', train_label_onehot_shuffled)
    # 保存训练集
    torch.save(train_dataset_shuffled, 'dataset/au_train_data_n5_w4_onma_23.pt')
    torch.save(train_label_onehot_shuffled, 'dataset/au_train_label_n5_w4_onma_23.pt')




