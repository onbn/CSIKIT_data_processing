import CSIKit.visualization.plot_scenario
from CSIKit.reader import IWLBeamformReader
from CSIKit.util import csitools
from CSIKit.visualization.graph import *
from CSIKit.visualization.metric import *
from CSIKit.visualization.plot_scenario import *
import torch
import numpy as np

def process_file(csi_file):
    #读取单个csv文件中的所有sci信息
    csi_list = []
    np.seterr(divide='ignore', invalid='ignore')  #解决无效值问题，如除以0
    my_reader =IWLBeamformReader()
    csi_data = my_reader.read_file(csi_file)
    for i in range(len(csi_data.frames)):
        if (csi_data.frames[i].csi_matrix[::, ::, ::].shape[2] > 1) & (len(csi_list)<2800) :
            csi_list.append(csi_data.frames[i].csi_matrix[::, ::, 0:2])
    csi_data = np.array(csi_list)
    csi_data = csi_data[..., 0] / csi_data[..., 1]
    # 检测并替换 inf 值
    csi_data[np.isinf(csi_data)] = 0
    # 检测并替换 nan 值
    csi_data[np.isnan(csi_data)] = 0
    # 振幅+相位
    csi_data_an = np.angle(csi_data)
    csi_data_abs = np.abs(csi_data)        #abs求振幅
    csi_data = np.concatenate((csi_data_abs, csi_data_an), axis=1)
    # csi_data = np.abs(csi_data_abs)      #只算振幅时
    return csi_data

