import torch

#全局
def global_standardization(data):
    # tensor维度为(2250, 3, 400, 30)
    mean = data.mean()  # 计算全局均值
    std = data.std()  # 计算全局标准差
    # 进行标准化
    normalized_data = (data - mean) / std
    return normalized_data

#通道
def channel_standardization(data):
    normalized_data = torch.zeros_like(data)
    # 对每个通道进行标准化
    for i in range(data.shape[1]):  # 遍历通道
        channel = data[:, i, :, :]
        mean = channel.mean()  # 计算均值
        std = channel.std()  # 计算标准差
        normalized_data[:, i, :, :] = (channel - mean) / std  # 标准化
    return normalized_data