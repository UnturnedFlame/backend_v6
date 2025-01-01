import numpy as np
from scipy.interpolate import interp1d
import argparse
import random

def linear_interpolation_for_signal(data: np.ndarray):
    """
    对于一维信号的线性插值
    :param data: 需要进行插值的原始数据
    :return: array_filled, 插值后的数据
    """
    interpolated_data = data.copy()
    # 对于每一个数据向量，进行插值处理
    # for key in interpolated_data:
    #     if not key.startswith('__'):
    array = interpolated_data.flatten()  # 将数据展开成一维数组

    # 获取当前行的数据
    y = array
    x = np.arange(len(y))

    # 找到非NaN值和NaN值的索引
    nan_idx = np.isnan(y)
    not_nan_idx = ~nan_idx

    # 如果行中有NaN值且非NaN值数量大于等于2，才进行插值
    if nan_idx.any() and np.sum(not_nan_idx) >= 2:
        # 使用非NaN值进行线性插值
        interp_func = interp1d(x[not_nan_idx], y[not_nan_idx], kind='linear', fill_value="extrapolate")

        # 用插值结果替换NaN值
        y[nan_idx] = interp_func(x[nan_idx])
        array = y

    # 更新数据
    interpolated_data = array.reshape(1, -1)  # 确保数据仍然是一行
    
    return interpolated_data

if __name__ == '__main__':
    
    # 1.解析命令行参数
    parser = argparse.ArgumentParser(description='manual to this script')
    # 2.添加参数
    parser.add_argument('--raw-data-filepath', type=str, default = None)  # 原始数据的存放路径
    parser.add_argument('--interpolated-data-filepath', type=str, default = None)  # 插值后的数据的存放路径
    # 3.获取参数
    args = parser.parse_args()
    raw_data_filepath = args.raw_data_filepath  # 原始数据的存放路径
    interpolated_data_filepath = args.interpolated_data_filepath   # 插值后的数据的存放路径
    
    # 加载原始数据并进行插值
    raw_data = np.load(raw_data_filepath)
    interpolated_data = linear_interpolation_for_signal(raw_data)
    
    # 保存插值结果, 并打印出保存的文件路径
    results_filepath = np.save(interpolated_data_filepath, interpolated_data)
    
    print(f'{random.choice(range(100))}#', end='')
    