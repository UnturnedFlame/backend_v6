import numpy as np
from scipy.interpolate import interp1d
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