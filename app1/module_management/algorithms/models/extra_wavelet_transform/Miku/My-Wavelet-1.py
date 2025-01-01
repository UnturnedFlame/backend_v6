import numpy as np
import argparse
import pywt

# 专有小波变换处理算法
def wavelet_denoise(data, wavelet='db1', level=1):
    """
    小波变换降噪
    :param data: 原始数据
    :param wavelet: 使用的小波基
    :param level: 小波分解层数
    :return: 降噪后的数据
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    uthresh = sigma * np.sqrt(2 * np.log(len(data)))

    denoised_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]

    denoised_data = pywt.waverec(denoised_coeffs, wavelet)

    return denoised_data


if __name__ == '__main__':
    
    # 添加命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filepath', type=str, default='./input_data.npy')  # 输入数据
    parser.add_argument('--output-filepath', type=str, default='./output_data.npy')  # 输出结果

    # 解析命令行参数
    args = parser.parse_args()
    input_filepath = args.input_filepath
    output_filepath = args.output_filepath
    
    # 小波变换处理数据
    input_data = np.load(input_filepath)  # 原始信号序列
    raw_data = input_data.flatten()
    output_data = wavelet_denoise(raw_data) 
    # 保存处理结果
    np.save(output_filepath, output_data)