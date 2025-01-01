import os
import random
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import joblib
import torch
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from scipy.interpolate import PchipInterpolator, interp1d, lagrange, CubicSpline
from scipy.io import savemat
from torch import nn
import subprocess

from app1.module_management.algorithms.functions.feature_extraction import time_domain_extraction, \
    frequency_domain_extraction

# 预处理部分的结果输出的目录
output_root_dir = r'app1/module_management/algorithms/functions/preprocessing_results'


# 小波变换降噪
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


# 对信号应用小波变换的处理方法
def wavelet_transform_processing(raw_data, wavelet='db1', level=1, user_dir=None, extra_algorithm=None,
                                 multiple_sensor=False):
    """
    对信号应用小波变换的处理方法
    :param multiple_sensor: 是否为多传感器数据
    :param extra_algorithm: 如果不为None，则调用用户上传的专有小波变换处理的算法
    :param user_dir: 保存结果的用户目录
    :param level: 小波的层数
    :param wavelet: 使用的小波类型
    :param raw_data: 原始数据
    :return: 降噪后的数据的存放路径以及降噪后结果图像的存放路径（以字典形式返回）, multiple_sensor=True为多传感器
    """
    # print(raw_data)
    # print(raw_data.shape)
    # 图片存放路径
    # save_path_dir = os.path.join(r'app1/module_management/algorithms/functions/'
    #                              r'preprocessing_results/wavelet_trans/single_signal/', filename)

    save_path_dir = output_root_dir + '/' + 'wavelet_trans/' + user_dir

    username = user_dir.split('/')[0]  # 用户名
    if extra_algorithm is not None:
        extra_algorithm_dir = os.path.dirname(extra_algorithm)  # 存放增值服务小波变换组件算法的目录
        extra_algorithm_user_dir = extra_algorithm_dir + '/' + username  # 不同用户调用增值组件时，每个用户都有自己的文件夹存放数据
        if not os.path.exists(extra_algorithm_user_dir):
            os.makedirs(extra_algorithm_user_dir)
    # else:
    #     print('无效的用户目录')
    #     # figure_path 作为报错信息
    #     return {'transformed_data': None, 'figure_path': '无效的用户目录'}, False
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    matplotlib.use('Agg')
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
    plt.rcParams['font.size'] = 12

    # 判断该数据是否为单传感器数据
    try:
        if not multiple_sensor:
            # 单传感器的数据
            # multiple_sensor = False
            if extra_algorithm is None:
                # 调用小波降噪
                transformed_data = wavelet_denoise(raw_data.flatten(), wavelet, int(level))
            else:
                print(f'extra algorithm {extra_algorithm}')
                # 调用用户上传的专有小波变换算法
                input_filepath = extra_algorithm_user_dir + '/input_data.npy'  # 以文件形式向专有小波变换算法传递输入数据
                np.save(input_filepath, raw_data.flatten())
                extra_algorithm_filename = extra_algorithm.split('/')[-1]
                # 以shell命令运行专有小波变换算法脚本
                result = subprocess.run(shell=True, capture_output=True,
                                        args=f"cd {extra_algorithm_dir} & python ./{extra_algorithm_filename} "
                                             f"--input-filepath ./{username}/input_data.npy --output-filepath ./{username}/output_data.npy")
                error = result.stderr.decode('utf-8')
                if error:
                    print(f'专有小波变换算法运行出错: {error}')
                    return {'transformed_data': None, 'figure_path': error}, False
                else:
                    output_filepath = extra_algorithm_user_dir + '/output_data.npy'
                    if os.path.exists(output_filepath):
                        transformed_data = np.load(output_filepath)  # 读取专有小波变换算法处理的结果数据
                    else:
                        return {'transformed_data': None, 'figure_path': '专有小波变换算法输出结果路径无效'}

                # output_filepath = extra_algorithm_dir + '/output_data.npy'

            # 绘制处理结果数据的图像
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(raw_data.flatten())
            plt.title(f'原始数据')
            plt.subplot(2, 1, 2)
            plt.plot(transformed_data.flatten())
            plt.title(f'去噪后数据')
            plt.tight_layout()
            # 保存小波变换处理结果以及图像
            save_path = save_path_dir + '/' + 'result.png'
            data_save_path = save_path_dir + '/' + 'transformed_data.npy'
            np.save(data_save_path, transformed_data)
            plt.savefig(save_path)
            # print(transformed_data.shape)
            results = ({'figure_path': (save_path,), 'transformed_data': transformed_data,
                        'data_save_path': data_save_path}, multiple_sensor)

            return results
        # 多传感器数据
        else:
            # multiple_sensor = True
            shape = raw_data.shape
            # 确保多传感器的数据形状为2048*N, N为传感器数量
            if shape[0] < shape[1]:
                sensor_num = shape[0]
                raw_data = raw_data.T
            else:
                sensor_num = shape[1]
            # raw_data = raw_data.reshape(2048, -1)
            # print(raw_data.shape)
            # sensor_num = raw_data.shape[1]
            transformed_datas = []
            all_figure_paths = []

            for i in range(sensor_num):
                input_data = raw_data[:, i].flatten()
                if extra_algorithm is None:
                    # 小波变换降噪
                    transformed_data = wavelet_denoise(input_data, wavelet, int(level))
                else:
                    print(f'extra algorithm {extra_algorithm}')
                    extra_algorithm_filename = extra_algorithm.split('/')[-1]
                    # 调用用户上传的专有小波变换算法
                    input_filepath = extra_algorithm_user_dir + '/input_data.npy'  # 以文件形式向专有小波变换算法传递输入数据
                    np.save(input_filepath, input_data)
                    # 以shell命令运行专有小波变换算法脚本
                    result = subprocess.run(shell=True, capture_output=True,
                                            args=f"cd {extra_algorithm_dir} & python ./{extra_algorithm_filename} "
                                                 f"--input-filepath ./{username}/input_data.npy --output-filepath ./{username}/output_data.npy")
                    error = result.stderr.decode('utf-8')
                    if error:
                        print(f'专有小波变换算法运行出错: {error}')
                        return {'transformed_data': None, 'figure_path': error}, False
                    else:
                        output_filepath = extra_algorithm_dir + '/output_data.npy'
                        if os.path.exists(output_filepath):
                            transformed_data = np.load(output_filepath)  # 读取专有小波变换算法处理的结果数据
                        else:
                            return {'transformed_data': None, 'figure_path': '专有小波变换算法输出结果路径无效'}
                transformed_datas.append(transformed_data)

                # 绘制单个传感器小波降噪之后的结果图像
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.plot(input_data.flatten())
                plt.title(f'原始数据')

                plt.subplot(2, 1, 2)
                plt.plot(transformed_data.flatten())
                plt.title(f'去噪后数据')

                plt.tight_layout()
                save_path = save_path_dir + '/' + 'result_sensor_' + str(i) + '.png'

                plt.savefig(save_path)
                all_figure_paths.append(save_path)

            # 保存结果
            results = np.array(transformed_datas)
            data_save_path = save_path_dir + '/' + 'transformed_data.npy'
            # print(f'data_save_path_1: {data_save_path}')
            np.save(data_save_path, results.T)
            # print('results: ', results)
            return ({'transformed_data': results.T, 'figure_path': all_figure_paths, 'data_save_path': data_save_path},
                    multiple_sensor)
    except Exception as e:
        return {'transformed_data': None, 'figure_path': e}, False


# 用户私有插值算法
def private_interpolation(raw_data: np.ndarray, save_path, private_algorithm, user_dir):
    """

    :param raw_data: 原始数据
    :param save_path: 插值结果图像保存路径
    :param private_algorithm: 用户私有算法路径
    :param user_dir: 用户目录
    :return:
    """
    # 存放私用插值算法的文件目录
    # base_dir_of_algorithm = 'app1/module_management/algorithms/models/private_interpolation'
    base_dir_of_algorithm = private_algorithm  # 增值服务组件算法源文件的存放路径
    algorithm_name = base_dir_of_algorithm.split('/')[-1]  # 增值服务组件的算法名
    # 存放用户私有的插值算法的文件路径
    user_name = user_dir.split('/')[0]  # 调用该增值服务组件算法的用户名
    # private_algorithm_dir = base_dir_of_algorithm + '/' + user_name  # 上传私有插值算法的用户目录

    private_algorithm_dir = os.path.dirname(base_dir_of_algorithm)  # 该增值服务组件算法所在的目录
    print(f'private_algorithm: {private_algorithm_dir}')
    # 将数组形式的原始数据raw_data转换为字符串类型以作为运行私有算法时的命令行参数
    # list_data = raw_data.tolist()
    # str_data = "np.ndarray({})".format(list_data)
    # print(f'private_algorithm_dir: {private_algorithm_dir},\n private_algorithm: {private_algorithm}')

    # 向增值服务组件算法传递数据，以文件的形式保存数据和读取数据
    intermediate_data_dir = private_algorithm_dir + '/' + user_name  # 调用该增值服务组件算法的用户目录
    print(f'intermediate_data_dir: {intermediate_data_dir}')
    if not os.path.exists(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)
    input_data_path = intermediate_data_dir + '/input_data.npy'
    np.save(input_data_path, raw_data)

    interpolated_data_filepath = intermediate_data_dir + '/interpolated_data.npy'  # 存放私有插值算法插值结果的路径
    print(f'interpolated_data_filepath: {interpolated_data_filepath}')

    # interpolated_data_filepath = './interpolated_data.npy'  # 存放私有插值算法插值结果的路径

    # 以子进程的形式运行相应的私有算法的python脚本
    # result = subprocess.run(shell=True, capture_output=True,
    #                         args=f"cd {private_algorithm_dir} & python {private_algorithm}.py "
    #                         f"--raw-data-filepath=\"{intermediate_data_path}\"")
    # 在调用增值服务组件算法时，首先切换到该算法所在的目录下，再运行算法脚本
    result = subprocess.run(shell=True, capture_output=True,
                            args=f"cd {private_algorithm_dir} & python {algorithm_name} --raw-data-filepath "
                                 f"./{user_name}/input_data.npy --interpolated-data-filepath ./{user_name}/interpolated_data.npy")
    print(f'result: {result}')
    error = result.stderr.decode('utf-8')
    if error:
        print(f'增值服务插值算法运行出错: {error}')
        return {'interpolated_data': None, 'figure_path': error}, False
    interpolated_data = np.load(interpolated_data_filepath)
    # 将字符串类型的输出结果解析为numpy数组
    # interpolated_data: np.ndarray = eval(result.stdout)

    # 绘制结果图像

    plot_interpolation(raw_data, interpolated_data, save_path)

    return interpolated_data, save_path


def private_feature_selection():
    pass


# 对信号的插值，包含多传感器信号与单传感器的信号
def interpolation_for_signals(raw_data: np.ndarray, algorithm, filename, user_dir=None, private_algorithm=None,
                              multiple_sensor=False):
    """
    对信号的插值算法，包含多传感器信号与单传感器的信号
    :param private_algorithm: 如果不为None，则调用用户上传的私有算法
    :param user_dir: 插值处理结果的保存目录
    :param algorithm: 使用的插值算法
    :param raw_data: 原始数据
    :param filename: 原始数据的来源文件
    :param multiple_sensor: 是否为多传感器的数据
    :return: 插值后的数据以及插值结果图像的存放路径, multiple_sensor=True时为多传感器数据
    """
    # print('raw_data: ', raw_data.shape)
    # 创建保存插值处理结果的多级目录
    save_path_dir = output_root_dir + '/' + algorithm
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    save_path_dir = save_path_dir + '/' + user_dir
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    # 判断是否为多传感器的数据，此时为单传感器信号
    try:
        if not multiple_sensor:
            # multiple_sensor = False
            # 插值结果的图像保存路径
            save_path = save_path_dir + '/' + f'{filename}_result.png'
            if algorithm == 'neighboring_values_interpolation':
                # 邻近值插值算法
                data_interpolated, figure_path = neighboring_values_interpolation_for_signal(raw_data, save_path)
                # print('data_interpolated: ', data_interpolated.shape)
            elif algorithm == 'polynomial_interpolation':
                # 多项式插值算法
                data_interpolated, figure_path = polynomial_interpolation_for_signal(raw_data, save_path)
                # print('data_interpolated: ', data_interpolated.shape)
            elif algorithm == 'bicubic_interpolation':
                # 三次样条插值算法
                data_interpolated, figure_path = bicubic_interpolation_for_signal(raw_data, save_path)
            elif algorithm == 'lagrange_interpolation':
                # 拉格朗日插值算法
                data_interpolated, figure_path = lagrange_interpolation_for_signal(raw_data, save_path)
            elif algorithm == 'newton_interpolation':
                # 牛顿插值算法
                data_interpolated, figure_path = newton_interpolation_for_signal(raw_data, save_path)
            elif algorithm == 'linear_interpolation':
                # 线性插值算法
                data_interpolated, figure_path = linear_interpolation_for_signal(raw_data, save_path)
            elif algorithm == 'deeplearning_interpolation':
                # 深度学习的插值算法
                data_interpolated, figure_path = deeplearning_interpolation(raw_data, save_path)
            else:
                # 用户私有的插值算法
                data_interpolated, figure_path = private_interpolation(raw_data, save_path, private_algorithm, user_dir)
            data_save_path = save_path_dir + '/' + 'data_interpolated.mat'
            savemat(data_save_path, {'data_interpolated': data_interpolated})
            return {'interpolated_data': data_save_path, 'figure_paths': [figure_path]}, multiple_sensor
        else:
            # multiple_sensor = True
            shape = raw_data.shape
            # 如果是多传感器的信号，确保数据形状为2048*N
            if shape[0] < shape[1]:
                raw_data = raw_data.T

            sensor_num = raw_data.shape[1]
            all_data_interpolated = []
            all_figure_paths = []
            # 根据选择算法，对来自每一个传感器的长度为2048的信号进行插值
            if algorithm == 'neighboring_values_interpolation':
                # 邻近值插值算法
                for i in range(sensor_num):
                    save_path = save_path_dir + '/sensor_' + str(i + 1) + '.png'
                    data_interpolated, figure_path = neighboring_values_interpolation_for_signal(
                        raw_data[:, i].flatten(), save_path)
                    all_data_interpolated.append(data_interpolated)
                    all_figure_paths.append(figure_path)
            elif algorithm == 'polynomial_interpolation':
                for i in range(sensor_num):
                    save_path = save_path_dir + '/sensor_' + str(i + 1) + '.png'
                    data_interpolated, figure_path = polynomial_interpolation_for_signal(raw_data[:, i].flatten(),
                                                                                         save_path)
                    all_data_interpolated.append(data_interpolated)
                    all_figure_paths.append(figure_path)
            elif algorithm == 'bicubic_interpolation':
                # 三次样条插值算法
                for i in range(sensor_num):
                    save_path = save_path_dir + '/sensor_' + str(i + 1) + '.png'
                    data_interpolated, figure_path = bicubic_interpolation_for_signal(raw_data[:, i].flatten(),
                                                                                      save_path)
                    all_data_interpolated.append(data_interpolated)
                    all_figure_paths.append(figure_path)
            elif algorithm == 'lagrange_interpolation':
                # 拉格朗日插值算法
                for i in range(sensor_num):
                    save_path = save_path_dir + '/sensor_' + str(i + 1) + '.png'
                    data_interpolated, figure_path = lagrange_interpolation_for_signal(
                        raw_data[:, i].flatten().reshape(1, -1), save_path)
                    all_data_interpolated.append(data_interpolated)
                    all_figure_paths.append(figure_path)
            elif algorithm == 'newton_interpolation':
                # 牛顿插值算法
                for i in range(sensor_num):
                    save_path = save_path_dir + '/sensor_' + str(i + 1) + '.png'
                    data_interpolated, figure_path = newton_interpolation_for_signal(
                        raw_data[:, i].flatten().reshape(1, -1), save_path)
                    all_data_interpolated.append(data_interpolated)
                    all_figure_paths.append(figure_path)
            elif algorithm == 'linear_interpolation':
                # 线性插值算法
                for i in range(sensor_num):
                    save_path = save_path_dir + '/sensor_' + str(i + 1) + '.png'
                    data_interpolated, figure_path = linear_interpolation_for_signal(
                        raw_data[:, i].flatten().reshape(1, -1), save_path)
                    all_data_interpolated.append(data_interpolated)
                    all_figure_paths.append(figure_path)
            elif algorithm == 'deeplearning_interpolation':
                # 深度学习的插值算法
                for i in range(sensor_num):
                    save_path = save_path_dir + '/sensor_' + str(i + 1) + '.png'
                    data_interpolated, figure_path = deeplearning_interpolation(raw_data[:, i].flatten(), save_path)
                    all_data_interpolated.append(data_interpolated)
                    all_figure_paths.append(figure_path)
            else:
                # 用户私有的多传感器插值算法
                for i in range(sensor_num):
                    save_path = save_path_dir + '/sensor_' + str(i + 1) + '.png'
                    data_interpolated, figure_path = private_interpolation(
                        raw_data=raw_data[:, i].flatten().reshape(1, 2048), save_path=save_path,
                        private_algorithm=private_algorithm, user_dir=user_dir)
                    all_data_interpolated.append(data_interpolated)
                    all_figure_paths.append(figure_path)
            results = np.array(all_data_interpolated)
            data_save_path = save_path_dir + '/' + 'data_interpolated.mat'
            savemat(data_save_path, {'data_interpolated': results.T})
            # print('interpolation results: ', results.shape)
            return {'interpolated_data': data_save_path, 'figure_paths': all_figure_paths}, multiple_sensor
    except Exception as e:
        print('插值处理模块异常..........')
        return {'interpolated_data': None, 'figure_paths': e}, False


# 针对四个阶段时序数据的小波变换
def wavelet_denoise_four_stages(mat_data, filename):
    stages = ['stage_1', 'stage_2', 'stage_3', 'stage_4']
    dir_path = os.path.join(r'app1/module_management/algorithms/functions/'
                            r'preprocessing_results/wavelet_trans/four_stages/', filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    results = {'all_save_paths': {}, 'denoised_datas': {}}

    # 对四个阶段的数据分别进行小波降噪
    for stage in stages:
        save_path = os.path.join(dir_path, stage + '.png')
        results['all_save_paths'][stage] = save_path
        example_data = mat_data.get(stage)
        if example_data is not None:
            example_data = example_data.flatten()
        denoised_data = wavelet_denoise(example_data)
        results['denoised_datas'][stage] = denoised_data
        # 绘制小波降噪后的图像
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(example_data)
        plt.title(f'原始 {stage} 数据')

        plt.subplot(2, 1, 2)
        plt.plot(denoised_data)
        plt.title(f'去噪后 {stage} 数据')

        plt.tight_layout()
        plt.savefig(save_path)

    return results


# 获取文件类型
def get_filetype(datafile):
    if datafile is not None and len(datafile) > 0:
        return os.path.basename(datafile).split('.')[1]


# 制造有缺失值的数据，测试阶段使用
# def make_data(data: np.ndarray):
#     while (True):
#         missing_value_start = random.choice(range(data.shape[1]))
#         if missing_value_start < data.shape[1] - 10:
#             break
#     missing_value_end = missing_value_start + 10
#
#     new_data = data.copy()
#     new_data[0, missing_value_start: missing_value_end] = np.nan
#
#     return new_data, missing_value_start, missing_value_end


# 绘制插值处理结果图像（对信号插值）
def plot_interpolation(raw_data: np.ndarray, interpolated_data: np.ndarray, save_path):
    """
    绘制插值处理后的结果图像
    :param raw_data: 原始数据
    :param interpolated_data: 插补后数据
    :param save_path: 保存插补后数据的路径
    :return: 无返回值
    """
    if len(raw_data.shape) == 2:
        raw_data = raw_data[0, :]
    if len(interpolated_data.shape) == 2:
        interpolated_data = interpolated_data[0, :]

    print("插值处理结果图像绘制................")
    print(raw_data.shape)
    print(interpolated_data.shape)
    matplotlib.use('Agg')
    # 设置全局字体属性，这里以SimHei为例
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['font.size'] = 30

    # 设置label字体大小
    font = FontProperties(size=25)

    # 图例展示区间
    # 使用numpy.isnan找到缺失值的布尔数组
    is_nan = np.isnan(raw_data)

    # 找到第一个缺失值的下标
    start_missing = np.where(is_nan)[0][0] if np.any(is_nan) else None

    # 找到最后一个缺失值的下标
    end_missing = np.where(is_nan)[0][-1] if np.any(is_nan) else None
    if start_missing and end_missing:
        start = start_missing - 50
        end = end_missing + 50
    else:
        start = 0
        end = len(raw_data)
    print(f"start_missing: {start_missing}, end_missing: {end_missing}")
    print(f"start: {start}, end: {end}")

    raw_data_display = raw_data[start:end]
    interpolated_data_display = interpolated_data[start:end]

    plt.figure(figsize=(20, 10))
    # 绘制测试样本散点图
    plt.scatter(np.array(range(start, end)), raw_data_display, c='blue', marker='o', label='原始信号样本')

    # 绘制模拟样本散点图，跳过缺失值
    valid_indices = ~np.isnan(interpolated_data_display)
    plt.scatter(np.array(range(start, end))[valid_indices], interpolated_data_display[valid_indices], c='red',
                marker='x',
                label='插补后信号样本')
    print(f"interpolated_data_display: {interpolated_data_display.shape}")

    # 标注缺失值的区间
    # plt.axvspan(start_missing, end_missing, alpha=0.3, color='yellow', label='缺失值')

    # 添加图例
    plt.legend(prop=font)

    # 设置x轴和y轴的标签
    plt.xlabel('时间点')
    plt.ylabel('采样值')

    # 显示图形
    plt.savefig(save_path)


# 邻近值插补
def neighboring_values_interpolation_for_signal(data: np.ndarray, save_path):
    """
    邻近值插值算法
    :param data: 需要进行插值的原始数据
    :param save_path: 结果图像的保存路径
    :return: （array_filled， figure_path），插值后的数据以及显示插值结果的图片的存放路径
    """
    data = data.flatten()
    nan_indices = np.isnan(data)

    # 使用前向临近值插补
    array_filled_forward = data.copy()
    for i in range(1, len(data)):
        if np.isnan(array_filled_forward[i]):
            array_filled_forward[i] = array_filled_forward[i - 1]  # 用前一个值填补

    # 使用后向临近值插补
    array_filled_backward = data.copy()
    for i in range(len(data) - 2, -1, -1):
        if np.isnan(array_filled_backward[i]):
            array_filled_backward[i] = array_filled_backward[i + 1]  # 用后一个值填补

    # 将前向插补和后向插补的结果结合起来
    array_filled = np.where(np.isnan(array_filled_forward), array_filled_backward, array_filled_forward)

    # 指定保存路径
    # save_path = os.path.join(output_root_dir, 'neighboring_values_interpolation', filename + '_interpolated.mat')
    # figure_path = os.path.join(output_root_dir, 'neighboring_values_interpolation', filename + '_interpolated.png')
    plot_interpolation(data, array_filled, save_path)

    # 保存插补后的MAT文件
    # savemat(save_path, {'data': array_filled}, format='4')

    return array_filled, save_path


# 多项式插值算法的关键代码
def fill_missing_with_pchip(column, index):
    """
    :param column: 插值算法应用的列
    :param index: 对应的下标
    :return: 对原始列插值后的列数据
    """
    # 仅对数值型数据应用np.isfinite
    if pd.api.types.is_numeric_dtype(column):
        finite_values = column[np.isfinite(column)]
        if len(finite_values) > 1:  # 确保至少有两个有限数值以进行插值
            # 创建PCHIP插值器
            pchip = PchipInterpolator(index[np.isfinite(column)], finite_values)
            # 填充缺失值
            return pchip(index)
        else:
            print("Not enough valid data points to interpolate.")
            return column  # 返回原始列，因为没有足够的数据进行插值
    else:
        print("Column contains non-numeric data, skipping interpolation.")
        return column  # 返回原始列，因为包含非数值数据


# 多项式插补算法
def polynomial_interpolation(input_file):
    """
    多项式插值算法
    :param input_file: 输入数据文件路径
    :return: 插值后数据的存放路径
    """
    # 读取Excel文件
    df = pd.read_excel(input_file)
    # 对第二列之后的每一列进行插补
    for column_name in df.columns[1:]:  # 从第二列开始
        df[column_name] = fill_missing_with_pchip(df[column_name], df.index)

    basename = os.path.basename(input_file)
    filename = basename.split('.')[0]
    save_dir = os.path.join(output_root_dir, 'linear_interpolation', filename)
    save_raw_dir, save_result_dir = os.path.join(save_dir, 'raw_'), os.path.join(save_dir, 'result_')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_filename = os.path.join(save_dir, basename)

    # 将插补后的数据保存到新的Excel文件
    df.to_excel(output_filename, index=False)
    # 绘制波形图并保存
    waveform_save_path = {'原始数据': waveform_drawing(input_file, save_raw_dir),
                          '结果数据': waveform_drawing(output_filename, save_result_dir)}

    return waveform_save_path, output_filename


# 对信号的多项式插值
def polynomial_interpolation_for_signal(data: np.ndarray, figure_save_path):
    """
    对信号的多项式插值
    :param data: 需要进行插值的原始数据
    :param figure_save_path: 插值结果图像的保存路径
    :return: （array_filled， figure_path），插值后的数据以及显示插值结果的图片的存放路径
    """
    data = data.flatten()
    nan_indices = np.isnan(data)
    non_nan_indices = ~nan_indices

    x_non_nan = np.arange(len(data))[non_nan_indices]
    y_non_nan = data[non_nan_indices]

    interpolator = interp1d(x_non_nan, y_non_nan, kind='linear', fill_value='extrapolate')
    array_filled = data.copy()
    array_filled[nan_indices] = interpolator(np.arange(len(data))[nan_indices])

    # save_path = os.path.join(output_root_dir, 'polynomial_interpolation', filename + '_interpolated.mat')
    # figure_path = os.path.join(output_root_dir, 'polynomial_interpolation', filename + '_interpolated.png')

    # savemat(save_path, {'data': array_filled}, format='4')
    plot_interpolation(data, array_filled, figure_save_path)

    return array_filled, figure_save_path


# 对信号的拉格朗日插值
def lagrange_interpolation_for_signal(data: np.ndarray, figure_save_path):
    """
    对信号的拉格朗日插值算法
    :param data: 需要进行插值的原始数据
    :param figure_save_path: 结果图像保存路径
    :return: （array_filled， figure_path），插值后的数据以及显示插值结果的图片的存放路径
    """

    # 定义插值函数
    def lagrange_insert(arr):
        n = len(arr)
        for i in range(n):
            if np.isnan(arr[i]):
                left, right = i - 1, i + 1
                while np.isnan(arr[left]) and left >= 0:
                    left -= 1
                while np.isnan(arr[right]) and right < n:
                    right += 1
                if left < 0 or right >= n:
                    continue
                arr[i] = lagrange([left, right], [arr[left], arr[right]])(i)
        return arr

    interpolated_data = data.copy().reshape(1, -1)
    # 对数据进行插值处理
    for i in range(interpolated_data.shape[0]):
        interpolated_data[i] = lagrange_insert(interpolated_data[i])

    # save_path = os.path.join(output_root_dir, 'lagrange_interpolation', filename + '_interpolated.mat')
    # savemat(save_path, {'data': interpolated_data}, format='4')

    # figure_path = os.path.join(output_root_dir, 'lagrange_interpolation', filename + '_interpolated.png')
    plot_interpolation(data, interpolated_data, figure_save_path)

    return interpolated_data, figure_save_path


# 对信号的三次样条插值
def bicubic_interpolation_for_signal(data: np.ndarray, figure_save_path):
    """
    对信号的三次样条插值算法
    :param data: 需要进行插值的原始数据
    :param figure_save_path: 插值结果图像的保存路径
    :return: （array_filled， figure_path），插值后的数据以及显示插值结果的图片的存放路径
    """
    data = data.flatten()
    nan_indices = np.isnan(data)
    non_nan_indices = ~nan_indices

    x_non_nan = np.arange(len(data))[non_nan_indices]
    y_non_nan = data[non_nan_indices]

    cs = CubicSpline(x_non_nan, y_non_nan, bc_type='natural')
    array_filled = data.copy()
    array_filled[nan_indices] = cs(np.arange(len(data))[nan_indices])

    # save_path = os.path.join(output_root_dir, 'bicubic_interpolation', filename + '_interpolated.mat')
    # figure_path = os.path.join(output_root_dir, 'bicubic_interpolation', filename + '_interpolated.png')

    # savemat(save_path, {'data': array_filled}, format='4')
    plot_interpolation(data, array_filled, figure_save_path)

    return array_filled, figure_save_path


# 定义一个函数，用于对指定列进行双三次插值
def cubic_spline_interpolation(df, column):
    """
    双三次插值算法的关键代码
    :param df: 要进行插值的数据
    :param column: 应用插值算法的列
    :return: 无返回值
    """
    # 检查列是否为数值型
    if pd.api.types.is_numeric_dtype(df[column]):
        # 获取数值型列的非NaN值
        valid_mask = ~df[column].isnull()
        x = np.arange(len(df))  # 创建一个与DataFrame长度相同的索引数组
        y = df[column][valid_mask].values  # 获取有效的y值

        # 如果有效y值的数量大于1，则进行插值
        if len(y) > 1:
            # 创建插值函数
            interp_func = interp1d(x[valid_mask], y, kind='cubic',
                                   bounds_error=False, fill_value="extrapolate")
            # 应用插值函数
            df[column] = interp_func(x)
        else:
            print(f"Not enough valid data points to interpolate for column '{column}'.")
    else:
        print(f"Column '{column}' is not numeric, skipping interpolation.")


def bicubic_interpolation(input_file):
    """
    双三次插值算法
    :param input_file: 输入的原始数据
    :return: 插值后的数据的存放路径，以及插值结果图像的存放路径
    """
    # 读取Excel文件
    df = pd.read_excel(input_file)

    # 从第二列开始遍历DataFrame的每一列
    for column in df.columns[1:]:
        cubic_spline_interpolation(df, column)

    basename = os.path.basename(input_file)
    filename = basename.split('.')[0]
    save_dir = os.path.join(output_root_dir, 'linear_interpolation', filename)
    save_raw_dir, save_result_dir = os.path.join(save_dir, 'raw_'), os.path.join(save_dir, 'result_')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_filename = os.path.join(save_dir, basename)

    # 将插补后的数据保存到新的Excel文件
    df.to_excel(output_filename, index=False)
    # 绘制波形图并保存
    waveform_save_path = {'原始数据': waveform_drawing(input_file, save_raw_dir),
                          '结果数据': waveform_drawing(output_filename, save_result_dir)}

    return waveform_save_path, output_filename


# 拉格朗日插值
def lagrange_insert(s, n, k=3):
    y = s.reindex(list(range(n - k, n)) + list(range(n + 1, n + 1 + k)))
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)


def lagrange_interpolation(input_file):
    """
    拉格朗日插值算法
    :param input_file: 输入数据的文件的路径
    :return: 输出数据文件路径
    """
    df = pd.read_excel(input_file)
    for i in df.columns:
        for j in range(len(df)):
            if (df[i].isnull())[j]:
                df[i][j] = lagrange_insert(df[i], j)

    basename = os.path.basename(input_file)
    filename = basename.split('.')[0]
    save_dir = os.path.join(output_root_dir, 'linear_interpolation', filename)
    save_raw_dir, save_result_dir = os.path.join(save_dir, 'raw_'), os.path.join(save_dir, 'result_')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_filename = os.path.join(save_dir, basename)

    # 将插补后的数据保存到新的Excel文件
    df.to_excel(output_filename, index=False)
    # 绘制波形图并保存
    waveform_save_path = {'原始数据': waveform_drawing(input_file, save_raw_dir),
                          '结果数据': waveform_drawing(output_filename, save_result_dir)}

    return waveform_save_path, output_filename


# 牛顿插值算法
def newton_insert(s, n, k=4):
    y = s.reindex(list(range(n - k, n)) + list(range(n + 1, n + 1 + k)))
    # df['时间'] = df['时间'].dt.strftime('%Y-%m-%d %H:%M')
    x = pd.Series(list(range(n - k, n)) + list(range(n + 1, n + 1 + k)))

    y = y[y.notnull()]
    print("-------------------------------------------------------------------------")
    print(y)
    print(x)

    # 现在, 温度和对应的序号都有了
    # 差分
    def divided_diff(x, y):
        if len(y) == 1:
            return y.iloc[0]
        else:
            return (divided_diff(x[1:], y[1:]) - divided_diff(x[:-1], y[:-1])) / (x.iloc[-1] - x.iloc[0])

    # 构造插值多项式
    def newton_polynomial(x, y):
        if len(y) == 1:
            return y.iloc[0]
        else:
            return newton_polynomial(x[1:], y[:-1]) + divided_diff(x, y) * np.prod(np.array(x.iloc[0]) - np.array(x))

    interpolated_value = newton_polynomial(x, y)
    # return interpolated_value
    return round(interpolated_value, 6)


def newton_interpolation(input_file):
    """
    牛顿插值算法
    :param input_file: input file path
    :return: output file path
    """
    df = pd.read_excel(input_file)
    for i in df.columns:
        for j in range(len(df)):
            if (df[i].isnull())[j]:
                df[i][j] = newton_insert(df[i], j)

    basename = os.path.basename(input_file)
    filename = basename.split('.')[0]
    save_dir = os.path.join(output_root_dir, 'linear_interpolation', filename)
    save_raw_dir, save_result_dir = os.path.join(save_dir, 'raw_'), os.path.join(save_dir, 'result_')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_filename = os.path.join(save_dir, basename)

    # 将插补后的数据保存到新的Excel文件
    df.to_excel(output_filename, index=False)
    # 绘制波形图并保存
    waveform_save_path = {'原始数据': waveform_drawing(input_file, save_raw_dir),
                          '结果数据': waveform_drawing(output_filename, save_result_dir)}

    return waveform_save_path, output_filename


def newton_interpolation_for_signal(data: np.ndarray, figure_save_path):
    """
    对于一维信号的牛顿插值
    :param data: 需要进行插值的原始数据
    :param figure_save_path: 结果图像保存路径
    :return: （array_filled， figure_path），插值后的数据以及显示插值结果的图片的存放路径
    """

    # 定义牛顿插值函数
    def NewtonInsert(arr):
        n = len(arr)
        for i in range(n):
            if np.isnan(arr[i]):
                left, right = i - 1, i + 1
                while np.isnan(arr[left]) and left >= 0:
                    left -= 1
                while np.isnan(arr[right]) and right < n:
                    right += 1
                if left < 0 or right >= n:
                    continue

                # 差分
                def divided_diff(x, y):
                    if len(y) == 1:
                        return y[0]
                    else:
                        return (divided_diff(x[1:], y[1:]) - divided_diff(x[:-1], y[:-1])) / (x[-1] - x[0])

                # 构造插值多项式
                def newton_polynomial(x, y):
                    if len(y) == 1:
                        return y[0]
                    else:
                        return newton_polynomial(x[1:], y[:-1]) + divided_diff(x, y) * np.prod(
                            np.array(x[0]) - np.array(x))

                # 获取插值结果
                interpolated_value = newton_polynomial(np.array([left, right]), np.array([arr[left], arr[right]]))
                arr[i] = interpolated_value

        return arr

    interpolated_data = data.copy().reshape(1, -1)
    # 对数据进行插值处理
    for i in range(interpolated_data.shape[0]):
        interpolated_data[i] = NewtonInsert(interpolated_data[i])

    plot_interpolation(data, interpolated_data, figure_save_path)

    return interpolated_data, figure_save_path


# 线性插值
def linear_insert(s, n, k=1):
    # 获取相邻点的值
    y = s.reindex(range(n - k, n + k + 1))
    y = y.dropna()  # 去除空值
    if y.empty:  # 处理边界情况
        return np.nan

    x = y.index
    # 计算插值结果
    interpolated_value = np.interp(n, x, y)
    return interpolated_value


# 使用深度学习模型的插值算法
def deeplearning_interpolation(raw_data: np.ndarray, figure_save_path: str):
    """
    使用深度学习模型LSTM的插值算法
    :param figure_save_path: 结果图像的保存路径
    :param raw_data: 原始信号
    :return: 插补后信号，以及展现插补后结果的图片
    """
    # print(f'raw_data.shape: {raw_data.shape}')
    data_array = raw_data.flatten()[0:144]
    # print(f'data_array.shape: {data_array.shape}')
    test_data_size = 12

    train_data = data_array[:-test_data_size]
    test_data = data_array[-test_data_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

    # 将训练集转换为张量
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    # 假设我们每天采集到的数据数量是12条
    train_window = 12

    # 相当于拿一个长度为12的窗在132个元素上从左向右滑动，一次滑动一个元素；每滑动一次，把窗的右侧紧邻的一个元素当做标签；窗可以滑动132-12=120次，因此获得120个结果
    # 这样我们得到的所有train_inout_seq可以正在时间方面更加相关
    def create_inout_sequences(input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw):
            train_seq = input_data[i:i + tw]
            train_label = input_data[i + tw:i + tw + 1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    # 对数据进行预处理之后 开始训练模型，这里采用长短期记忆网络（LSTM）
    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size

            self.lstm = nn.LSTM(input_size, hidden_layer_size)

            self.linear = nn.Linear(hidden_layer_size, output_size)

            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                                torch.zeros(1, 1, self.hidden_layer_size))

        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    # 构造神经网络
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 进行模型训练，这里设定训练轮次为100轮
    epochs = 100
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    fut_pred = 12
    test_inputs = train_data_normalized[-train_window:].tolist()

    model.eval()
    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())

    # 设置窗口大小
    train_window = 12  # 或者你定义的任何窗口大小

    data_interpolated = raw_data.copy()
    # 对于每一个数据向量，进行插值处理
    # for key in data_interpolated:
    #     if not key.startswith('__'):
    array = data_interpolated.flatten()  # 将数据展开成一维数组

    # 获取当前行的数据
    y = array
    x = np.arange(len(y))

    # 找到非NaN值和NaN值的索引
    nan_idx = np.isnan(y)
    not_nan_idx = ~nan_idx

    # 如果行中有NaN值且非NaN值数量大于等于2，才进行插值
    if nan_idx.any() and np.sum(not_nan_idx) >= 2:
        # 使用非NaN值进行插值
        # interp_func = interp1d(x[not_nan_idx], y[not_nan_idx], kind='linear', fill_value="extrapolate")

        # 使用深度学习模型进行插值
        for idx in np.where(nan_idx)[0]:
            # 取当前 NaN 值前的 train_window 个数据点
            start_idx = max(0, idx - train_window)
            seq = y[start_idx:idx]  # 取到 NaN 之前的数据
            if len(seq) < train_window:
                # 如果长度不足 train_window，用前面的数据填充
                seq = np.pad(seq, (train_window - len(seq), 0), 'constant', constant_values=np.nan)

            seq = torch.tensor(seq, dtype=torch.float32).view(-1, 1,
                                                              1)  # 转换为张量并调整形状为 [seq_len, batch_size, input_size]
            with torch.no_grad():
                # 模型预测
                # model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                #                torch.zeros(1, 1, model.hidden_layer_size))
                pred = model(seq).item()

            # 用预测结果替换 NaN 值
            y[idx] = pred
        array = y

    # 更新数据
    data_interpolated = array.reshape(1, -1)  # 确保数据仍然是一行
    # save_path = os.path.join(output_root_dir, 'deeplearning_interpolation', filename + '_interpolated.mat')
    # 保存处理后的数据到新的MAT文件
    # savemat(save_path, {'data': data_interpolated}, format='4')

    # figure_path = os.path.join(output_root_dir, 'deeplearning_interpolation', filename + '_interpolated.png')
    plot_interpolation(raw_data, data_interpolated, figure_save_path)

    return data_interpolated, figure_save_path


# 线性插值算法
def linear_interpolation(input_file):
    """
    线性插值
    :param input_file: input file path
    :return: output file path
    """
    df = pd.read_excel(input_file)
    # 遍历数据框中的每一列，对缺失值进行插值
    for column in df.columns:
        for index, value in df[column].items():
            if pd.isnull(value):
                df.at[index, column] = linear_insert(df[column], index)

    basename = os.path.basename(input_file)
    filename = basename.split('.')[0]
    save_dir = os.path.join(output_root_dir, 'linear_interpolation', filename)
    save_raw_dir, save_result_dir = os.path.join(save_dir, 'raw_'), os.path.join(save_dir, 'result_')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_filename = os.path.join(save_dir, basename)

    # 将插补后的数据保存到新的Excel文件
    df.to_excel(output_filename, index=False)
    # 绘制波形图并保存
    waveform_save_path = {'原始数据': waveform_drawing(input_file, save_raw_dir),
                          '结果数据': waveform_drawing(output_filename, save_result_dir)}

    return waveform_save_path, output_filename


def linear_interpolation_for_signal(data: np.ndarray, figure_save_path):
    """
    对于一维信号的线性插值
    :param data: 需要进行插值的原始数据
    :param figure_save_path: 结果图像的保存路径
    :return: （array_filled， figure_path），插值后的数据以及显示插值结果的图片的存放路径
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
    # save_path = os.path.join(output_root_dir, 'linear_interpolation', filename + '_interpolated.mat')
    # savemat(save_path, {'data': interpolated_data}, format='4')

    # figure_path = os.path.join(output_root_dir, 'linear_interpolation', filename + '_interpolated.png')
    plot_interpolation(data, interpolated_data, figure_save_path)

    return interpolated_data, figure_save_path


# 绘制插值处理后结果的图像
def waveform_drawing(datafile, output_dir):
    """
    绘制插值后结果波形图
    :param datafile: 输入数据文件（插值后的数据）
    :param output_dir: 结果图像存放的目录
    :return: 结果图像的存放路径
    """
    # 设置全局字体属性，这里以SimHei为例
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['font.size'] = 30

    # 设置label字体大小
    font = FontProperties(size=25)

    # 读取Excel文件
    data = pd.read_excel(datafile, parse_dates=['时间'])
    data['时间'] = pd.to_datetime(data['时间'])

    # data = pd.read_excel('data/Insert.xlsx')
    data = data.tail(1000)

    # 提取时间点和各项性能指标
    time = data.iloc[:, 1]
    performance_indicators = (data.iloc[:, 2: 5], data.iloc[:, 5: 8], data.iloc[:, 8: 11], data.iloc[:, 11:])
    labels = ('发电机温度随时间变化折线图', '电网电压随时间变化折线图', '电网电流随时间变化折线图',
              '其他性能指标随时间变化折线图')
    out_feature_name = ('发电机温度', '电网电压', '电网电流', '其他性能指标')
    out_features = (output_dir + 'temperature.png', output_dir + 'grid_voltage.png', output_dir + 'current.png',
                    output_dir + 'other.png')
    save_path = {k: v for (k, v) in zip(out_feature_name, out_features)}

    # 绘制折线图
    for (performance, label, path) in zip(performance_indicators, labels, out_features):
        plt.figure(figsize=(32, 8))
        for column in performance.columns:
            plt.plot(time, performance[column], label=column)

        # 使用AutoDateLocator自动选择最佳的时间间隔
        locator = mdates.AutoDateLocator()
        plt.gca().xaxis.set_major_locator(locator)

        plt.xlabel('时间点')
        plt.ylabel('性能指标')
        plt.title(label)
        plt.legend(prop=font)
        plt.grid(True)
        plt.savefig(path)
    plt.cla()
    return save_path


def extract_time_domain_for_three_dims(datafile):
    """
    读取三维数据文件并进行人工时域特征提取
    :param datafile:
    :return: filepath of extracted time domain features
    """
    file_type = get_filetype(datafile)
    if file_type == 'csv':
        data = pd.read_csv(datafile)
    elif file_type == 'npy':
        data = np.load(datafile)
    else:
        return 'Invalid file'

    features = []
    columns = []

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            time_domain_feature = time_domain_extraction(data[i, j, :])
            features.append(time_domain_feature.values())
            if i == data.shape[0] - 1 and j == data.shape[1] - 1:
                columns = list(time_domain_feature.keys())

    df = pd.DataFrame(data=features, columns=columns)
    out_filename = 'time_' + os.path.basename(datafile).split('.')[0] + '.csv'
    output_file = os.path.join(
        'app1/module_management/algorithms/functions/preprocessing_results/feature_extraction/time_domain',
        out_filename)
    df.to_csv(output_file, index=False)
    return output_file


# 对三维的数据文件进行频域特征的提取
def extract_frequency_domain_for_three_dims(datafile):
    """
    读取三维数据文件并进行人工频域特征提取
    :param datafile: 传入的数据文件
    :return: 包含每一行所提取数据的频域特征的 pd.Dataframe
    """
    file_type = get_filetype(datafile)
    if file_type == 'csv':
        data = pd.read_csv(datafile)
    elif file_type == 'npy':
        data = np.load(datafile)
    else:
        return 'Invalid file'

    features = []
    columns = []

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            fre_domain_feature = frequency_domain_extraction(data[i, j, :], sample_rate=4000)
            features.append(fre_domain_feature.values())
            if i == data.shape[0] - 1 and j == data.shape[1] - 1:
                columns = list(fre_domain_feature.keys())

    df = pd.DataFrame(data=features, columns=columns)
    out_filename = 'freq_' + os.path.basename(datafile).split('.')[0] + '.csv'
    output_file = os.path.join(
        'app1/module_management/algorithms/functions/preprocessing_results/feature_extraction/time_domain',
        out_filename)
    df.to_csv(output_file, index=False)
    return output_file


def extract_features_for_three_dims(datafile, features_to_extract):
    """
    读取三维数据文件并进行人工时域和频域特征提取
    :param features_to_extract: 所要提取的时域和频域的特征：{time_domain: ['', '', ...], frequency_domain: ['', '', ...]}
    :param datafile: 传入要提取特征的数据文件
    :return:
    """
    file_type = get_filetype(datafile)
    if file_type == 'csv':
        data = pd.read_csv(datafile)
    elif file_type == 'npy':
        data = np.load(datafile)
    else:
        return 'Invalid file'

    # all_time_features = ['最大值', '最小值', '中位数', '峰峰值', '均值', '方差', '标准差', '峰度', '偏度', '整流平均值',
    #                      '均方根',
    #                      '方根幅值', '波形因子', '峰值因子', '脉冲因子', '裕度因子', '四阶累积量', '六阶累积量']
    # all_frequency_features = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值',
    #                           '谱峭度的标准差', '谱峭度的峰度', '谱峭度的偏度']

    # 对单个信号提取频域特征
    def extract_frequency_domain_features(data, frequency_domain_features):
        features = []
        columns = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                fre_domain_feature = frequency_domain_extraction(data[i, j, :], frequency_domain_features)
                features.append(fre_domain_feature.values())
                if i == data.shape[0] - 1 and j == data.shape[1] - 1:
                    columns = list(fre_domain_feature.keys())

        return pd.DataFrame(data=features, columns=columns)

    # 对单个信号提取时域特征
    def extract_time_domain_features(data, time_domain_features):
        features = []
        columns = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                time_domain_feature = time_domain_extraction(data[i, j, :], time_domain_features)
                features.append(time_domain_feature.values())
                if i == data.shape[0] - 1 and j == data.shape[1] - 1:
                    columns = list(time_domain_feature.keys())
        return pd.DataFrame(data=features, columns=columns)

    time_domain_features = None
    frequency_domain_features = None

    # 根据要提取的时域以及频域的信号（features_to_extract），提取传入信号的时域以及频域的特征
    for domain, features in features_to_extract.items():
        if features:
            if domain == 'time_domain':
                time_domain_features = extract_time_domain_features(data, features)
            else:
                frequency_domain_features = extract_frequency_domain_features(data, features)

    features = pd.concat([time_domain_features, frequency_domain_features], axis=1)
    out_filename = os.path.basename(datafile).split('.')[0] + '.csv'
    output_file = os.path.join(
        'app1/module_management/algorithms/functions/preprocessing_results/feature_extraction/time_frequency_domain',
        out_filename)
    features.to_csv(output_file, index=False)
    return output_file


def extract_signal_features(input_data: np.ndarray, features_to_extract, filename=None, save=False, user_dir=None):
    """
    对输入的一维信号进行人工时域和频域特征的提取
    :param user_dir: 特征提取结果保存到的用户目录
    :param save: save the features or not，选择是否要保存提取特征
    :param filename: the filename of the input_data，所传入原始数据的来源文件
    :param input_data: input data，要提取特征的数据文件
    :param features_to_extract: 要提取的时域以及频域的特征：{time_domain: feature list, frequency_domain: feature list}
    :return:
    """

    # all_time_features = ['最大值', '最小值', '中位数', '峰峰值', '均值', '方差', '标准差', '峰度', '偏度', '整流平均值',
    #                      '均方根',
    #                      '方根幅值', '波形因子', '峰值因子', '脉冲因子', '裕度因子', '四阶累积量', '六阶累积量']
    # all_frequency_features = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值',
    #                           '谱峭度的标准差', '谱峭度的峰度', '谱峭度的偏度']

    time_domain_features = None
    frequency_domain_features = None

    all_features = []
    try:
        for _, features in features_to_extract.items():
            all_features.extend(features)

        # 获取用户要提取的所有特征
        if len(input_data.shape) == 2:
            new_data = input_data.flatten()
        else:
            new_data = input_data.copy()
        num_example = int(len(new_data) / 2048)
        split_data = np.ndarray(shape=(num_example, 2048))
        # 将new_data按照2048长度的切分
        for i in range(num_example):
            split_data[i, :] = new_data[i * 2048:(i + 1) * 2048]

        # print(f'split_data: {split_data.shape}')
        # print('new_data: ', new_data, '\n', 'new_data.shape: ', new_data.shape)
        # print('new_data has nan: ', np.sum(np.isnan(new_data)))
        all_example_features_extracted = pd.DataFrame(columns=all_features)
        all_example_features_extracted_group_by_sensor = {'传感器1': []}
        for i in range(num_example):
            # 根据用户需要提取时域和频域特征
            for domain, features in features_to_extract.items():
                if features:
                    if domain == 'time_domain':
                        time_domain_features = time_domain_extraction(split_data[i, :], features)
                        values = [list(time_domain_features.values())]
                        time_domain_features = pd.DataFrame(data=values, columns=features)
                    else:
                        frequency_domain_features = frequency_domain_extraction(split_data[i, :], features)
                        values = [list(frequency_domain_features.values())]
                        frequency_domain_features = pd.DataFrame(data=values, columns=features)
            single_features_extracted = pd.concat([time_domain_features, frequency_domain_features], axis=1)
            # single_features_extracted_group_by_sensor = {'sensor_1': single_features_extracted.iloc[0].tolist()}
            if save and filename is not None:
                # 单传感器特征保存
                all_example_features_extracted_group_by_sensor['传感器1'].append(
                    single_features_extracted.iloc[0].tolist())
            all_example_features_extracted = pd.concat([all_example_features_extracted, single_features_extracted])
            # 将single_features_extracted作为all_example_features_extracted的一行数据插入
            # all_example_features_extracted.append(single_features_extracted, ignore_index=True)
        # print(f'all_example_features_extracted: {all_example_features_extracted}')
        # print(f'all_example_features_extracted_group_by_sensor: {all_example_features_extracted_group_by_sensor}')

        # print('features_extracted: ', features_extracted)
        # 根据需要将提取到的特征进行保存，或者直接返回
        if save and filename is not None:
            # 对提取到的单传感器特征进行保存
            out_filename = filename + '.csv'
            output_file_dir = ('app1/module_management/algorithms/functions/preprocessing_results/feature_extraction'
                               '/time_frequency_domain/') + user_dir
            if not os.path.exists(output_file_dir):
                os.makedirs(output_file_dir)
            output_file = output_file_dir + '/' + out_filename

            all_example_features_extracted.to_csv(output_file, index=False)
            return output_file, {'features_extracted_group_by_sensor': all_example_features_extracted_group_by_sensor,
                                 'features_name': all_features}, num_example
        else:
            # 作为多传感器数据中一个传感器信号的特征提取结果返回
            return all_example_features_extracted.iloc[0].tolist()
    except Exception as e:
        print('单传感器特征提取出错: ', str(e))
        return None, {}


# 提取多传感器信号的特征（包含时域特征和频域特征，具体特征根据输入的特征名进行提取）
def extract_features_with_multiple_sensors(input_data: np.ndarray, features_to_extract, filename, user_dir=None):
    """
    提取多传感器信号的特征
    :param user_dir: 结果保存的用户目录
    :param input_data: 传入的多传感器原始信号
    :param features_to_extract: 要提取的所有特征
    :param filename: 原始信号的来源文件
    :return: 提取到的特征的存放路径（以字符串的形式），以及根据传感器分组的特征（以字典的形式返回）
    """

    # 帧长以及步长
    frame_length = 2048
    step_size = 2048

    # Feature columns time_feature_columns = ['mean', 'var', 'std', 'skewness', 'kurtosis', 'cumulant_4th',
    # 'cumulant_6th', 'max', 'min', 'median', 'peak_to_peak', 'rectified_mean', 'rms', 'root_amplitude',
    # 'waveform_factor', 'peak_factor', 'impulse_factor', 'margin_factor'] freq_feature_columns = ['centroid_freq',
    # 'msf', 'rms_freq', 'freq_variance', 'freq_std', 'spectral_kurt_mean', 'spectral_kurt_peak']
    # 所有的时域特征
    time_feature_columns = ['均值', '方差', '标准差', '偏度', '峰度', '四阶累积量', '六阶累积量', '最大值', '最小值',
                            '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值', '波形因子', '峰值因子', '脉冲因子',
                            '裕度因子']
    # 所有的频域特征
    freq_feature_columns = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值',
                            '谱峭度的峰度']

    all_features = []
    for _, features in features_to_extract.items():
        all_features.extend(features)

    # 传感器名称
    sensor_names = ['X维力(N)', 'Y维力(N)', 'Z维力(N)', 'X维振动(g)', 'Y维振动(g)', 'Z维振动(g)', 'AE-RMS (V)']

    extracted_features_group_by_sensor = {sensor_name: [] for sensor_name in sensor_names}

    combined_columns = []
    try:
        for sensor in sensor_names:
            combined_columns.extend([f"{sensor}_{feature}" for feature in all_features])

        # 多传感器的特征
        combined_features_df = pd.DataFrame(columns=combined_columns)

        num_frames = (input_data.shape[0] - frame_length) // step_size + 1
        # 检查数据是否为多传感器数据
        new_data = input_data.copy()
        if len(new_data.flatten()) // 2048 <= 1:
            print('数据不合规范')
            return None, {}
        for frame_idx in range(num_frames):
            combined_features = []
            for i, sensor_name in enumerate(sensor_names):
                frame = input_data[frame_idx * step_size:frame_idx * step_size + frame_length, i]
                # time_features = compute_time_domain_features(frame)
                # freq_features = compute_frequency_domain_features(frame, frame_length)
                features = extract_signal_features(frame, features_to_extract, save=False)
                extracted_features_group_by_sensor[sensor_name].append(features)
                combined_features.extend(features)
                # if frame_idx == num_frames - 1:
                #     print(f'extracted_features_group_by_sensor: { len(extracted_features_group_by_sensor[sensor_name])}')
            # combined_features.append(1)
            combined_features_df.loc[frame_idx] = combined_features
        # print(f'combined_features_df: {combined_features_df.shape}')
        out_filename = filename + '.csv'
        output_file_dir = 'app1/module_management/algorithms/functions/preprocessing_results/feature_extraction/time_frequency_domain/' + user_dir
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)
        output_file = output_file_dir + '/' + out_filename

        combined_features_df.to_csv(output_file, index=False)

        return output_file, {'features_extracted_group_by_sensor': extracted_features_group_by_sensor,
                             'features_name': all_features}, num_frames
    except Exception as e:
        print('多传感器特征提取出错：', str(e))
        return None, {}


def plot_signal(example, filename, multiple_sensor=False):
    # example, filename = load_data(example_filepath)
    # 设置字体以支持中文显示
    matplotlib.use('Agg')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
    plt.figure(figsize=(20, 10))

    if not multiple_sensor:
        if len(example.shape) == 2:
            example = example[0, :]
        else:
            example = example
        plt.plot(example)
        plt.title('信号波形图')

        plt.xlabel('采样点', fontsize=18)
        plt.ylabel('信号值', fontsize=18)
    else:

        # 创建图形和子图
        # plt.figure(figsize=(20, 10))  # 设置图形的大小
        num_sensors = example.shape[1]  # 获取传感器的数量
        for i in range(num_sensors):
            plt.subplot(num_sensors, 1, i + 1)  # 创建子图，num_sensors 行 1 列，当前是第 i+1 个子图
            plt.plot(example[:, i])  # 绘制第 i 个传感器的信号
            plt.title(f'Sensor {i + 1}')  # 设置子图的标题
            plt.xlabel('采样点', fontsize=18)  # 设置 x 轴标签
            plt.ylabel('信号值', fontsize=18)  # 设置 y 轴标签

        # 调整子图之间的间距
        plt.title('信号波形图')
        plt.tight_layout()
    save_path = 'app1/module_management/algorithms/functions/fault_diagnosis/' + filename + '.png'

    plt.savefig(save_path)

    return save_path


choose_features = ['标准差', '均方根', '方差', '整流平均值', '方根幅值', '峰峰值', '六阶累积量', '均值', '四阶累积量',
                   '最小值']

choose_features_multiple = ['X维力(N)_六阶累积量', 'X维力(N)_峰峰值', 'X维力(N)_重心频率', 'X维力(N)_最大值',
                            'X维力(N)_四阶累积量',
                            'X维力(N)_方差', 'X维力(N)_裕度因子', 'X维力(N)_标准差', 'X维力(N)_均方根',
                            'X维力(N)_方根幅值']


# def get_extra_algorithm_dir(user_dir, algorithm_type):
#     """
#     返回对应用户的专有算法的存放目录
#     :param user_dir: 用户名对应的目录
#     :return: 对应用户名和算法类型的专有算法的存放目录
#     """
#     username = user_dir.split('/')[0]
#     extra_algorithm_dir = "app1/module_management/algorithms/models/" +
#
#     return "app1/module_management/algorithms/functions/fault_diagnosis" +


# 无量纲化, 主要用于SVM等需要标准化的机器学习算法的预处理
def dimensionless(input_data, features_group_by_sensor, user_dir=None, option=None, multiple_sensor=False,
                  use_log=False, extra_algorithm_filepath=None):
    """
    :param extra_algorithm_filepath: 如果不为None则使用用户上传的专用算法
    :param user_dir: 标准化结果的保存路径
    :param use_log: 使用训练模型时用到的标准化方法，否则对于输入的信号序列进行标准化
    :param multiple_sensor: 是否为多传感器的数据
    :param features_group_by_sensor: 提取到的不同传感器的特征
    :param input_data: 输入的数据
    :param option: 选择使用的归一化算法
    :return: 标准化后的特征数据, multiple_sensor=True时为多传感器数据
    """
    # print(f'xxxxxxxxxxx....')
    # print(f'scaler_input_data: {input_data}')
    data_scaled = input_data.copy()  # 实际使用的数据样本
    if features_group_by_sensor is not None:
        data_scaled_display = features_group_by_sensor.copy()  # 用于展示在前端页面中的数据样本
    if multiple_sensor is None:
        if input_data.flatten().shape[0] <= 2048:
            multiple_sensor = False
        else:
            multiple_sensor = True
    # 用户上传的专用算法的存放目录
    username = user_dir.split('/')[0]  # 用户名

    # input_filepath = base_dir + '/intermediate_data.npy'
    if extra_algorithm_filepath is not None:
        # 存放用户上传的训练模型
        base_dir = os.path.dirname(extra_algorithm_filepath)
        print(f"base_dir: {base_dir}")
        log_dir = base_dir + '/' + os.path.basename(extra_algorithm_filepath).split('.')[0] + '.pkl'
        print(f'log_dir: {log_dir}')
    else:
        log_dir = None
    # scaled_data = base_dir + '/scaled_data.npy'
    # 根据传入参数选择标准化方法
    if option == 'max_min':
        if use_log and log_dir is not None:
            # 使用用户上传的专有无量纲化模型
            data_scaler = joblib.load(log_dir)
        else:
            # 使用标准的无量纲化模型
            data_scaler = MinMaxScaler()

    elif option == 'z_score':
        if not use_log:
            data_scaler = StandardScaler()
        elif log_dir is not None:
            # 使用用户上传的专有无量纲化模型
            data_scaler = joblib.load(log_dir)
        else:
            # 选择svm模型训练时的标准化进行逆标准化
            if not multiple_sensor:
                # 针对单传感器的数据标准化
                data_scaler = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/scaler_2.pkl')
            else:
                # 针对多传感器的数据标准化
                data_scaler = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/mutli_scaler'
                                          '.pkl')

    elif option == 'max_abs_scaler':
        if use_log and log_dir is not None:
            # 使用用户上传的专有无量纲化模型
            data_scaler = joblib.load(log_dir)
        else:
            # 使用标准的无量纲化模型
            data_scaler = MaxAbsScaler()
    else:
        if use_log and log_dir is not None:
            # 使用用户上传的专有无量纲化模型
            data_scaler = joblib.load(log_dir)
        else:
            # 使用标准的无量纲化模型
            data_scaler = RobustScaler()
    # 避免存放文件的路径错误，根据用户名生成对应的用户目录
    if option:
        save_path_dir = output_root_dir + '/' + 'data_scale' + '/' + option
    else:
        save_path_dir = output_root_dir + '/' + 'data_scale' + '/' + os.path.basename(extra_algorithm_filepath).split('.')[0]
    print(f'save_path_dir_123: {save_path_dir}')
    save_path_dir += '/' + user_dir
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    # if not os.path.exists(save_path_dir):
    #     os.makedirs(save_path_dir)
    if extra_algorithm_filepath is not None:
        username = user_dir.split('/')[0]  # 调用该增值无量纲化组件的用户的用户名
        # 存放用户专用无量纲化算法的用户目录
        base_dir = os.path.dirname(extra_algorithm_filepath)
        extra_algorithm_name = extra_algorithm_filepath.split('/')[-1]  # 增值无量纲化算法的文件名
        extra_algorithm_user_dir = base_dir + '/' + username  # 不同用户调用增值组件时，每个用户都有自己的文件夹存放数据
        if not os.path.exists(extra_algorithm_user_dir):
            os.makedirs(extra_algorithm_user_dir)
    try:
        if not use_log:
            # 对输入的信号序列进行标准化
            if not multiple_sensor:
                # 对于单传感器数据进行标准化
                # 使用开发者用户上传的专用算法
                if extra_algorithm_filepath:
                    input_filepath = extra_algorithm_user_dir + '/input_data.npy'
                    scaled_data = extra_algorithm_user_dir + '/scaled_data.npy'
                    np.save(input_filepath, input_data.reshape(-1, 1))
                    result = subprocess.run(shell=True, capture_output=True,
                                            args=f"cd {base_dir} & python ./{extra_algorithm_name} --input-filepath "
                                                 f"./{username}/input_data.npy --output-filepath ./{username}/scaled_data.npy")
                    print(f'result: {result}')
                    error = result.stderr.decode('utf-8')
                    if not error:
                        data_scaled = np.load(scaled_data)
                    else:
                        print('专有无量纲化方法出错...')
                        return None, error, None, multiple_sensor
                # 使用系统中集成的无量纲化方法
                else:
                    data_scaled = data_scaler.fit_transform(input_data.reshape(-1, 1))
                save_path = save_path_dir + '/' + 'data_scale_result.png'
                print(f"Saving data to {save_path}")
                result_figure = plot_scaler(input_data.flatten(), data_scaled.flatten(), save_path)
                data_scaled_save_path = save_path_dir + '/' + 'data_scaled.npy'
                np.save(data_scaled_save_path, data_scaled)

                print('运行专有无量纲化方法结束...')
                return data_scaled.reshape(-1), [result_figure], data_scaled_save_path, multiple_sensor
            else:
                # 对于多传感器数据进行标准化
                shape = input_data.shape
                if shape[0] < shape[1]:
                    input_data = input_data.T
                # input_data = input_data.reshape(2048, -1)
                num_sensors = input_data.shape[1]
                all_data_scaled = []
                all_figure_paths = []
                for sensor in range(num_sensors):
                    # 对于单传感器数据进行标准化
                    if extra_algorithm_filepath:
                        # 使用用户上传的专用算法
                        username = user_dir.split('/')[0]  # 用户名
                        # 存放用户专用无量纲化算法的用户目录

                        input_filepath = base_dir + '/input_data.npy'
                        scaled_data = base_dir + '/scaled_data.npy'
                        np.save(input_filepath, input_data[:, sensor])
                        result = subprocess.run(shell=True, capture_output=True,
                                                args=f"cd {base_dir} & python ./{extra_algorithm_name} --input-filepath "
                                                     f"./input_data.npy --output-filepath ./scaled_data.npy")
                        error = result.stderr.decode('utf-8')
                        if not error:
                            data_scaled = np.load(scaled_data)
                        else:
                            print(f'Error: {error}')
                            return None, error, None, multiple_sensor
                    else:
                        # 使用系统集成的无量纲化算法
                        data_scaled = data_scaler.fit_transform(input_data[:, sensor].reshape(-1, 1))
                    # data_scaled = data_scaler.fit_transform(input_data[:, sensor].reshape(-1, 1))
                    all_data_scaled.append(data_scaled)
                    save_path = save_path_dir + '/' + f'scale_result_sensor_{sensor}.png'
                    result_figure = plot_scaler(input_data[:, sensor].flatten(), data_scaled.flatten(), save_path)
                    all_figure_paths.append(result_figure)
                data_scaled_save_path = save_path_dir + '/' + 'data_scaled.npy'
                np.save(data_scaled_save_path, all_data_scaled)
                return np.array(all_data_scaled).reshape(-1,
                                                         num_sensors), all_figure_paths, data_scaled_save_path, multiple_sensor
        else:
            # 根据模型训练时保存的训练数据的参数对输入的样本特征进行逆标准化
            if not multiple_sensor:
                # 单传感器的标准化
                # data_scaler = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/scaler_2.pkl')
                data_scaled[choose_features] = data_scaler.transform(data_scaled[choose_features])

            else:
                # data_scaler = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/mutli_scaler.pkl')
                # 多传感器的标准化
                data_scaled[choose_features_multiple] = data_scaler.transform(data_scaled[choose_features_multiple])

                index_start = 0
                # 需要展示在前端页面中的标准化以后的数据
                try:
                    for k, v in data_scaled_display.items():
                        features_num = len(v)
                        data_scaled_display[k] = data_scaled.iloc[:, index_start:features_num].to_list()
                        index_start += features_num
                except Exception as e:
                    print(str(e))
            data_scaled_save_path = save_path_dir + '/' + 'data_scaled.npy'
            np.save(data_scaled_save_path, data_scaled)
            return data_scaled, data_scaled_display, data_scaled_save_path, multiple_sensor
    except Exception as e:
        print('无量纲化方法出错......')
        return None, e, None, multiple_sensor


# 绘制标准化的结果图像
def plot_scaler(raw_data, data_scaled, save_path):
    """
    绘制标准化的结果图像
    :param raw_data: 原始数据
    :param data_scaled: 标准化后的数据
    :param save_path: 结果图像的保存路径
    :return: 结果图像文件存放路径
    """
    matplotlib.use('Agg')
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题

    fig, (axe1, axe2) = plt.subplots(2, 1, figsize=(12, 6), sharex='col')
    axe1.set_title('原始信号', fontsize=15)
    axe1.set_ylabel('采样值', fontsize=12)
    axe1.plot(raw_data)
    axe2.set_title('标准化后信号', fontsize=15)
    axe2.set_ylabel('采样值', fontsize=12)
    axe2.set_xlabel('采样点', fontsize=12)
    axe2.plot(data_scaled)

    fig.savefig(save_path)

    return save_path
