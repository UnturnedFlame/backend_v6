"""
主要实现对单个信号的时频域特征提取，共27个特征
时域特征：均值、方差、标准差、峰度、偏度、四阶累积量、六阶累积量、最大值、最小值、中位数、峰峰值、整流平均值、均方根、方根幅值、波形因子、峰值因子、脉冲因子、裕度因子
频域特征：重心频率、均方频率、均方根频率、频率方差、频率标准差、谱峭度的均值、谱峭度的标准差、谱峭度的峰度、谱峭度的偏度

时域特征提取函数：timeDomain_extraction(signal)
频域特征提取函数：FreDomain_extraction(signal,sample_rate)
输入信号signal为numpy数组（一维），采样率sample_rate为double类型
输出为字典类型


"""
import numpy as np
import scipy.io
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy.fft import fft

'''
主要完成信号的时域特征提取, 其中各个函数的传入参数为数组形式的一维离散信号
'''


# 最大值
def max(signal):
    max_v = np.max(signal)
    return max_v


# 最小值
def min(signal):
    min_v = np.min(signal)
    return min_v


# 中位数
def median(signal):
    median = np.median(signal)
    return median


# 峰峰值
def peak_peak(signal):
    pk_pk = np.max(signal) - np.min(signal)
    return pk_pk


# 均值
def mean(signal):
    mean = np.mean(signal)
    return mean


# 方差
def variance(signal):
    var = np.var(signal, ddof=1)
    return var


# 标准差
def std(signal):
    std = np.std(signal, ddof=1)
    return std


# 峰度（峭度）
# def kurtosis(signal):
#     kurt = st.kurtosis(signal, fisher=False)
#     return kurt


# 偏度
# def time_skew(signal):
#     skew = st.skew(signal)
#     return skew


# 均方根值
def root_mean_square(signal):
    rms = np.sqrt((np.mean(signal ** 2)))
    return rms


# 波形因子，信号均方根值和整流平均值的比值
def waveform_factor(signal):
    rms = root_mean_square(signal)  # 计算RMS有效值，即均方根值
    ff = rms / commutation_mean(signal)
    return ff


# 峰值因子，信号峰值与RMS有效值的比值
def peak_factor(signal):
    peak_max = np.max(np.abs(signal))  # 计算峰值
    rms = root_mean_square(signal)  # 计算RMS
    pf = peak_max / rms  # 峰值因子
    return pf


# 脉冲因子，信号峰值与整流平均值的比值
def pulse_factor(signal):
    peak_max = np.max(np.abs(signal))  # 计算峰值
    pf = peak_max / np.mean(np.abs(signal))
    return pf


# 裕度因子，信号峰值和方根幅值的比值
def margin_factor(signal):
    peak_max = np.max(np.abs(signal))  # 计算峰值
    rampl = root_amplitude(signal)  # 计算方根幅值
    margin_factor = peak_max / rampl  # 裕度因子
    return margin_factor


# 方根幅值
def root_amplitude(signal):
    rampl = ((np.mean(np.sqrt(np.abs(signal))))) ** 2
    return rampl


# 整流平均值，信号绝对值的平均值
def commutation_mean(signal):
    cm = np.mean(np.abs(signal))
    return cm


# 四阶累积量
def fourth_order_cumulant(signal):
    # 计算一阶、二阶中心距
    mean = np.mean(signal)
    variance = np.var(signal)

    # 计算四阶累积量
    # fourth_order_cumulant = np.mean((signal - np.mean(signal))**4)
    fourth_order_cumulant = np.mean((signal - mean) ** 4) - 3 * variance ** 2
    return fourth_order_cumulant


# 六阶累积量
def sixth_order_cumulant(signal):
    # 计算一阶、二阶中心距
    mean = np.mean(signal)
    variance = np.var(signal)

    # 计算六阶累积量
    sixth_order_cumulant = np.mean((signal - mean) ** 6) - 15 * variance * np.mean(
        (signal - mean) ** 4) + 30 * variance ** 3
    # data_centered = signal - np.mean(signal)
    # sixth_order_cumulant = np.mean(data_centered**6) - 15 * np.mean(data_centered**2) * np.mean(data_centered**4) + 30 * np.mean(signal)**2 * np.mean(data_centered**4)
    return sixth_order_cumulant


# 提取输入信号的时域特征
def time_domain_extraction(signal, features_to_extract):
    """
    提取输入信号的时域特征
    :param signal: 输入信号
    :param features_to_extract: 要提取的时域特征
    :return: 提取到的时域特征，以字典的形式返回
    """
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    if signal.ndim == 2:
        signal = signal.reshape(-1)

    # all_time_features = ['最大值', '最小值', '中位数', '峰峰值', '均值', '方差', '标准差', '峰度', '偏度', '整流平均值', '均方根',
    #                      '方根幅值', '波形因子', '峰值因子', '脉冲因子', '裕度因子', '四阶累积量', '六阶累积量']

    timeDomain_feature = {}
    for feature in features_to_extract:
        match feature:
            case '最大值':
                timeDomain_feature['最大值'] = max(signal)  # 最大值
            case '最小值':
                timeDomain_feature['最小值'] = min(signal)  # 最小值
            case '中位数':
                timeDomain_feature['中位数'] = median(signal)  # 中位数
            case '峰峰值':
                timeDomain_feature['峰峰值'] = peak_peak(signal)  # 峰峰值
            case '均值':
                timeDomain_feature['均值'] = mean(signal)  # 均值
            case '方差':
                timeDomain_feature['方差'] = variance(signal)  # 方差
            case '标准差':
                timeDomain_feature['标准差'] = std(signal)  # 标准差
            case '峰度':
                timeDomain_feature['峰度'] = kurtosis(signal)  # 峰度
            case '偏度':
                timeDomain_feature['偏度'] = skew(signal)  # 偏度
            case '整流平均值':
                timeDomain_feature['整流平均值'] = commutation_mean(signal)  # 整流平均值
            case '均方根':
                timeDomain_feature['均方根'] = root_mean_square(signal)  # 均方根
            case '方根幅值':
                timeDomain_feature['方根幅值'] = root_amplitude(signal)  # 方根幅值
            case '波形因子':
                timeDomain_feature['波形因子'] = waveform_factor(signal)  # 波形因子
            case '峰值因子':
                timeDomain_feature['峰值因子'] = peak_factor(signal)  # 峰值因子
            case '脉冲因子':
                timeDomain_feature['脉冲因子'] = pulse_factor(signal)  # 脉冲因子
            case '裕度因子':
                timeDomain_feature['裕度因子'] = margin_factor(signal)  # 裕度因子
            case '四阶累积量':
                timeDomain_feature['四阶累积量'] = fourth_order_cumulant(signal)  # 四阶累积量
            case '六阶累积量':
                timeDomain_feature['六阶累积量'] = sixth_order_cumulant(signal)  # 六阶累积量

    return timeDomain_feature


'''主要完成信号的时域特征提取, 其中各个函数的传入参数为数组形式的一维离散信号'''


# 频域相关指标
# 重心频率（Centroid Frequency）
def centroid_frequency(frequencies, power_spectrum):
    cf = (np.sum(frequencies * power_spectrum)) / (np.sum(power_spectrum))
    return cf.item()


# 均方频率
def msf_fn(frequencies, power_spectrum):
    msf = np.sum((frequencies ** 2) * power_spectrum) / (np.sum(power_spectrum))
    return msf.item()


# 均方根频率
def rmsf_fn(frequencies, power_spectrum):
    rmsf = np.sqrt((np.sum((frequencies ** 2) * power_spectrum)) / (np.sum(power_spectrum)))
    return rmsf.item()


# 频率方差
def vf_fn(frequencies, power_spectrum):
    cf = centroid_frequency(frequencies, power_spectrum)
    vf = np.sum(((frequencies - cf) ** 2) * power_spectrum) / (np.sum(power_spectrum))
    return vf.item()


# 频率标准差
def rvf_fn(frequencies, power_spectrum):
    vf = vf_fn(frequencies, power_spectrum)
    rvf = np.sqrt(vf)
    return rvf


# 谱峭度相关指标
def sk_one(frequencies, power_spectrum):
    u1 = np.sum(frequencies * power_spectrum, axis=0) / np.sum(power_spectrum, axis=0)
    u2 = np.sqrt(np.sum((frequencies - u1) ** 2 * power_spectrum, axis=0) / np.sum(power_spectrum, axis=0))
    sk = np.sum((frequencies - u1) ** 4 * power_spectrum, axis=0) / ((u2 ** 4) * np.sum(power_spectrum, axis=0))
    return sk


# 谱峭度的均值
def sk_mean(SK):
    skMean = np.mean(SK)
    return skMean


# 谱峭度的标准差
def sk_std(SK):
    skStd = np.std(SK)
    return skStd


# 谱峭度的偏值
def sk_skewness(SK):
    skSkewness = skew(SK)
    return skSkewness


# 谱峭度的峰度
def sk_kurtosis(SK):
    skKurtosis = kurtosis(SK)
    # skKurtosis = np.mean(power_spectrum ** 4) / (np.mean(power_spectrum ** 2)) ** 2 - 3
    return skKurtosis


# 频域特征的提取
def frequency_domain_extraction(signal, features_to_extract, sample_rate=25600):
    """
    提取输入数据的频域特征
    :param signal: 输入的信号
    :param features_to_extract: 要提取的频域特征
    :param sample_rate: 采样率
    :return: 提取到的频域特征，以字典的形式返回
    """
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    if signal.ndim == 2:
        signal = signal.reshape(-1)

    fft_vals = np.abs(fft(signal))

    frequencies, t, power_spectrum = scipy.signal.spectrogram(signal, sample_rate, window='hamming', nperseg=256,
                                                              noverlap=128)
    frequencies = frequencies.reshape(-1, 1)
    # SK = SKone(frequencies, power_spectrum)
    SK = kurtosis(fft_vals)
    FreDomain_feature = {}
    for feature in features_to_extract:
        match feature:
            case '重心频率':
                FreDomain_feature['重心频率'] = centroid_frequency(frequencies, power_spectrum)  # 重心频率
            case '均方频率':
                FreDomain_feature['均方频率'] = msf_fn(frequencies, power_spectrum)  # 均方频率
            case '均方根频率':
                FreDomain_feature['均方根频率'] = rmsf_fn(frequencies, power_spectrum)  # 均方根频率
            case '频率方差':
                FreDomain_feature['频率方差'] = vf_fn(frequencies, power_spectrum)  # 频率方差
            case '频率标准差':
                FreDomain_feature['频率标准差'] = rvf_fn(frequencies, power_spectrum)  # 频率标准差
            case '谱峭度的均值':
                # FreDomain_feature['谱峭度的均值'] = SKMean(SK)  # 谱峭度的均值
                FreDomain_feature['谱峭度的均值'] = np.mean(SK)
            case '谱峭度的标准差':
                # FreDomain_feature['谱峭度的标准差'] = SKStd(SK)  # 谱峭度的标准差
                FreDomain_feature['谱峭度的标准差'] = np.std(SK)
            case '谱峭度的偏度':
                # FreDomain_feature['谱峭度的偏值'] = SKSkewness(SK)  # 谱峭度的偏值
                FreDomain_feature['谱峭度的偏度'] = skew(SK)
            case '谱峭度的峰度':
                # FreDomain_feature['谱峭度的峰度'] = SKKurtosis(SK)  # 谱峭度的峰度
                FreDomain_feature['谱峭度的峰度'] = np.max(SK)

    return FreDomain_feature


# 用于提取输入数据的频域特征
def fre_dict2dict(data):
    all_frequency_features = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值',
                              '谱峭度的标准差', '谱峭度的峰度', '谱峭度的偏度']
    fre_feature = []
    for row in range(data.shape[0]):
        tmp_data = data[row, :]
        tmp_fre_feature = frequency_domain_extraction(tmp_data, all_frequency_features)
        fre_feature.append(tmp_fre_feature)
    values_list = []
    for d in fre_feature:
        # 将当前字典的所有值按顺序添加到二维列表中
        values_list.append(np.array(list(d.values())))
    fre_features_array = np.array(values_list)
    fre_features = {}
    fre_features['重心频率'] = fre_features_array[:, 0]  # 重心频率
    fre_features['均方频率'] = fre_features_array[:, 1]  # 均方频率
    fre_features['均方根频率'] = fre_features_array[:, 2]  # 均方根频率
    fre_features['频率方差'] = fre_features_array[:, 3]  # 频率方差
    fre_features['频率标准差'] = fre_features_array[:, 4]  # 频率标准差
    fre_features['谱峭度的均值'] = fre_features_array[:, 5]  # 谱峭度的均值
    fre_features['谱峭度的标准差'] = fre_features_array[:, 6]  # 谱峭度的标准差
    fre_features['谱峭度的偏值'] = fre_features_array[:, 7]  # 谱峭度的偏值
    fre_features['谱峭度的峰度'] = fre_features_array[:, 8]  # 谱峭度的峰度
    return fre_features


# 获取用于单传感器健康评估的样本特征
def get_features(data_path, time_key_list, fre_key_list):
    all_time_features = ['最大值', '最小值', '中位数', '峰峰值', '均值', '方差', '标准差', '峰度', '偏度',
                         '整流平均值', '均方根', '方根幅值', '波形因子', '峰值因子', '脉冲因子', '裕度因子',
                         '四阶累积量', '六阶累积量']
    data_all = scipy.io.loadmat(data_path)
    data_1 = data_all['stage_1']
    data_2 = data_all['stage_2']
    data_3 = data_all['stage_3']
    data_4 = data_all['stage_4']

    time_fea = {}
    time_fea[f'Stage_{1}'] = time_domain_extraction(data_1, all_time_features)
    time_fea[f'Stage_{2}'] = time_domain_extraction(data_2, all_time_features)
    time_fea[f'Stage_{3}'] = time_domain_extraction(data_3, all_time_features)
    time_fea[f'Stage_{4}'] = time_domain_extraction(data_4, all_time_features)

    fre_fea = {}
    fre_fea[f'Stage_{1}'] = fre_dict2dict(data_1)
    fre_fea[f'Stage_{2}'] = fre_dict2dict(data_2)
    fre_fea[f'Stage_{3}'] = fre_dict2dict(data_3)
    fre_fea[f'Stage_{4}'] = fre_dict2dict(data_4)

    Allstage_array_list = []
    for k in range(4):
        fre_tmp_data = fre_fea[f'Stage_{k + 1}']
        time_tmp_data = time_fea[f'Stage_{k + 1}']
        Perstage_array_list = []
        Perstage_array_list.extend([time_tmp_data[key] for key in time_key_list])
        Perstage_array_list.extend([fre_tmp_data[key] for key in fre_key_list])
        Perstage_array = np.array(Perstage_array_list).T
        Allstage_array_list.append(Perstage_array)
    Allstage_array = np.array(Allstage_array_list)

    return Allstage_array


def GetTest(test_path, time_key_list, fre_key_list):
    data = scipy.io.loadmat(test_path)
    data = data['test_data']
    time_dict = time_domain_extraction(data, time_key_list)
    fre_dict = frequency_domain_extraction(data, fre_key_list)
    Perstage_array_list = []
    Perstage_array_list.extend([time_dict[key] for key in time_key_list])
    Perstage_array_list.extend([fre_dict[key] for key in fre_key_list])

    Perstage_array = np.array(Perstage_array_list).T.astype(np.float64)
    return Perstage_array


# 按照健康评估所用到的特征对单传感器信号进行特征提取
def all_feature_extraction(signal):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    frame_length = 2048
    # step_size = 2048
    signal_length = signal.shape[0]

    num_frames = signal_length // frame_length

    print(f"num_frames: {num_frames}")

    extracted_features = []
    for i in range(num_frames):
        frame = signal[i*frame_length:(i+1)*frame_length]
        # 时域特征
        extr_feature = {}
        extr_feature['最大值'] = max(frame)  # 最大值
        extr_feature['最小值'] = min(frame)  # 最小值
        extr_feature['中位数'] = median(frame)  # 中位数
        extr_feature['峰峰值'] = peak_peak(frame)  # 峰峰值
        extr_feature['均值'] = mean(frame)  # 均值
        extr_feature['方差'] = variance(frame)  # 方差
        extr_feature['标准差'] = std(frame)  # 标准差
        extr_feature['峰度'] = kurtosis(frame, axis=0)  # 峰度
        extr_feature['偏度'] = skew(frame, axis=0)  # 偏度
        extr_feature['整流平均值'] = commutation_mean(frame)  # 整流平均值
        extr_feature['均方根'] = root_mean_square(frame)  # 均方根
        extr_feature['方根幅值'] = root_amplitude(frame)  # 方根幅值
        extr_feature['波形因子'] = waveform_factor(frame)  # 波形因子
        extr_feature['峰值因子'] = peak_factor(frame)  # 峰值因子
        extr_feature['脉冲因子'] = pulse_factor(frame)  # 脉冲因子
        extr_feature['裕度因子'] = margin_factor(frame)  # 裕度因子
        extr_feature['四阶累积量'] = fourth_order_cumulant(frame)  # 四阶累积量
        extr_feature['六阶累积量'] = sixth_order_cumulant(frame)  # 六阶累积量

        # 频域特征的提取
        def fre_domain_extraction(signal, sample_rate=25600):
            if not isinstance(signal, np.ndarray):
                signal = np.array(signal)

            # if signal.ndim == 2:
            #     signal = signal.reshape(-1)

            frequencies, t, power_spectrum = scipy.signal.spectrogram(signal, sample_rate, window='hamming', nperseg=256,
                                                                      noverlap=128)

            # print(f"frequencies : {frequencies}")
            frequencies = frequencies.reshape(-1, 1)

            SK = sk_one(frequencies, power_spectrum)
            FreDomain_feature = {}
            FreDomain_feature['重心频率'] = centroid_frequency(frequencies, power_spectrum)  # 重心频率
            FreDomain_feature['均方频率'] = msf_fn(frequencies, power_spectrum)  # 均方频率
            FreDomain_feature['均方根频率'] = rmsf_fn(frequencies, power_spectrum)  # 均方根频率
            FreDomain_feature['频率方差'] = vf_fn(frequencies, power_spectrum)  # 频率方差
            FreDomain_feature['频率标准差'] = rvf_fn(frequencies, power_spectrum)  # 频率标准差
            FreDomain_feature['谱峭度的均值'] = sk_mean(SK)  # 谱峭度的均值
            FreDomain_feature['谱峭度的标准差'] = sk_std(SK)  # 谱峭度的标准差
            FreDomain_feature['谱峭度的偏值'] = sk_skewness(SK)  # 谱峭度的偏值
            FreDomain_feature['谱峭度的峰度'] = sk_kurtosis(SK)  # 谱峭度的峰度

            return FreDomain_feature

        fre_feature = []
        # for row in range(signal.shape[0]):
        #     tmp_data = signal[row, :]
        #     tmp_fre_feature = fre_domain_extraction(tmp_data)
        #     fre_feature.append(tmp_fre_feature)
        fre_feature.append(fre_domain_extraction(frame))
        values_list = []
        for d in fre_feature:
            # 将当前字典的所有值按顺序添加到二维列表中
            values_list.append(np.array(list(d.values())))
        fre_features_array = np.array(values_list)
        print(f"FreDomain_feature: {fre_features_array.shape}")
        extr_feature['重心频率'] = fre_features_array[:, 0]  # 重心频率
        extr_feature['均方频率'] = fre_features_array[:, 1]  # 均方频率
        extr_feature['均方根频率'] = fre_features_array[:, 2]  # 均方根频率
        extr_feature['频率方差'] = fre_features_array[:, 3]  # 频率方差
        extr_feature['频率标准差'] = fre_features_array[:, 4]  # 频率标准差
        extr_feature['谱峭度的均值'] = fre_features_array[:, 5]  # 谱峭度的均值
        extr_feature['谱峭度的标准差'] = fre_features_array[:, 6]  # 谱峭度的标准差
        extr_feature['谱峭度的偏值'] = fre_features_array[:, 7]  # 谱峭度的偏值
        extr_feature['谱峭度的峰度'] = fre_features_array[:, 8]  # 谱峭度的峰度

        extracted_features.append(extr_feature)

    print(f"examples_shape: {len(extracted_features)}")

    return extracted_features


# 获取用于多传感器健康评估的样本特征
def get_multiple_sensors_example(data_all, sensor1_key_list, sensor2_key_list, sensor3_key_list):
    # 根据不同的传感器进行特征提取，以作为多传感器健康评估算法的输入
    shape = data_all.shape
    if shape[0] < shape[1]:
        data_all = data_all.T
    data_sensor1 = data_all[:, 0].flatten()
    data_sensor2 = data_all[:, 3].flatten()
    data_sensor3 = data_all[:, 6].flatten()

    sensor1_dict = all_feature_extraction(data_sensor1)
    sensor2_dict = all_feature_extraction(data_sensor2)
    sensor3_dict = all_feature_extraction(data_sensor3)

    print(f"sensor1_key_list: {sensor1_key_list}")
    print(f"sensor2_key_list: {sensor2_key_list}")
    print(f"sensor3_key_list: {sensor3_key_list}")

    num_examples = len(sensor1_dict)
    Perstage_array_list = []

    def get_array_value(x):
        if isinstance(x, np.ndarray):
            return x.item()
        else:
            return x

    for i in range(num_examples):
        tmp_array_list = []

        tmp_array_list.extend([get_array_value(sensor1_dict[i][key]) for key in sensor1_key_list])
        tmp_array_list.extend([get_array_value(sensor2_dict[i][key]) for key in sensor2_key_list])
        tmp_array_list.extend([get_array_value(sensor3_dict[i][key]) for key in sensor3_key_list])

        Perstage_array_list.append(tmp_array_list)

    print(f"Perstage_array_list: {Perstage_array_list}")

    # for index, item in enumerate(Perstage_array_list):
    #     if not isinstance(item, np.ndarray):
    #         new_item = np.array(item)
    #         print(f"item_{index}: {new_item}")
    #         Perstage_array_list[index] = new_item
    # 最终用于多传感器健康评估算法的样本特征
    Perstage_array = np.array(Perstage_array_list).astype(np.float64)

    print(f"----Perstage_array: {Perstage_array}")

    return Perstage_array
