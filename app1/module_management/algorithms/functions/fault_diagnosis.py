import pickle
import subprocess

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import json

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from app1.module_management.algorithms.models.model_ulcnn import ULCNN
# from app1.module_management.algorithms.models.Simplemodel import SimModel, SimModel_2
# from app1.module_management.algorithms.functions.load_data import load_data
# from app1.module_management.algorithms.models.gru_lstm import GRUModel, LSTM

from app1.module_management.algorithms.models.model_ulcnn import ULCNN
from app1.module_management.algorithms.models.Simplemodel import SimModelSingle, SimModelMultiple
from app1.module_management.algorithms.functions.load_data import load_data
from app1.module_management.algorithms.models.gru_lstm import GRUModel, LSTM
from app1.module_management.algorithms.models import mutli_sensor_1, mutli_sensor_2, mutli_sensor_3, mutli_sensor_4, \
    single_sensor_1, single_sensor_2, single_sensor_3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
train_data = pd.read_csv('app1/module_management/algorithms/functions/datas/vibration_features_with_labels.csv')

choose_features_eng = ['std', 'rms', 'var', 'rectified_mean', 'root_amplitude', 'peak_to_peak', 'cumulant_6th', 'mean',
                        'cumulant_4th', 'min']


choose_features = ['标准差', '均方根', '方差', '整流平均值', '方根幅值', '峰峰值', '六阶累积量', '均值',
                   '四阶累积量', '最小值']
choose_features_multiple = ['X维力(N)_六阶累积量', 'X维力(N)_峰峰值', 'X维力(N)_重心频率', 'X维力(N)_最大值', 'X维力(N)_四阶累积量',
                            'X维力(N)_方差', 'X维力(N)_裕度因子', 'X维力(N)_标准差', 'X维力(N)_均方根', 'X维力(N)_方根幅值']


# 解决提取特征中英文名称不一致的问题
def replace_features_name(data: pd.DataFrame) -> pd.DataFrame:
    """
    将中文的特征替换为英文形式的特征名
    :param data: 输入的数据（提取到的特征）
    :return: 将特征名称替换后的数据
    """
    # features_in_english = ['mean', 'var', 'std', 'skewness', 'kurtosis', 'cumulant_4th', 'cumulant_6th', 'max',
    # 'min', 'median', 'peak_to_peak', 'rectified_mean', 'rms', 'root_amplitude', 'waveform_factor', 'peak_factor',
    # 'impulse_factor', 'margin_factor'] + ['centroid_freq', 'msf', 'rms_freq', 'freq_variance', 'freq_std',
    # 'spectral_kurt_mean', 'spectral_kurt_std', 'spectral_kurt_peak', 'spectral_kurt_skew']
    names_mapping = {'均值': 'mean', '方差': 'var', '标准差': 'std', '偏度': 'skewness', '峰度': 'kurtosis',
                     '四阶累积量': 'cumulant_4th', '六阶累积量': 'cumulant_6th', '最大值': 'max', '最小值': 'min',
                     '中位数': 'median', '峰峰值': 'peak_to_peak', '整流平均值': 'rectified_mean', '均方根': 'rms',
                     '方根幅值': 'root_amplitude', '波形因子': 'waveform_factor', '峰值因子': 'peak_factor',
                     '脉冲因子': 'impulse_factor', '裕度因子': 'margin_factor', '重心频率': 'centroid_freq',
                     '均方频率': 'msf', '均方根频率': 'rms_freq', '频率方差': 'freq_variance',
                     '频率标准差': 'freq_std', '谱峭度的均值': 'spectral_kurt_mean',
                     '谱峭度的标准值': 'spectral_kurt_std',
                     '谱峭度的峰度': 'spectral_kurt_peak',
                     '谱峭度的偏度': 'spectral_kurt_skew'}

    data.rename(columns=names_mapping, inplace=True)
    return data


# SVM故障诊断
def diagnose_with_svc_model(data_with_selected_features, multiple_sensor=False, user_dir=None):
    """
    使用SMV的故障诊断
    :param user_dir: 故障诊断结果的保存目录
    :param multiple_sensor: 是否为多传感器数据
    :param data_with_selected_features: 输入样本以及选择的特征
    :return: “0”代表无故障，“1”代表有故障
    """
    example = data_with_selected_features.get('extracted_features').copy()
    selected_features = data_with_selected_features.get('features_name')
    example_filepath = data_with_selected_features.get('filepath')
    # example = replace_features_name(example)
    try:
        predictions = []
        num_examples = example.shape[0]
        num_has_fault = 0  # 记录有故障的样本的数量
        x_axis = []  # 横坐标，即样本的索引
        for i in range(num_examples):
            temp_example = example[i:i+1]
            if not multiple_sensor:
                # 树模型没有标准化，svc（支持向量机）有
                scaler = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/scaler_2.pkl')
                # 使用训练阶段保存的 StandardScaler 对测试数据进行同样的变换
                # train_data[choose_features] = scaler.transform(train_data[choose_features])
                temp_example[choose_features] = scaler.transform(temp_example[choose_features])
                # 预测结果为“0”代表无故障，“1”代表有故障
                svc_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/svc_model_2.pkl')
                svc_prediction = svc_model.predict(temp_example[choose_features][0:1])[0]
            else:
                scaler = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/mutli_scaler.pkl')
                temp_example[choose_features_multiple] = scaler.transform(temp_example[choose_features_multiple])

                svc_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/mutli_svc_model.pkl')
                svc_prediction = svc_model.predict(temp_example[choose_features_multiple][0:1])[0]
            if svc_prediction == 1:
                num_has_fault += 1
                predictions.append(1)
                x_axis.append(f"样本{i+1}（有故障）")
            else:
                predictions.append(0)
                x_axis.append(f"样本{i+1}（无故障）")

        indicator = {}
        # 将example按列划分保存到indicator中，其中indicator中的key为列名，value为该列的值
        scaler = StandardScaler()
        for col in selected_features:
            # 将列转换为二维数组，因为 MinMaxScaler 需要二维输入
            column_data = example[[col]].values
            # 进行归一化
            scaled_column = scaler.fit_transform(column_data)
            # 精确到小数点后三位
            indicator[col] = [round(num, 3) for num in scaled_column.flatten()]
            # indicator[col] = example[col].to_list()
            # 将归一化后的结果转换为列表并存储到 indicator 中
            indicator[col] = scaled_column.flatten().tolist()
        # 补充的故障诊断结果
        save_path_of_complementary_result = 'app1/module_management/algorithms/functions/fault_diagnosis/' + user_dir
        if not os.path.exists(save_path_of_complementary_result):
            os.makedirs(save_path_of_complementary_result)
        complementary_figure, complementary_summary = complementary_result_of_fault_diagnosis(predictions,
                                                                                              save_path_of_complementary_result)
        num_has_no_fault = num_examples - num_has_fault
        figure_path = plot_diagnosis(example_filepath, multiple_sensor, user_dir=user_dir)

        return indicator, x_axis, num_has_fault, num_has_no_fault, figure_path, complementary_figure, complementary_summary
    except Exception as e:
        print("svm故障诊断模块出现异常, ", str(e))
        return None, e, 0, 0, None, None, None


# 随机森林的故障诊断
def diagnose_with_random_forest_model(data_with_selected_features, multiple_sensor=False, user_dir=None):
    """
    随机森林故障诊断
    :param user_dir: 故障诊断结果的保存目录
    :param multiple_sensor: 是否为多传感器数据
    :param data_with_selected_features: 输入样本以及选择的特征
    :return: indicator, x_axis, num_has_fault, num_has_not_fault, figure_path
    """
    example = data_with_selected_features.get('extracted_features').copy()
    selected_features = data_with_selected_features.get('features_name')
    print(f'selected features: {selected_features}')
    print(f'example: {example}')
    print(f'example.shape: {example.shape}')
    example_filepath = data_with_selected_features.get('filepath')
    # example = replace_features_name(example)
    # 根据提取的特征进行故障诊断，对连续数据切分得到的每个样本都进行预测
    try:
        predictions = []
        num_examples = example.shape[0]
        num_has_fault = 0  # 记录有故障的样本的数量
        x_axis = []  # 横坐标，即样本的索引
        for i in range(num_examples):
            if not multiple_sensor:
                # 单传感器的模型预测
                random_forest_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/random_forest'
                                                  '/random_forest_model_2.pkl')
                random_forest_predictions = random_forest_model.predict(example[choose_features][i:i+1])
            else:
                # 多传感器的模型预测
                random_forest_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/random_forest'
                                                  '/mutli_random_forest_model.pkl')
                random_forest_predictions = random_forest_model.predict(example[choose_features_multiple][i:i+1])
            print(f'random_forest_predictions: {random_forest_predictions}')
            # 统计有故障的样本的数量，同时记录下有故障样本的索引
            if random_forest_predictions[0] == 1:
                num_has_fault += 1
                predictions.append(1)
                x_axis.append(f'样本{i+1}（有故障）')
                print(f'有故障的样本索引：{i+1}')
            else:
                x_axis.append(f'样本{i+1}（无故障）')
                predictions.append(0)
                print(f'无故障的样本索引：{i+1}')
            # predictions.append(random_forest_predictions[0])

        # 补充的故障诊断结果
        save_path_of_complementary_result = 'app1/module_management/algorithms/functions/fault_diagnosis/' + user_dir
        if not os.path.exists(save_path_of_complementary_result):
            os.makedirs(save_path_of_complementary_result)
        complementary_figure, complementary_summary = complementary_result_of_fault_diagnosis(predictions, save_path_of_complementary_result)

        num_has_not_fault = num_examples - num_has_fault
        indicator = {}

        # 将example按列划分保存到indicator中，其中indicator中的key为列名，value为该列的值
        scaler = StandardScaler()
        for col in selected_features:
            # 将列转换为二维数组，因为 MinMaxScaler 需要二维输入
            column_data = example[[col]].values
            # 进行归一化
            scaled_column = scaler.fit_transform(column_data)
            # 精确到小数点后三位
            indicator[col] = [round(num, 3) for num in scaled_column.flatten()]
            # indicator[col] = example[col].to_list()
            # 将归一化后的结果转换为列表并存储到 indicator 中
            indicator[col] = scaled_column.flatten().tolist()

        figure_path = plot_diagnosis(example_filepath, multiple_sensor, user_dir=user_dir)
        return indicator, x_axis, num_has_fault, num_has_not_fault, figure_path, complementary_figure, complementary_summary
    except Exception as e:
        print("随机森林故障诊断模块出现异常, ", str(e))
        return None, e, 0, 0, None, None, None


# 补充的故障诊断的结果图
def complementary_result_of_fault_diagnosis(predictions_array, save_path):
    """

    :param predictions_array: 预测的结果数组
    :param save_path: 结果保存路径
    :return: 结果图片保存路径，总结文本
    """
    matplotlib.use('Agg')
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
    plt.rcParams['font.size'] = 15

    # 创建一个示例数据序列（大部分为0，少部分为1），本序列用以模拟故障诊断算法结果（以正常/故障二分类为例，正常为0，故障为1）
    # predictions_array = np.random.choice([0, 1], size=100, p=[0.9, 0.1])  # 90% 为0，10% 为1
    if not isinstance(predictions_array, np.ndarray):
        predictions_array = np.array(predictions_array)

    # 查找值为1的索引
    fault_indices = np.where(predictions_array == 1)[0]

    # 添加具体故障样本信息，本变量将作为故障诊断的结果生成到报告中
    summary_line = (
        f" {len(predictions_array)} 个测试样本中，总共有 {len(fault_indices)} 个故障样本，"
        f"分别为第 {', '.join(map(str, fault_indices))} 个样本\n"
    )

    summary_save_path = os.path.join(save_path, 'summary.txt')
    # 将具体故障样本信息写入到 index.txt 文件
    with open(summary_save_path, 'w') as f:
        f.writelines(summary_line)

    # 打印到控制台确认
    # print("故障样本标签已保存到 index.txt 文件中。")
    # print(summary_line)

    # 创建横坐标序列
    x = np.arange(len(predictions_array))

    # 创建散点图
    plt.figure(figsize=(8, 6))

    # 绘制数据为1的点，红色圆圈
    plt.scatter(x[predictions_array == 1]+1, predictions_array[predictions_array == 1], color='red', marker='x', label='故障', s=100)

    # 绘制数据为0的点，黑色叉号
    plt.scatter(x[predictions_array == 0]+1, predictions_array[predictions_array == 0], color='green', marker='o', label='正常', s=50)

    # 设置纵坐标范围
    plt.ylim(-0.1, 1.1)

    # 隐藏纵坐标数值
    plt.yticks([])

    # 设置横坐标标签
    plt.ylabel('诊断结果', fontsize=20)
    plt.xlabel('样本序号', fontsize=20)

    # 不设置标题
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=15)

    # 显示图例
    plt.legend(fontsize=20)

    figure_save_path = os.path.join(save_path, 'complementary.png')

    print(f"..........Saving figure to {figure_save_path}")
    # 保存图片到本地
    plt.savefig(figure_save_path, dpi=300, bbox_inches='tight')  # 保存图片为高质量 PNG 格式

    return figure_save_path, summary_line[0]


# 故障预测（包含多传感器与单传感器）
def time_regression(data_stream, private_algorithm=None):
    """
    可能出现故障的时间预测
    :param private_algorithm: 不为None时表示使用用户上传的私有故障预测算法
    :param data_stream: 传入的数据流
    :return: 此前故障诊断的结果，以及预测结果图像的存放路径
    """
    example: pandas.DataFrame = data_stream.get('extracted_features').copy()  # 用于故障预测的样本数据
    raw_data_filepath = data_stream.get('filepath')
    user_dir = data_stream.get('user_dir')
    multiple_sensor = data_stream.get('multiple_sensor')
    have_fault = data_stream.get('diagnosis_result')
    # 没有故障预测故障时间
    try:
        if have_fault == 0:
            if private_algorithm is not None:
                # 使用用户私有算法进行故障预测
                # 存放私用故障检测算法的文件目录
                base_dir_of_algorithm = 'app1/module_management/algorithms/models/private_fault_prediction'
                # 用户私有算法目录
                username = user_dir.split('/')[0]
                user_private_dir = base_dir_of_algorithm + '/' + username
                if not os.path.exists(user_private_dir):
                    os.makedirs(user_private_dir)

                # private_algorithm_filepath = user_private_dir + '/' + private_algorithm + '.py'  # 用户私有故障预测算法
                # private_model_filepath = user_private_dir + '/' + private_algorithm + '.pkl'  # 用户私有故障预测模型
                # 通过文件读取的形式向用户私有故障预测算法源文件
                input_filepath = user_private_dir + '/intermediate_data.pkl'   # 输入数据的文件路径
                example.to_pickle(input_filepath)  # 将样本保存到输入数据的文件路径

                # 以shell脚本的形式运行用户私有的故障预测算法
                result = subprocess.run(shell=True, capture_output=True,
                                        args=f"cd {user_private_dir} & python ./{private_algorithm}.py --input"
                                             f"-filepath ./intermediate_data.pkl --model-filepath ./{private_algorithm}.pkl")
                prediction = result.stdout.decode('utf-8').split('#')[0]
                print(f'prediction: {prediction}')
                time_to_fault = float(prediction)
                print(f'time_to_fault: {time_to_fault}')
            else:
                # 使用系统中集成的线性回归算法进行故障预测
                if multiple_sensor:
                    # 多传感器预测
                    reg_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/regression'
                                            '/mutli_time_reg.pkl')
                    time_to_fault = reg_model.predict(example[choose_features_multiple][0:1])
                else:
                    # 单传感器预测
                    example = replace_features_name(example)
                    # print('example: ', example.columns())
                    reg_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/regression'
                                            '/single_time_reg.pkl')

                    time_to_fault = reg_model.predict(example[choose_features_eng][0:1])
        # 存在故障时不用预测
        else:
            time_to_fault = [0]

        # 绘制信号图像
        figure_path = plot_diagnosis(raw_data_filepath, multiple_sensor, user_dir=user_dir)

        return time_to_fault[0], figure_path
    except Exception as e:
        return None, e


# 定义GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hx=None):
        # 如果提供了隐藏状态，则使用它；否则自动初始化
        out, h_n = self.gru(x, hx)
        # 取最后一个时间步的输出用于分类
        out = self.fc(out[:, -1, :])
        return out, h_n


# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hx=None):
        # 如果提供了隐藏状态，则使用它；否则自动初始化
        out, (h_n, c_n) = self.lstm(x, hx)
        # 取最后一个时间步的输出用于分类
        out = self.fc(out[:, -1, :])
        return out, (h_n, c_n)


# 模型基本参数
input_size = 2048  # 特征维度，假设为2048
hidden_size = 128  # 隐藏层大小
num_layers = 2  # GRU层数
num_classes = 2  # 类别数
LSTM_weights_path = 'app1/module_management/algorithms/models/fault_diagnosis/lstm/LSTM_model.pth'
GRU_weights_path = 'app1/module_management/algorithms/models/fault_diagnosis/gru/GRU_model.pth'

# 模型初始化
LSTM_model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
GRU_model = GRUClassifier(input_size, hidden_size, num_layers, num_classes)

if torch.cuda.is_available():
    state_dict1 = torch.load(GRU_weights_path)
    state_dict2 = torch.load(LSTM_weights_path)
else:
    state_dict1 = torch.load(GRU_weights_path, map_location=torch.device('cpu'))
    state_dict2 = torch.load(LSTM_weights_path, map_location=torch.device('cpu'))

# 使用模型实例加载状态字典
GRU_model.load_state_dict(state_dict1)
LSTM_model.load_state_dict(state_dict2)
# 将模型移动到适当的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRU_model = GRU_model.to(device)
LSTM_model = LSTM_model.to(device)

# 设置为评估模式
GRU_model.eval()
LSTM_model.eval()


# 深度学习模型GRU的故障诊断
def diagnose_with_gru_model(data_with_selected_features, multiple_sensor=False):
    """
    以gru模型进行故障诊断
    :param multiple_sensor: 是否为多传感器数据
    :param data_with_selected_features: 提取到的特征
    :return: integer "0" or "1"
    """
    examples = data_with_selected_features.get('raw_data')
    example_filepath = data_with_selected_features.get('filepath')
    user_dir = data_with_selected_features.get('user_dir')

    dims = len(examples.shape)
    shape = examples.shape
    # 数据形状为L*N, L为信号的长度，N为传感器的数量
    if shape[0] < shape[1]:
        examples = examples.T
    print(f'examples.shape: ', examples.shape)
    # 将数据按照2048的长度进行样本划分，不足的部分丢弃
    num_examples = int(examples.shape[0] / 2048)

    print(f'num_examples: {num_examples}')
    # examples = examples[:num_examples * 2048]

    num_has_fault = 0
    predictions = []
    try:
        with torch.no_grad():
            for i in range(num_examples):
                example = examples[i*2048:(i+1)*2048]
                print(f'example.shape: ', example.shape)
                if not multiple_sensor:
                    GRU_model.eval()
                    # 单传感器算法
                    # if dims == 2:
                    #     outputs, h_n = GRU_model(torch.tensor(example[0, :]).reshape(1, 1, -1).to(device))
                    # else:
                    #     outputs, h_n = GRU_model(torch.tensor(example).reshape(1, 1, -1).to(device))
                    outputs, h_n = GRU_model(torch.tensor(example).type(torch.float32).reshape(1, 1, -1).to(device))
                    _, predicted = torch.max(outputs, 1)
                    prediction = predicted.tolist()[0]
                else:
                    # 多传感器算法
                    gru_model = GRUModel(input_size=7, hidden_size=64, num_layers=4, output_size=1).to(device)
                    gru_model.load_state_dict(torch.load('app1/module_management/algorithms/models/fault_diagnosis'
                                                         '/gru/GRU_model_multiple_sensor.pth', map_location=device))
                    gru_model.eval()
                    inputs = torch.from_numpy(example).type(torch.FloatTensor).reshape((-1, 2048, 7)).to(device)
                    predicted = gru_model.forward(inputs)
                    # 根据阈值判断有无故障
                    if predicted.reshape(-1).item() < 0:
                        prediction = 0
                    else:
                        prediction = 1
                if prediction == 1:
                    print(f"样本{i + 1}，有故障")
                    num_has_fault += 1
                else:
                    print(f"样本{i + 1}，无故障")
                predictions.append(prediction)

        # 补充的故障诊断结果
        save_path_of_complementary_result = 'app1/module_management/algorithms/functions/fault_diagnosis/' + user_dir
        if not os.path.exists(save_path_of_complementary_result):
            os.makedirs(save_path_of_complementary_result)
        complementary_figure, complementary_summary = complementary_result_of_fault_diagnosis(predictions,
                                                                                                      save_path_of_complementary_result)
        num_has_no_fault = num_examples - num_has_fault
        figure_path = plot_diagnosis(example_filepath, multiple_sensor, user_dir=user_dir)
        return None, None, num_has_fault, num_has_no_fault, figure_path, complementary_figure, complementary_summary
    except Exception as e:
        print('gru故障诊断出现异常, ', str(e))
        return None, e, 0, 0, None, None, None


# 深度学习模型LSTM的故障诊断
def diagnose_with_lstm_model(data_with_selected_features, multiple_sensor=False):
    """
    以lstm模型进行故障诊断
    :param multiple_sensor: 是否为多传感器数据
    :param data_with_selected_features: 提取到的特征
    :return: 故障诊断结果："0" 代表无故障 ，"1"代表有故障
    """
    examples = data_with_selected_features.get('raw_data')
    example_filepath = data_with_selected_features.get('filepath')
    user_dir = data_with_selected_features.get('user_dir')
    if examples.shape[0] < examples.shape[1]:
        examples = examples.T
    num_examples = int(examples.shape[0] / 2048)
    num_has_fault = 0
    predictions = []
    try:
        # 单传感器算法
        with torch.no_grad():
            for i in range(num_examples):
                example = examples[i*2048:(i+1)*2048]
                if not multiple_sensor:
                    # 单传感器的lstm模型
                    LSTM_model.eval()
                    # if dims == 2:
                    # else:
                    #     outputs, h_n = LSTM_model(torch.tensor(example, dtype=torch.float32).reshape(1, 1, -1).to(device))
                    outputs, h_n = LSTM_model(torch.tensor(example, dtype=torch.float32).reshape(1, 1, -1).to(device))

                    _, predicted = torch.max(outputs, 1)
                    prediction = predicted.tolist()[0]
                else:
                    # 多传感器的lstm模型
                    lstm_model = LSTM(input_size=7, hidden_size=64, num_layers=4, output_size=1).to(device)
                    lstm_model.load_state_dict(torch.load(
                        'app1/module_management/algorithms/models/fault_diagnosis/lstm/LSTM_model_mutiple_sensor.pth', map_location=device))
                    lstm_model.eval()
                    inputs = torch.from_numpy(example).type(torch.FloatTensor).reshape((-1, 2048, 7)).to(device)
                    predicted = lstm_model.forward(inputs)
                    # 根据阈值判断有无故障
                    if predicted.reshape(-1).item() < 0:
                        prediction = 0
                    else:
                        prediction = 1
                if prediction == 1:
                    print(f"样本{i + 1}，有故障")
                    num_has_fault += 1
                else:
                    print(f"样本{i + 1}，无故障")

        # 补充的故障诊断结果
        save_path_of_complementary_result = 'app1/module_management/algorithms/functions/fault_diagnosis/' + user_dir
        if not os.path.exists(save_path_of_complementary_result):
            os.makedirs(save_path_of_complementary_result)
        complementary_figure, complementary_summary = complementary_result_of_fault_diagnosis(predictions,
                                                                                              save_path_of_complementary_result)
        num_has_no_fault = num_examples - num_has_fault
        figure_path = plot_diagnosis(example_filepath, multiple_sensor, user_dir=user_dir)
        return None, None, num_has_fault, num_has_no_fault, figure_path, complementary_figure, complementary_summary
    except Exception as e:
        return None, e, 0, 0, None, None, None


# 一维卷积模型的故障诊断
def diagnose_with_ulcnn(datastream, multiple_sensor=False):
    """
    一维卷积模型的故障诊断
    :param multiple_sensor: 是否为多传感器数据
    :param datastream: 提取到的特征
    :return: 故障诊断结果："0" 代表无故障 ，"1"代表有故障
    """
    examples = datastream.get('raw_data')  # 原始振动信号
    example_filepath = datastream.get('filepath')  # 原始信号的来源文件
    user_dir = datastream.get('user_dir')  # 用户目录

    if examples.shape[0] < examples.shape[1]:
        examples = examples.T
    num_examples = int(examples.shape[0] / 2048)
    num_has_fault = 0
    predictions = []
    try:
        for i in range(num_examples):
            example = examples[i * 2048:(i + 1)]
            if not multiple_sensor:
                # 单传感器的算法
                example = example.flatten()
                input_data = torch.from_numpy(example).type(torch.FloatTensor).reshape(1, 1, 2048).to(device)

                input_data = input_data.reshape(-1, 1, 2048)
                # 加载模型
                model = ULCNN(num_classes=2, feature_dim=1, num_layers=6, encoder_step=32, conv_kernel_size=3).to(device)
                model.load_state_dict(torch.load('app1/module_management/algorithms/models/fault_diagnosis/ulcnn'
                                                 '/ulcnn_single_sensor_accuracy1.0.pth', map_location=device))
                # 预测
                model.eval()
                pre = model(input_data)

                # 根据结果判断是否具有故障
                result = pre.argmax().item()
            else:
                input_data = torch.from_numpy(example).type(torch.FloatTensor).mT.unsqueeze(dim=0).to(device)
                # 加载模型
                model = ULCNN(num_classes=2, feature_dim=7, num_layers=6, encoder_step=32, conv_kernel_size=3).to(device)
                model.load_state_dict(torch.load('app1/module_management/algorithms/models/fault_diagnosis/ulcnn'
                                                 '/ulcnn_model_loss_3.7562787532806396.pth', map_location=device))
                # 预测
                model.eval()
                pre = model(input_data)
                # 根据结果判断有无故障
                # print(f'prediction: {pre}')
                result = pre.argmax().item()
            if result == 1:
                print(f"样本{i + 1}，有故障")
                num_has_fault += 1
                predictions.append(1)
            else:
                predictions.append(0)
                print(f"样本{i + 1}，无故障")
                # 补充的故障诊断结果
        save_path_of_complementary_result = 'app1/module_management/algorithms/functions/fault_diagnosis/' + user_dir
        if not os.path.exists(save_path_of_complementary_result):
            os.makedirs(save_path_of_complementary_result)
        complementary_figure, complementary_summary = complementary_result_of_fault_diagnosis(predictions,
                                                                                              save_path_of_complementary_result)
        num_has_no_fault = num_examples - num_has_fault
        figure_path = plot_diagnosis(example_filepath, multiple_sensor, user_dir=user_dir)
        return None, None, num_has_fault, num_has_no_fault, figure_path, complementary_figure, complementary_summary
    except Exception as e:
        return None, e, 0, 0, None, None, None


# 生成频谱图
def generate_mel_spectrum(signal, sample_rate):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=400,
        hop_length=200,
        window_fn=torch.hamming_window,
        n_mels=40
    )

    # 应用MelSpectrogram转换
    mel_ = mel_spec(signal)

    # 由于mel_spec是一个三维张量 (n_mels, n_frames, 1)，需要对其进行调整以便绘图
    # 去掉最后一个维度
    mel_ = mel_.squeeze(-1)

    # 转换到对数刻度（通常是dB），因为频谱值的范围可能非常大
    result = 20 * torch.log10(torch.clamp(mel_, min=1e-9))  # 防止log(0)
    return result


# 绘制信号的时频图
def spectrum_figure(example_filepath, multiple_sensor=False, num_sensor=7, user_dir=None):
    """
    绘制信号的时频图
    :param user_dir: 将结果图像保存到的用户目录
    :param example_filepath: 输入的信号
    :param num_sensor: 传感器的数量
    :param multiple_sensor: 是否为多传感器的数据
    :return: 绘制的时频图的保存路径
    """

    signal, filename = load_data(example_filepath)
    signal = torch.from_numpy(signal).reshape(2048, -1).type(torch.FloatTensor)
    matplotlib.use('Agg')
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
    plt.rcParams['font.size'] = 15

    # log_mel_spec = spectrum(audio_signal)
    # 使用matplotlib绘制
    # plt.imshow(log_mel_spec.numpy(), origin='lower', aspect='auto', interpolation='nearest', cmap='viridis')
    # plt.colorbar(format='%+2.0f dB')
    # plt.ylabel('Mel Frequency Bins')
    # plt.xlabel('Time [s]')
    # plt.title('Mel Spectrogram')
    # plt.tight_layout()

    if not multiple_sensor:
        plt.figure(figsize=(16, 8))
        # 绘制图像
        log_mel_spec = generate_mel_spectrum(signal.flatten(), sample_rate=25600)   # 生成频谱图
        plt.imshow(log_mel_spec.numpy(), origin='lower', aspect='auto', interpolation='nearest', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.ylabel('频率',)
        plt.xlabel('时间',)
        plt.title('信号时频图')
    else:
        plt.figure(figsize=(20, 15))
        # plt.figure(figsize=(20, 10))  # 设置图形的大小
        sensor_names = ['X维力(N)', 'Y维力(N)', 'Z维力(N)', 'X维振动(g)', 'Y维振动(g)', 'Z维振动(g)', 'AE-RMS (V)']
        # sensor_names = ['传感器1', '传感器2', '传感器3', '传感器4', '传感器5', '传感器6', '传感器7']

        num_sensors = signal.shape[1]  # 获取传感器的数量
        for i, sensor_name in enumerate(sensor_names):
            log_mel_spec = generate_mel_spectrum(signal[:, i].flatten(), sample_rate=50000)  # 生成频谱图
            plt.subplot(num_sensors, 1, i + 1)  # 创建子图，num_sensors 行 1 列，当前是第 i+1 个子图
            plt.imshow(log_mel_spec.numpy(), origin='lower', aspect='auto', interpolation='nearest', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'传感器{i + 1}-{sensor_name}')  # 设置子图的标题
            plt.xlabel('时间', )  # 设置 x 轴标签
            plt.ylabel('频率', )  # 设置 y 轴标签
            if i + 1 >= num_sensor:
                break
        # 调整子图之间的间距
        plt.tight_layout()
    # 保存图像
    save_path_dir = 'app1/module_management/algorithms/functions/fault_diagnosis/simModel/' + user_dir
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    save_path = save_path_dir + '/' + f'{filename}_diagnosis.png'
    plt.savefig(save_path)

    return save_path


# 依据时频图的模型的故障诊断
def diagnose_with_simmodel(datastream, multiple_sensor=False):
    """
    依据时频图的模型的故障诊断
    :param multiple_sensor: 是否为多传感器数据
    :param datastream: 提取到的特征
    :return: 故障诊断结果："0" 代表无故障 ，"1"代表有故障
    """
    examples = datastream.get('raw_data')
    example_filepath = datastream.get('filepath')
    user_dir = datastream.get('user_dir')

    if examples.shape[0] < examples.shape[1]:
        examples = examples.T
    num_examples = int(examples.shape[0] / 2048)
    num_has_fault = 0
    predictions = []
    # 单传感器的算法
    try:
        for i in range(num_examples):
            example = examples[i * 2048:(i + 1) * 2048]
            if not multiple_sensor:
                # 单传感器的时频图卷积模型的故障诊断
                example = example.flatten()
                input_data = torch.from_numpy(example).type(torch.FloatTensor).reshape(1, 2048).to(device)

                # 加载模型
                model = SimModelSingle().to(device)
                model.load_state_dict(torch.load('app1/module_management/algorithms/models/fault_diagnosis/simModel'
                                                 '/sim_model_single_sensor.pth', map_location=device))
                # 预测
                model.eval()
                pre = model(input_data)

                # 根据结果判断是否具有故障
                result = pre.argmax().item()
            else:
                # 多传感器的时频图卷积模型的故障诊断
                input_data = torch.from_numpy(example).type(torch.cuda.FloatTensor).unsqueeze(dim=0).mT

                # 加载模型
                model = SimModelMultiple().to(device)
                model.load_state_dict(torch.load('app1/module_management/algorithms/models/fault_diagnosis/simModel'
                                                 '/sim_model_0.05694399029016495.pth', map_location=device))
                # 预测
                model.eval()
                pre = model(input_data)
                # 根据推理结果判断是否有故障
                result = pre.argmax().item()
            if result == 1:
                num_has_fault += 1
                predictions.append(1)
            else:
                predictions.append(0)
                # 补充的故障诊断结果
        save_path_of_complementary_result = 'app1/module_management/algorithms/functions/fault_diagnosis/' + user_dir
        if not os.path.exists(save_path_of_complementary_result):
            os.makedirs(save_path_of_complementary_result)
        complementary_figure, complementary_summary = complementary_result_of_fault_diagnosis(predictions,
                                                                                              save_path_of_complementary_result)
        num_has_no_fault = num_examples - num_has_fault
        # 绘制故障诊断的结果图像，多传感器为7传感器数据
        figure_path = spectrum_figure(example_filepath, multiple_sensor, num_sensor=7, user_dir=user_dir)
        return None, None, num_has_fault, num_has_no_fault, figure_path, complementary_figure, complementary_summary
    except Exception as e:
        return None, e, 0, 0, None, None, None


def diagnose_with_user_private_algorithm(datastream, private_algorithm, deeplearning=True, multiple_sensor=False):
    """
    使用用户的私有算法进行故障诊断
    :param deeplearning: 当为True时，为深度学习的故障诊断算法，当为False时为机器学习的故障诊断
    :param private_algorithm: 用户私有算法
    :param datastream: 数据流
    :param multiple_sensor: 是否为多传感器
    :return: 故障诊断结果："0" 代表无故障 ，"1"代表有故障；结果图像的保存路径
    """
    example = datastream.get('raw_data')  # 原始数据
    example_filepath = datastream.get('filepath')  # 原始数据来源文件
    user_dir = datastream.get('user_dir')  # 调用该增值服务算法的用户的用户目录
    username = user_dir.split('/')[0]  # 调用该增值服务算法的用户名

    # 存放私用故障诊断算法的文件目录
    # base_dir_of_algorithm_ml = 'app1/module_management/algorithms/models/private_fault_diagnosis_ml'  # 机器学习的故障诊断
    # base_dir_of_algorithm_dl = 'app1/module_management/algorithms/models/private_fault_diagnosis_dl'  # 深度学习的故障诊断
    print(f'private_algorithm: {private_algorithm}')
    extra_algorithm_dir = os.path.dirname(private_algorithm)  # 由路径中提取出算法源文件的父目录
    algorithm_name = os.path.basename(private_algorithm).split('.')[0]  # 由路径中提取出算法文件的名字
    print(f'algorithm_name: {algorithm_name}')
    # 通过中间文件的读写将作为故障诊断依据的数据输入用户私有故障诊断算法
    extra_algorithm_user_dir = extra_algorithm_dir + '/' + username  # 不同用户调用私有算法时，每个用户都有自己的文件夹存放数据
    if not os.path.exists(extra_algorithm_user_dir):
        os.makedirs(extra_algorithm_user_dir)

    # 以子进程的形式运行相应的私有故障诊断算法的python脚本
    if deeplearning:
        # 深度学习的私有故障诊断算法
        if isinstance(example, np.ndarray):
            flattened_example = example.flatten()
            if flattened_example.shape[0] > 2048:
                multiple_sensor = True
            else:
                multiple_sensor = False
        # 存放用户私有的故障诊断算法的文件路径
        # private_algorithm_dir = base_dir_of_algorithm_dl + '/' + username

        input_data_filepath = extra_algorithm_user_dir + '/input_data.npy'
        np.save(input_data_filepath, example)  # 输入样本数据保存为.npy的类型

        result = subprocess.run(shell=True, capture_output=True,
                                args=f"cd {extra_algorithm_dir} & python {algorithm_name}.py "
                                     f"--input-filepath ./{username}/input_data.npy "
                                     f"--model-filepath ./{algorithm_name}.pth --output-filepath ./{username}/output.pkl")
        print(f"deeplearning_shell_result: {result}")

        # prediction = 1 if '1' in result.stdout.decode('utf-8') else 0
    else:
        # 机器学习的私有故障诊断算法
        # 存放用户私有的故障诊断算法的文件路径
        # private_algorithm_dir = base_dir_of_algorithm_ml + '/' + username

        extracted_features: pandas.DataFrame = datastream.get('extracted_features')  # 前置模块提取的人工特征
        multiple_sensor = datastream.get('multiple_sensor')

        """由于系统对shell命令的传入长度有限制，将输入数据直接转为字符串的形式作为shell命令的传入参数并不适用"""
        # # 将字典类型的人工特征(extracted_features)转换为json字符串，以作为调用私有算法时的shell命令的传入参数
        # features_json = json.dumps(extracted_features)

        # 通过中间文件的读写将作为故障诊断依据的数据输入用户私有故障诊断算法

        input_data_filepath = extra_algorithm_user_dir + '/input_data.pkl'
        # print(f"extracted_features: {extracted_features}")

        extracted_features.to_pickle(input_data_filepath)

        result = subprocess.run(shell=True, capture_output=True,
                                args=f"cd {extra_algorithm_dir} & python {algorithm_name}.py "
                                     f"--input-filepath  ./{username}/input_data.pkl --output-filepath ./{username}/output.pkl "
                                     f"--model-filepath ./{algorithm_name}.pkl")
        # 诊断结果由私有算法打印为stdout，1为有故障，0为无故障
        print(f"shell_result: {result}")
        # prediction = 1 if '1' in result.stdout.decode('utf-8') else 0
    results_save_path = extra_algorithm_user_dir + '/output.pkl'
    results = pickle.load(open(results_save_path, 'rb'))
    # 返回错误信息
    error = result.stderr.decode('utf-8')
    if not deeplearning:
        if error:
            return None, error
        indicator = results['indicator']
        x_axis = results['x_axis']
        num_has_fault = results['num_has_fault']
        num_has_not_fault = results['num_has_not_fault']
        predictions = results['predictions']
        print('---------------------------------------------------')
        print('增值故障诊断算法运行结果.....')
        print('indicator: ', indicator)
        print('x_axis: ', x_axis)
        print('num_has_fault: ', num_has_fault)
        print('num_has_not_fault: ', num_has_not_fault)
        print('---------------------------------------------------')
        figure_path = plot_diagnosis(example_filepath, multiple_sensor, user_dir=user_dir)

        # 补充的故障诊断结果
        save_path_of_complementary_result = 'app1/module_management/algorithms/functions/fault_diagnosis/' + user_dir
        if not os.path.exists(save_path_of_complementary_result):
            os.makedirs(save_path_of_complementary_result)
        complementary_figure, complementary_summary = complementary_result_of_fault_diagnosis(predictions,
                                                                                              save_path_of_complementary_result)
        return indicator, x_axis, num_has_fault, num_has_not_fault, figure_path, complementary_figure, complementary_summary
    else:
        if error:
            return None, error
        num_has_fault = results['num_has_fault']
        num_has_no_fault = results['num_has_no_fault']
        predictions = results['predictions']
        figure_path = plot_diagnosis(example_filepath, user_dir=user_dir)
        # 补充的故障诊断结果
        save_path_of_complementary_result = 'app1/module_management/algorithms/functions/fault_diagnosis/' + user_dir
        if not os.path.exists(save_path_of_complementary_result):
            os.makedirs(save_path_of_complementary_result)
        complementary_figure, complementary_summary = complementary_result_of_fault_diagnosis(predictions,
                                                                                              save_path_of_complementary_result)
        return None, None, num_has_fault, num_has_no_fault, figure_path, complementary_figure, complementary_summary


# 绘制信号波形图
def plot_diagnosis(example_filepath, multiple_sensor=False, num_sensor=7, user_dir=None):
    """
    绘制信号的波形图
    :param user_dir: 用户模型运行结果的存放目录
    :param num_sensor: 传感器的数量
    :param example_filepath: 输入信号
    :param multiple_sensor: 是否为多传感器数据
    :return: 绘制的图形的保存路径
    """

    example, filename = load_data(example_filepath)
    # example = example.reshape(2048, -1)
    matplotlib.use('Agg')
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
    plt.rcParams['font.size'] = 15

    if not multiple_sensor:
        plt.figure(figsize=(16, 8))
        # if len(example.shape) == 2:
        #     example = example[0, :]
        # else:
        #     example = example
        plt.plot(example.flatten())
        plt.title('信号波形图')

        plt.xlabel('采样点')
        plt.ylabel('信号值')
    else:
        plt.figure(figsize=(20, 15))
        # 创建图形和子图  
        # plt.figure(figsize=(20, 10))  # 设置图形的大小
        # sensor_names = ['X维力(N)', 'Y维力(N)', 'Z维力(N)', 'X维振动(g)', 'Y维振动(g)', 'Z维振动(g)', 'AE-RMS (V)']
        # sensor_names = ['传感器1', '传感器2', '传感器3', '传感器4', '传感器5', '传感器6', '传感器7']
        sensor_names = [f'传感器{i}' for i in range(num_sensor)]  # 传感器名称
        num_sensors = example.shape[1]  # 获取传感器的数量
        for i, sensor_name in enumerate(sensor_names):

            plt.subplot(num_sensors, 1, i + 1)  # 创建子图，num_sensors 行 1 列，当前是第 i+1 个子图
            plt.plot(example[:, i])  # 绘制第 i 个传感器的信号
            plt.title(f'传感器{i + 1}-{sensor_name}')  # 设置子图的标题
            plt.xlabel('时间点', )  # 设置 x 轴标签
            plt.ylabel('信号值', )  # 设置 y 轴标签
            if i+1 >= num_sensor:
                break

        # 调整子图之间的间距
        # plt.title('信号波形图')
        plt.tight_layout()
    # save_path_dir = 'app1/module_management/algorithms/functions/fault_diagnosis/ulcnn/' + user_dir
    save_path_dir = 'app1/module_management/algorithms/functions/fault_diagnosis/' + user_dir
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    save_path = save_path_dir + '/' + f'{filename}_diagnosis.png'
        
    plt.savefig(save_path)

    return save_path


# 额外扩充模型故障诊断接口
def additional_fault_diagnose(datastream, multiple_sensor=False, model_type='additional_model_1'):
    """
    扩充故障诊断模型，包括模型：
    基于多传感器信号级加权融合的故障检测与诊断技术 (additional_model_one_multiple)
    基于多传感器信号时频表征自适应加权融合的故障检测与诊断技术  (additional_model_two_multiple)
    基于多传感器特征级融合的深度学习故障检测与诊断技术 (additional_model_three_multiple)
    多传感器决策级融合的深度学习故障检测与诊断 (additional_model_four_multiple)
    基于单传感器的知识型 1D 时域深度学习故障诊断 (additional_model_five)
    基于单传感器的时域和频域协同注意学习故障诊断 (additional_model_six)
    基于单传器的多域深度特征融合故障检测 (additional_model_seven)
    :param multiple_sensor: 是否为多传感器数据
    :param model_type: 模型类型
    :param datastream: 输入数据流
    :return: 预测故障类型，0表示无故障，1表示有故障
    """
    # if model_type == 'additional_model_1':
    #     predicted_class = mutli_sensor_1.fault_diagnose(input_signal)
    # elif model_type == 'additional_model_2':
    #     predicted_class = mutli_sensor_2.fault_diagnose(input_signal)
    # elif model_type == 'additional_model_3':
    #     predicted_class = mutli_sensor_3.fault_diagnose(input_signal)
    # elif model_type == 'additional_model_4':
    #     predicted_class = mutli_sensor_4.fault_diagnose(input_signal)
    # else:
    #     print("未知的模型类型")
    #     predicted_class = 0

    examples = datastream.get('raw_data')
    example_filepath = datastream.get('filepath')
    user_dir = datastream.get('user_dir')

    if examples.shape[0] < examples.shape[1]:
        examples = examples.T
    num_examples = int(examples.shape[0] / 2048)
    num_has_fault = 0
    predictions = []

    # 单传感器的算法
    try:
        for i in range(num_examples):
            example = examples[i * 2048:(i + 1) * 2048]
            if not multiple_sensor:
                # 单传感器的时频图卷积模型的故障诊断
                if model_type == 'additional_model_five_deeplearning':
                    predicted_class = single_sensor_1.fault_diagnose(example.flatten())
                elif model_type == 'additional_model_six_deeplearning':
                    predicted_class = single_sensor_2.fault_diagnose(example.flatten())
                elif model_type == 'additional_model_seven_deeplearning':
                    predicted_class = single_sensor_3.fault_diagnose(example.flatten())
                else:
                    print("未知的模型类型")
                    predicted_class = 0

            else:
                # 多传感器模型的故障诊断
                # input_data = torch.from_numpy(example).type(torch.cuda.FloatTensor).unsqueeze(dim=0).mT
                if model_type == 'additional_model_one_multiple_deeplearning':
                    predicted_class = mutli_sensor_1.fault_diagnose(example)
                elif model_type == 'additional_model_two_multiple_deeplearning':
                    predicted_class = mutli_sensor_2.fault_diagnose(example)
                elif model_type == 'additional_model_three_multiple_deeplearning':
                    predicted_class = mutli_sensor_3.fault_diagnose(example)
                elif model_type == 'additional_model_four_multiple_deeplearning':
                    predicted_class = mutli_sensor_4.fault_diagnose(example)
                else:
                    print("未知的模型类型")
                    predicted_class = 0
            print(f'样本{i}的故障类型为：{predicted_class}')

            if predicted_class == 1:
                num_has_fault += 1
                predictions.append(1)
            else:
                predictions.append(0)
        num_has_no_fault = num_examples - num_has_fault
        # 补充的故障诊断结果
        save_path_of_complementary_result = 'app1/module_management/algorithms/functions/fault_diagnosis/' + user_dir
        if not os.path.exists(save_path_of_complementary_result):
            os.makedirs(save_path_of_complementary_result)
        complementary_figure, complementary_summary = complementary_result_of_fault_diagnosis(predictions,
                                                                                              save_path_of_complementary_result)
        # print(f'num_has_fault: {num_has_fault}, num_has_no_fault: {num_has_no_fault}')
        # 绘制故障诊断的结果图像，多传感器为7传感器数据
        figure_path = plot_diagnosis(example_filepath, multiple_sensor, user_dir=user_dir)
        return None, None, num_has_fault, num_has_no_fault, figure_path, complementary_figure, complementary_summary
    except Exception as e:
        return None, e, 0, 0, None, None, None


if __name__ == '__main__':
    # 训练
    # train_Rf_model()
    # train_SVC()
    # 测试
    # diagnose_with_svc_model()
    # 测试
    # file_path = './已划分数据_4阶段.mat'  # MATLAB文件路径
    # stage = 'stage_3'
    print('hello world')

    # predicted = predict_with_model(input_data, GRU_model)
    # print(predicted)
