import base64
import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app1.module_management.algorithms.functions.fault_diagnosis import diagnose_with_svc_model, \
    diagnose_with_random_forest_model, time_regression, diagnose_with_ulcnn, diagnose_with_simmodel, \
    diagnose_with_gru_model, diagnose_with_lstm_model, diagnose_with_user_private_algorithm, additional_fault_diagnose
from app1.module_management.algorithms.functions.feature_selection import feature_imp, mutual_information_importance, \
    correlation_coefficient_importance
from app1.module_management.algorithms.functions.health_evaluation import model_eval
from app1.module_management.algorithms.functions.load_data import load_data
# from app1.module_management.algorithms.functions.speech_processing import mel_spectrogram, audio_sep, add_noise, \
#     verify_speaker
# from app1.module_management.algorithms.functions.load_model import load_model_with_pytorch_lightning
from app1.module_management.algorithms.functions.preprocessing import bicubic_interpolation, polynomial_interpolation, \
    newton_interpolation, linear_interpolation, lagrange_interpolation, extract_signal_features, \
    wavelet_transform_processing, \
    extract_features_with_multiple_sensors, dimensionless, interpolation_for_signals

"""
处理用户运行算法模型请求的算法引擎
"""


# 将图片编码为base64编码的方法
def encode_image_to_base64(image_path):
    """
    将图片编码为base64编码
    :param image_path: image saved path
    :return: the base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        # Base64编码返回的是bytes，所以需要将其解码为str
        return encoded_string.decode('utf-8')


# 用于完成预处理、故障诊断等任务的计算引擎
def add_user_dir(datastream: dict):
    if not datastream.get('user_dir') and datastream.get('filepath'):
        split_path = datastream['filepath'].split('/')
        datastream['user_dir'] = split_path[-2] + '/' + split_path[-1].split('.')[0]


class Reactor:
    def __init__(self, schedule, algorithm_dict, params_dict, multiple_sensor):
        """
        该算法引擎类的实例的初始化操作
        :param schedule: 各模块的运行流程
        :param params_dict: 各模块所使用的算法的具体参数
        :param algorithm_dict: 各模块使用的算法
        :return: 无返回值
        """
        # 使得self.schedule中不包含数据源
        self.multiple_sensor = multiple_sensor
        self.schedule = schedule
        self.current_module = ''
        # 存储前端返回的需要运行的模型的信息，包括用到的模块，模块的具体信息，算法的使用参数等
        self.module_configuration = {module: {'usage': False, 'algorithm': '', 'params': {}, 'result': {},
                                              'introduction': ''} for module in
                                     ['音频分离', '声纹识别', '说话人注册', '添加噪声', '插值处理', '特征提取',
                                      '层次分析模糊综合评估', '层次朴素贝叶斯评估', '层次逻辑回归评估', '健康评估',
                                      '小波变换', '特征选择', '故障诊断', '故障预测',
                                      '无量纲化']}

        for module in self.schedule:
            self.module_configuration[module]['algorithm'] = algorithm_dict[module]
            self.module_configuration[module]['params'] = params_dict[algorithm_dict[module]]
        # print(f'Module configuration: {self.module_configuration}')
        # 返回给前端的各模块的运行结果
        self.results_to_response = {n: {} for n in
                                    ['音频分离', '声纹识别', '说话人注册', '添加噪声', '插值处理', '特征提取',
                                     '层次分析模糊综合评估', '层次朴素贝叶斯评估', '层次逻辑回归评估', '健康评估',
                                     '小波变换', '特征选择', '故障诊断', '故障预测',
                                     '无量纲化']}
        # 运行过程中各个模块之间数据交换的格式，类似于数据流，每次传递数据时包含上一个模块的运行结果
        self.data_stream = {'filepath': None, 'raw_data': None, 'extracted_features': None,
                            'filename': None, 'user_dir': None, 'features_name': None, 'diagnosis_result': None,
                            'features_group_by_sensor': None, 'multiple_sensor': multiple_sensor}
        self.gradio_app = None
        self.lightning_model = None

    # 构造各模块间传递的数据流的方法
    def construct_data_stream(self, filepath=None, raw_data=None, extracted_features=None,
                              filename=None, user_dir=None, features_name=None, diagnosis_result=None,
                              features_group_by_sensor=None, multiple_sensor=None):
        lst = [filepath, raw_data, extracted_features, filename, user_dir, features_name, diagnosis_result,
               features_group_by_sensor, multiple_sensor]
        index = 0
        for k in self.data_stream.keys():
            if lst[index] is not None:
                self.data_stream[k] = lst[index]
            index = index + 1
            if index > len(lst) - 1:
                break
        # 向数据流中加入当前用户的目录，便于将该用户运行模型时得到的结果保存在该用户的目录下
        add_user_dir(self.data_stream)

        print('---------------------------------------------------')
        print(f"now {self.current_module} has been finished.......")
        # print(f'Data stream: {self.data_stream}')

    # 无量纲化
    def dimensionless(self, data_with_selected_features):
        """
        无量纲化处理
        :param data_with_selected_features: 传入的数据流
        :return: 构造的数据流
        """
        self.current_module = '无量纲化'
        use_algorithm = self.module_configuration['无量纲化']['algorithm']
        multiple_sensor = self.multiple_sensor  # 是否为多传感器的数据
        # print(f'use_algorithm: {use_algorithm}')
        if 'private_scaler' in use_algorithm:
            private_scale_algorithm = self.module_configuration['无量纲化']['params']['algorithm']  # 私有无量纲化算法
        else:
            private_scale_algorithm = None
        # print(f'private_scale_algorithm: {private_scale_algorithm}')
        # useLog = self.module_configuration['无量纲化']['params']['useLog']
        if isinstance(data_with_selected_features, str):
            # 读取原始输入信号的文件
            data, filename = load_data(data_with_selected_features)
            # 获取用户名
            self.construct_data_stream(data_with_selected_features)
            user_dir = self.data_stream.get('user_dir')
            raw_data_filepath = data_with_selected_features
            useLog = False
        else:
            # 输入的是其他模块处理过后传递的数据流
            user_dir = data_with_selected_features.get('user_dir')
            raw_data_filepath = data_with_selected_features.get('filepath')
            filename = data_with_selected_features.get('filename')
            # 如果提取特征非空，则是对样本特征进行标准化
            if self.data_stream['extracted_features'] is not None:
                data = data_with_selected_features.get('extracted_features')
                useLog = True
            # 提取特征结果为空，则是对信号序列进行标准化
            else:
                data = data_with_selected_features.get('raw_data')
                useLog = False

        # 如果输入的数据已经经过特征提取则是对特征进行标准化，因为规定的输入样本数量为1，因此需要使用模型训练时使用的标准化模型进行逆标准化
        if self.data_stream['extracted_features'] is not None:

            # 使用模型训练时的标准化模型对提取到的样本特征进行逆标准化
            features_group_by_sensor = data_with_selected_features.get('features_group_by_sensor')
            # multiple_sensor = data_with_selected_features.get('multiple_sensor')

            scaled_data, scaled_data_display, data_scaled_save_path, _ = dimensionless(data,
                                                                                       features_group_by_sensor,
                                                                                       user_dir=user_dir,
                                                                                       option=use_algorithm,
                                                                                       multiple_sensor=multiple_sensor,
                                                                                       use_log=useLog,
                                                                                       extra_algorithm_filepath=private_scale_algorithm)
            # self.module_configuration['无量纲化']['result']['raw_data'] = data
            # self.module_configuration['无量纲化']['result']['scaled_data'] = scaled_data

            # self.results_to_response['无量纲化']['raw_data'] = [list(row) for index, row in data.iterrows()]
            # self.results_to_response['无量纲化']['scaled_data'] = [list(row) for index, row in scaled_data.iterrows()]
            if scaled_data is None:
                # 模型运行出错，打印错误信息
                print(scaled_data_display)
                return ''
            self.results_to_response['无量纲化']['raw_data'] = features_group_by_sensor  # 未经过标准化的样本特征
            self.results_to_response['无量纲化']['scaled_data'] = scaled_data_display  # 经过标准化后的样本特征
            self.results_to_response['无量纲化']['features_name'] = data_with_selected_features.get('feature_name')
            self.results_to_response['无量纲化']['datatype'] = 'table'
        else:
            # 对原始信号序列进行标准化
            # if isinstance(data_with_selected_features, dict):
            #     multiple_sensor = data_with_selected_features.get('multiple_sensor')
            # else:
            #     multiple_sensor = None
            scaled_data, scaled_data_display, data_scaled_save_path, ms = dimensionless(data, None,
                                                                                        user_dir=user_dir,
                                                                                        option=use_algorithm,
                                                                                        multiple_sensor=multiple_sensor,
                                                                                        use_log=useLog,
                                                                                        extra_algorithm_filepath=private_scale_algorithm)
            # if multiple_sensor is None:
            #     multiple_sensor = ms
            if scaled_data is None:
                # 无量纲化处理出错，打印错误信息
                print(f'scaled_data_display: {scaled_data_display}')
                return ''
            for num, figure in enumerate(scaled_data_display):
                self.results_to_response['无量纲化'][f'sensor{num + 1}_figure_Base64'] = figure
                self.results_to_response['无量纲化']['datatype'] = 'figure'

        # 将标准化以后的数据放入数据流
        if not useLog:
            self.construct_data_stream(raw_data=scaled_data, filepath=raw_data_filepath,
                                       filename=filename, multiple_sensor=multiple_sensor)
        else:
            self.construct_data_stream(raw_data=scaled_data, filepath=raw_data_filepath,
                                       filename=filename, multiple_sensor=multiple_sensor)
        return self.data_stream

    # 插值处理
    def interpolation_v1(self, datafile):
        self.module_configuration['插值处理']['result']['原始数据'] = pd.read_excel(datafile)
        use_algorithm = self.module_configuration['插值处理']['algorithm']
        if use_algorithm == 'polynomial_interpolation':
            waveform, result_data = polynomial_interpolation(datafile)
        elif use_algorithm == 'bicubic_interpolation':
            waveform, result_data = bicubic_interpolation(datafile)
        elif use_algorithm == 'newton_interpolation':
            waveform, result_data = newton_interpolation(datafile)
        elif use_algorithm == 'linear_interpolation':
            waveform, result_data = linear_interpolation(datafile)
        elif use_algorithm == 'lagrange_interpolation':
            waveform, result_data = lagrange_interpolation(datafile)
        else:
            waveform, result_data = '', ''
        self.module_configuration['插值处理']['result']['原始数据波形图'] = waveform.get('原始数据')
        self.module_configuration['插值处理']['result']['结果数据波形图'] = waveform.get('结果数据')
        self.module_configuration['插值处理']['result']['结果数据'] = pd.read_excel(result_data)

        return result_data

    def wavelet_transform_denoise(self, datafile):
        """
        小波变换去噪
        :param datafile: 输入的数据
        :return: 本类封装的数据流，datastream
        """
        self.current_module = '小波变换'
        # 如果输入的是字典形式的数据流（datastream）
        if isinstance(datafile, dict):
            raw_data = datafile.get('raw_data')
            filename = datafile.get('filename')
            user_dir = datafile.get('user_dir')

        # 否则直接从原始数据的文件路径读取原始信号
        else:
            raw_data, filename = load_data(datafile)
            self.construct_data_stream(datafile)
            user_dir = self.data_stream.get('user_dir')

        # 获取小波变换的参数
        algorithm = self.module_configuration['小波变换']['algorithm']

        if algorithm == 'wavelet_trans_denoise':
            # 小波变换降噪
            params: dict = self.module_configuration['小波变换']['params']
            wavelet = params.get('wavelet')
            wavelet_level = params.get('wavelet_level')
            results, multiple_sensor = wavelet_transform_processing(raw_data,
                                                                    wavelet=wavelet,
                                                                    level=wavelet_level,
                                                                    user_dir=user_dir,
                                                                    multiple_sensor=self.multiple_sensor)
        else:
            # 调用小波变换的专有算法
            extra_algorithm = self.module_configuration['小波变换']['params']
            print(f'extra_algorithm: {extra_algorithm}')
            results, multiple_sensor = wavelet_transform_processing(raw_data,
                                                                    user_dir=user_dir,
                                                                    extra_algorithm=extra_algorithm,
                                                                    multiple_sensor=self.multiple_sensor)
        transformed_data = results.get('transformed_data')  # 小波变换处理的结果数据
        transformed_data_save_path = results.get('data_save_path')
        figure_path = results.get('figure_path')

        # 构造数据流
        self.construct_data_stream(raw_data=transformed_data, filename=filename,
                                   filepath=transformed_data_save_path, multiple_sensor=multiple_sensor)

        if transformed_data is None:
            # 小波变换降噪模块运行出错，打印捕获到的错误信息
            print(f'figure_path: {figure_path}')
            return ''
        # results = wavelet_denoise_four_stages(data_mat, filename)
        if figure_path is not None:
            for num, figure in enumerate(figure_path):
                self.results_to_response['小波变换'][f'sensor{num + 1}_figure_Base64'] = figure
        else:
            print('无小波变换输出结果')
            # 多段数据的小波降噪
            # all_save_paths, denoise_datas = results.get('all_save_paths'), results.get('denoised_datas')
            # self.module_configuration['小波变换']['result']['figure_paths'] = {}
            # self.module_configuration['小波变换']['result']['denoised_datas'] = {}
            # for key, value in all_save_paths.items():
            #     self.module_configuration['小波变换']['result']['figure_paths'][key] = value
            # for key, value in denoise_datas.items():
            #     self.module_configuration['小波变换']['result']['denoised_datas'][key] = value
        # 构造数据流self.construct_data_stream(raw_data=denoised_data, filename=filename,
        #         #                            filepath=denoised_data_save_path, multiple_sensor=multiple_sensor)
        #
        return self.data_stream

    def interpolation_v2(self, datafile):
        """
        对信号的插值处理
        :param datafile: 输入的数据文件
        :return: 数据流
        """
        self.current_module = '插值处理'
        # 读取原始数据
        # raw_data, filename = load_data(datafile)
        use_algorithm = self.module_configuration['插值处理']['algorithm']

        private_interpolation = None
        # 用户私有的插值算法
        if 'private_interpolation' in use_algorithm:
            # 获取用户私有算法的算法名，后面用于调用具体的用户私有算法
            private_interpolation = self.module_configuration['插值处理']['params']
        print(f'private_interpolation: {private_interpolation}')

        # 如果输入的是字典形式的数据流（datastream）
        if isinstance(datafile, dict):
            raw_data = datafile.get('raw_data')
            filename = datafile.get('filename')
            # filepath = datafile.get('filepath')
            user_dir = datafile.get('user_dir')

        # 否则直接读取原始数据文件路径，也就是特征提取作为第一个模块
        else:
            raw_data, filename = load_data(datafile)
            # 获取用户名
            self.construct_data_stream(filepath=datafile)
            user_dir = self.data_stream.get('user_dir')

        results, multiple_sensor = interpolation_for_signals(raw_data, use_algorithm, filename, user_dir=user_dir,
                                                             private_algorithm=private_interpolation,
                                                             multiple_sensor=self.multiple_sensor)

        interpolated_data_path = results.get('interpolated_data')
        figure_path = results.get('figure_paths')

        if interpolated_data_path is None:
            # 插值处理出错，打印错误信息
            print(figure_path)
            return ''

        for num, figure in enumerate(figure_path):
            self.results_to_response['插值处理'][f'sensor{num + 1}_figure_Base64'] = figure

        # 插值后的数据
        interpolated_data, filename = load_data(interpolated_data_path)

        # 构造数据流
        self.construct_data_stream(filepath=interpolated_data_path, raw_data=interpolated_data,
                                   filename=filename, multiple_sensor=multiple_sensor)

        return self.data_stream

    # 特征提取
    def feature_extraction(self, datafile, multiple_sensor=False):
        """
        特征提取
        :param datafile: 传入的数据，有两种情况，分别为传入原始数据文件，和传入之前模块运行后传递进来的数据流
        :param multiple_sensor: 是否为多传感器的特征提取
        :return: 构造的数据流
        """
        self.current_module = '特征提取'
        all_time_features = ['最大值', '最小值', '中位数', '峰峰值', '均值', '方差', '标准差', '峰度', '偏度',
                             '整流平均值', '均方根', '方根幅值', '波形因子', '峰值因子', '脉冲因子', '裕度因子',
                             '四阶累积量', '六阶累积量']
        all_frequency_features = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值',
                                  '谱峭度的标准差', '谱峭度的峰度', '谱峭度的偏度']
        features = {'time_domain': [], 'frequency_domain': []}
        use_algorithm = self.module_configuration['特征提取']['algorithm']
        # if '_multiple' in use_algorithm:
        #     multiple_sensor = True
        multiple_sensor = self.multiple_sensor
        # 获取用户选择提取的特征
        for key, value in self.module_configuration['特征提取']['params'].items():
            if value:
                # 将特征分为时域和频域特征
                if key in all_time_features:
                    features['time_domain'].append(key)
                elif key in all_frequency_features:
                    features['frequency_domain'].append(key)
        print('features: ', features)
        # 如果输入的是字典形式的数据流（datastream）
        if isinstance(datafile, dict):
            data = datafile.get('raw_data')
            filename = datafile.get('filename')
            filepath = datafile.get('filepath')
            user_dir = datafile.get('user_dir')

        # 否则直接读取原始数据文件路径，也就是特征提取作为第一个模块
        else:
            data, filename = load_data(datafile)
            filepath = datafile
            # 获取用户名
            self.construct_data_stream(filepath=datafile)
            user_dir = self.data_stream.get('user_dir')

        # 单传感器特征提取
        if not multiple_sensor:
            features_save_path, features_with_name, num_examples = extract_signal_features(data, features, filename,
                                                                                           save=True,
                                                                                           user_dir=user_dir)
            if features_save_path is None:
                # 特征提取出错，返回报错
                return ''

            features_extracted, filename = load_data(features_save_path)

        # 多传感器特征提取
        else:
            features_save_path, features_with_name, num_examples = extract_features_with_multiple_sensors(data,
                                                                                                          features,
                                                                                                          filename,
                                                                                                          user_dir=user_dir)
            if features_save_path is None:
                # 特征提取出错，返回报错
                return ''

            features_extracted, filename = load_data(features_save_path)
        # 如果提取特征时出现错误(形式上由features_save_path)，将返回错误信息以及一个空字典，这时应该立即停止后续的计算
        if features_with_name == {}:
            # 打印错误信息
            print(features_save_path)
            return ''
        self.module_configuration['特征提取']['result'] = features_extracted
        self.results_to_response['特征提取']['features_with_name'] = features_with_name

        featuresToDrawLineChart = {}
        standard_scaler = StandardScaler()

        # print(f"features_with_name: {features_with_name}")

        features_name = features_with_name['features_name']  # 特征名

        features_extracted_group_by_sensor = features_with_name['features_extracted_group_by_sensor']  # 各个传感器的特征
        for sensor, features in features_extracted_group_by_sensor.items():
            features_array = np.array(features)
            # 对每一行数据进行标准化
            features_array = standard_scaler.fit_transform(features_array)
            # 将数组中的数据精度降低到小数点后三位
            features_array = np.round(features_array, 3)
            featuresToDrawLineChart[sensor] = {}
            print(features_array.shape)
            for i, feature_name in enumerate(features_name):
                featuresToDrawLineChart[sensor][feature_name] = features_array[:, i].flatten().tolist()
            # print(f'sensor: {sensor}, features: {featuresToDrawLineChart[sensor]}')

        # for col in features_extracted.columns:
        #     data_col = features_extracted[[col]].values
        #     data_normalized = standard_scaler.fit_transform(data_col)
        #     featuresToDrawLineChart[col] = data_normalized.flatten().tolist()
        # print(f'featuresToDrawLineChart: {featuresToDrawLineChart}')
        self.results_to_response['特征提取']['featuresToDrawLineChart'] = featuresToDrawLineChart

        # print(f'features_extracted: {features_extracted}')
        print(f'1_data: {data.shape}')
        # if len(data.flatten()) <= 2048:
        #     data = data.reshape(1, -1)
        # elif data.shape[0] == 2048:
        if data.shape[0] < data.shape[1]:
            data = data.T
        print(f'2_data: {data.shape}')
        num_sensors = data.shape[1]
        raw_data = [data[:, i].flatten().tolist() for i in range(num_sensors)]
        self.results_to_response['特征提取']['raw_data'] = raw_data
        print(f"raw_data: {len(self.results_to_response['特征提取']['raw_data'])}")
        self.results_to_response['特征提取']['num_examples'] = num_examples
        self.construct_data_stream(filepath=filepath, raw_data=data, extracted_features=features_extracted,
                                   filename=filename, features_name=features, multiple_sensor=multiple_sensor,
                                   features_group_by_sensor=features_with_name)

        return self.data_stream

    # 健康评估
    def health_evaluation(self, data_with_selected_features, algorithm='FAHP'):
        """
        健康评估
        :param algorithm: 使用的健康评估算法
        :param multiple_sensor: 是否为多传感器的健康评估
        :param data_with_selected_features: 输入的数据流
        :return:构造的数据流
        """
        self.current_module = '健康评估'
        example = data_with_selected_features.get('extracted_features')
        print(f'1__example: {example.shape}')
        # data_path = data_with_selected_features.get('filepath')
        user_dir = data_with_selected_features.get('user_dir')
        raw_data = data_with_selected_features.get('raw_data')

        extra_health_evaluation_filepath = None  # 专有健康评估算法
        model_path = None  # 健康评估模型的路径
        model_path_multiple_sensor = None
        # filename = data_with_selected_features.get('filename')
        if algorithm == 'FAHP':
            # 层次分析模糊综合评估
            # 单传感器的健康评估
            algorithm_name = '层次分析模糊综合评估'
            model_path = 'app1/module_management/algorithms/models/health_evaluation/model_1.pkl'
            # 多传感器的健康评估
            model_path_multiple_sensor = 'app1/module_management/algorithms/models/health_evaluation/model_2.pkl'
        elif algorithm == 'AHP':
            # 层次逻辑回归评估
            # 单传感器的健康评估
            algorithm_name = '层次逻辑回归评估'
            model_path = 'app1/module_management/algorithms/models/health_evaluation/model_5.pkl'
            # 多传感器的健康评估
            model_path_multiple_sensor = 'app1/module_management/algorithms/models/health_evaluation/model_6.pkl'
        elif algorithm == 'BHM':
            # 层次朴素贝叶斯评估
            # 单传感器的健康评估
            algorithm_name = '层次朴素贝叶斯评估'
            model_path = 'app1/module_management/algorithms/models/health_evaluation/model_3.pkl'
            # 多传感器的健康评估
            model_path_multiple_sensor = 'app1/module_management/algorithms/models/health_evaluation/model_4.pkl'
        else:
            # 专有健康评估算法
            algorithm_name = '健康评估'
            extra_health_evaluation_filepath = self.module_configuration[algorithm_name]['params']
            # 存放专有健康评估算法的目录
            # extra_health_evaluation_dir = os.path.dirname(extra_health_evaluation_filepath)
            # if not os.path.exists(extra_health_evaluation_dir):
            #     os.makedirs(extra_health_evaluation_dir)
        username = user_dir.split('/')[0]  # 调用该算法的用户的用户名
        # 健康评估结果存放路径
        save_path = f'app1/module_management/algorithms/functions/health_evaluation_results/' + user_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # if '_multiple' in self.module_configuration[algorithm_name]['algorithm']:
        #     multiple_sensor = True

        print(f"algorithm_name: {algorithm_name}")

        if not self.multiple_sensor:
            print(f"example: {example.shape}")
            # 单传感器健康评估，以及专有健康评估算法
            results_path = model_eval(example, raw_data, model_path, save_path,
                                      algorithm, extra_algorithm_filepath=extra_health_evaluation_filepath,
                                      username=username, multiple_sensors=self.multiple_sensor)
        else:
            # 多传感器健康评估，以及专有健康评估算法
            '''还没写增值服务组件的逻辑'''
            # results_path = model_eval_multiple_sensor(raw_data, model_path_multiple_sensor, save_path, algorithm,
            #                                           extra_algorithm_filepath=extra_health_evaluation_filepath,
            #                                           username=username)
            results_path = model_eval(example, raw_data, model_path_multiple_sensor, save_path,
                                      algorithm, extra_algorithm_filepath=extra_health_evaluation_filepath,
                                      username=username, multiple_sensors=self.multiple_sensor)
        if results_path.get('weights_bar') is None:
            # 程序运行出错，打印错误信息
            print(f"健康评估程序运行出错，请检查输入数据是否正确，或联系管理员")
            print(results_path.get('result_vector'))
            return ''
        suggestion_filepath_list = results_path.get('suggestion')
        # print(f'suggestion_file_path_list: {suggestion_filepath_list}')
        suggestion_list = []
        for suggestion_filepath in suggestion_filepath_list:
            with open(suggestion_filepath, 'r', encoding='gbk') as file:
                suggestion = file.read()
                suggestion_list.append(suggestion)
        # print(f'suggestion_list: {suggestion_list}')
        # 需要返回给前端的健康评估后的结果，此处获取的是多个样本的评估结果，为列表的形式

        self.results_to_response[algorithm_name]['评估建议'] = suggestion_list
        self.results_to_response[algorithm_name]['二级指标权重柱状图_Base64'] = results_path.get(
            'weights_bar')
        self.results_to_response[algorithm_name]['评估结果柱状图_Base64'] = results_path.get('result_bar')
        self.results_to_response[algorithm_name]['层级有效指标_Base64'] = results_path.get('tree')

        self.results_to_response[algorithm_name]['最终评估结果'] = results_path.get('final_result_suggestion')
        self.results_to_response[algorithm_name]['各样本状态隶属度'] = results_path.get('status_of_all_examples')

        return results_path

    # 特征选择
    def feature_selection(self, data_with_selected_features, multiple_sensor=False):
        """
        特征选择
        :param data_with_selected_features: 传入进来的数据流
        :param multiple_sensor: 是否为多传感器的特征选择
        :return: 构造的数据流
        """

        self.current_module = '特征选择'
        print(f'特征选择......')
        use_algorithm = self.module_configuration['特征选择']['algorithm']
        rule = self.module_configuration['特征选择']['params']['rule']
        threshold = self.module_configuration['特征选择']['params']['threshold' + str(rule)]
        user_dir = data_with_selected_features.get('user_dir')

        if '_multiple' in use_algorithm:
            multiple_sensor = True

        if 'feature_imp' in use_algorithm:
            # 树模型的特征选择
            selection_figure_path, features, corr_matrix_heatmap = feature_imp(multiple_sensor, rule, threshold,
                                                                               user_dir=user_dir)
        elif 'mutual_information_importance' in use_algorithm:
            # 互信息重要性的特征选择
            selection_figure_path, features, corr_matrix_heatmap = mutual_information_importance(multiple_sensor,
                                                                                                 rule, threshold,
                                                                                                 user_dir=user_dir)
        else:
            # 相关系数重要性的特征选择
            selection_figure_path, features, corr_matrix_heatmap = correlation_coefficient_importance(multiple_sensor,
                                                                                                      rule, threshold,
                                                                                                      user_dir=user_dir)

        # 需要返回给前端的结果
        self.results_to_response['特征选择']['figure_Base64'] = selection_figure_path
        self.results_to_response['特征选择']['heatmap_Base64'] = corr_matrix_heatmap
        self.results_to_response['特征选择']['selected_features'] = features
        if rule == 1:
            self.results_to_response['特征选择']['rule'] = f'选择重要性大于{threshold}的特征'
        else:
            self.results_to_response['特征选择'][
                'rule'] = f'所选特征的重要性的总和占所有特征的重要性比例不小于{threshold}（所有特征重要性的总和占比为1），优先选择重要性高的特征'
        self.construct_data_stream(features_name=features)

        return self.data_stream

    # 故障诊断
    def fault_diagnosis(self, input_data, multiple_sensor=False):
        """
        故障诊断
        :param input_data: 传入的数据流
        :param multiple_sensor: 是否为多传感器数据的故障诊断
        :return: 构造的数据流
        """

        self.current_module = '故障诊断'
        use_algorithm = self.module_configuration['故障诊断']['algorithm']
        # 如果传入的是原始信号序列，此时只能使用深度学习模型的故障诊断
        # 如果是机器学习的故障诊断算法则不可能是输入原始信号，应为前置模块提取的振动信号的时域或是频域特征
        if isinstance(input_data, str):
            data_for_deeplearning, filename = load_data(input_data)
            self.construct_data_stream(raw_data=data_for_deeplearning, filepath=input_data, filename=filename)
        # 获取用户名
        user_dir = self.data_stream.get('user_dir')

        # 如果是针对多传感器的算法，则multiple_sensor为True，否则为针对单传感器的算法
        multiple_sensor = self.multiple_sensor
        if '_multiple' in use_algorithm:
            multiple_sensor = True
        # 根据用户使用的具体算法进行故障诊断
        if 'random_forest' in use_algorithm:
            # 随机森林的故障诊断
            (indicator, x_axis, num_has_fault, num_has_no_fault,
             figure_path, complementary_figure, complementary_summary) = diagnose_with_random_forest_model(
                input_data,
                multiple_sensor, user_dir=user_dir)
            # print(f'num has fault: {num_has_fault}, num has no fault: {num_has_no_fault}')

            self.module_configuration['故障诊断']['result']['信号波形图'] = figure_path
        elif 'svc' in use_algorithm:
            # 支持向量机的故障诊断
            (indicator, x_axis, num_has_fault, num_has_no_fault,
             figure_path, complementary_figure, complementary_summary) = diagnose_with_svc_model(
                input_data,
                multiple_sensor,
                user_dir=user_dir)
            self.module_configuration['故障诊断']['result']['信号波形图'] = figure_path
        elif 'gru' in use_algorithm:
            # gru的故障诊断
            (indicator, x_axis, num_has_fault, num_has_no_fault,
             figure_path, complementary_figure, complementary_summary) = diagnose_with_gru_model(self.data_stream,
                                                                                                 multiple_sensor)
            print(f'indicator: {indicator}')
            print(f'num has fault: {num_has_fault}'
                  f'num has no fault: {num_has_no_fault}')
        elif 'lstm' in use_algorithm:
            # lstm的故障诊断
            (indicator, x_axis, num_has_fault, num_has_no_fault,
             figure_path, complementary_figure, complementary_summary) = diagnose_with_lstm_model(self.data_stream,
                                                                                                  multiple_sensor)
        elif 'ulcnn' in use_algorithm:
            # 一维卷积网络的故障诊断
            (indicator, x_axis, num_has_fault, num_has_no_fault,
             figure_path, complementary_figure, complementary_summary) = diagnose_with_ulcnn(self.data_stream,
                                                                                             multiple_sensor)
        elif 'spectrumModel' in use_algorithm:
            # 针对时频图的深度学习网络的故障诊断
            (indicator, x_axis, num_has_fault, num_has_no_fault,
             figure_path, complementary_figure, complementary_summary) = diagnose_with_simmodel(self.data_stream,
                                                                                                multiple_sensor)
        elif 'private_fault_diagnosis' in use_algorithm:
            # 用户私有的故障诊断算法，分为深度学习和机器学习的故障诊断，其输入有区别
            print(f'use_algorithm is {use_algorithm}')
            if 'machine_learning' in use_algorithm:
                deeplearning = False  # 调用私有的机器学习的故障诊断算法
            else:
                deeplearning = True  # 调用私有的深度学习的故障诊断算法
            private_fault_diagnosis_algorithm = self.module_configuration['故障诊断']['params']  # 增值服务故障诊断算法源文件的存放路径
            (indicator, x_axis, num_has_fault, num_has_no_fault,
             figure_path, complementary_figure, complementary_summary) = diagnose_with_user_private_algorithm(
                self.data_stream,
                private_fault_diagnosis_algorithm,
                deeplearning,
                multiple_sensor)
        else:
            # 扩充的故障诊断模型
            additional_model_type_list = ['additional_model_one_multiple', 'additional_model_two_multiple',
                                          'additional_model_three_multiple',
                                          'additional_model_four_multiple', 'additional_model_five',
                                          'additional_model_six', 'additional_model_seven']
            if use_algorithm in additional_model_type_list:
                (indicator, x_axis, num_has_fault, num_has_no_fault,
                 figure_path, complementary_figure, complementary_summary) = additional_fault_diagnose(
                    self.data_stream, multiple_sensor, model_type=use_algorithm)
            else:
                print("模型名称错误")
                return ''
        num_examples = num_has_no_fault + num_has_fault
        # 当有故障的样本个数大于70%，则认为该段连续样本有故障
        if num_has_fault >= int(num_examples * 0.7):
            diagnosis_result = 1  # 有故障
        else:
            diagnosis_result = 0  # 无故障

        if figure_path is not None:
            self.construct_data_stream(diagnosis_result=diagnosis_result)
            # self.results_to_response['故障诊断']['diagnosis_result'] = diagnosis_result
            print(f'diagnosis_result: {diagnosis_result}')
            if diagnosis_result == 0:
                self.module_configuration['故障诊断']['result'][
                    '诊断结果'] = (
                    f'### 由输入的振动信号，共截取到{num_examples}个样本，根据对应故障诊断模型预测，其中{num_has_fault}个样本存在故障，未达到样本总数的70%，由此判断该部件<span '
                    f'style=\"color: red\">无故障</span>')
                # self.results_to_response['故障诊断']['diagnosis_result'] = '### 由输入的振动信号，根据故障诊断算法得知该部件<span
                # style=\"color: red\">无故障</span>'
            else:
                self.module_configuration['故障诊断']['result'][
                    '诊断结果'] = (
                    f'### 由输入的振动信号，共截取到{num_examples}个长为2048的样本，根据对应故障诊断模型预测，其中{num_has_fault}个样本存在故障，达到样本总数的70'
                    f'%，由此判断该部件<span'
                    f' style=\"color: red\">存在故障</span>')
            self.results_to_response['故障诊断']['indicator'] = indicator if indicator else "none"
            self.results_to_response['故障诊断']['x_axis'] = x_axis if x_axis else "none"
            self.results_to_response['故障诊断']['num_has_fault'] = num_has_fault
            self.results_to_response['故障诊断']['num_has_no_fault'] = num_has_no_fault
            self.results_to_response['故障诊断']['diagnosis_result'] = str(diagnosis_result)
            self.results_to_response['故障诊断']['complementary_Base64'] = complementary_figure
            self.results_to_response['故障诊断']['complementary_summary'] = complementary_summary
            self.results_to_response['故障诊断']['resultText'] = self.module_configuration['故障诊断']['result'][
                '诊断结果']
        else:
            # 程序运行出错，打印错误信息
            print('error: ', x_axis)
            return ''
        if figure_path:
            self.results_to_response['故障诊断']['figure_Base64'] = figure_path
        return self.data_stream

    # 故障预测
    def fault_prediction(self, example_with_selected_features, multiple_sensor=False):
        """
        故障故障预测
        :param example_with_selected_features: 传入的数据流
        :param multiple_sensor: 是否为多传感器的故障预测
        :return: 构造的数据流
        """

        self.current_module = '故障预测'

        # 换算时间函数
        def format_seconds(seconds):
            days = seconds // (24 * 3600)
            seconds %= (24 * 3600)
            hours = seconds // 3600
            seconds %= 3600
            minutes = seconds // 60
            seconds %= 60

            # 使用格式化字符串来组合结果
            return f"{days}天{hours}小时{minutes}分钟{seconds:.1f}秒"

        use_algorithm = self.module_configuration['故障预测']['algorithm']
        if 'private_fault_prediction' in use_algorithm:
            private_fault_prediction = self.module_configuration['故障预测']['params']
        else:
            private_fault_prediction = None
        # if '_multiple' in use_algorithm:
        #     multiple_sensor = True
        # 线性回归故障预测
        time_to_fault, figure_path = time_regression(example_with_selected_features,
                                                     private_algorithm=private_fault_prediction)
        if time_to_fault == 0:
            time_to_fault_str = ''
            self.module_configuration['故障预测']['result'][
                'evaluation'] = '经算法预测，目前该部件<span style=\"color: red\">已经故障</span>'
            self.module_configuration['故障预测']['result']['figure_path'] = figure_path
        elif time_to_fault is None:
            return ''
        else:
            # 计算可能会出故障的时间
            time_to_fault_str = format_seconds(abs(time_to_fault) // 10)
            self.module_configuration['故障预测']['result'][
                'evaluation'] = f'目前该部件<span style=\"color: red\">还未出现故障</span>，预测<span style=\"color: red\">{time_to_fault_str}</span>后会出现故障'
            self.module_configuration['故障预测']['result']['figure_path'] = figure_path
        self.results_to_response['故障预测']['figure_Base64'] = figure_path
        self.results_to_response['故障预测']['time_to_fault'] = str(time_to_fault)
        self.results_to_response['故障预测']['time_to_fault_str'] = time_to_fault_str

        return example_with_selected_features

    def start(self, datafile, queue=None):
        """
        依据建立的模型进行业务处理
        :param datafile: 输入的原始数据文件的路径
        :return: 无返回值
        :param queue: 后端视图进行多线程任务处理用户的模型运行请求，通过该队列queue进行子进程和父进程间的通信
        """
        input_data = datafile
        file_type = datafile.split('.')[-1]
        outcome = 'xxx'

        for module in self.schedule:
            if outcome == '':
                break
            if module == '插值处理':
                if file_type == 'csv':
                    # 对csv形式存储的数据进行的插值
                    outcome = self.interpolation_v1(input_data)
                elif file_type == 'mat' or file_type == 'npy':
                    # 对信号序列进行的插值
                    outcome = self.interpolation_v2(input_data)
            elif module == '特征提取':
                outcome = self.feature_extraction(input_data)
            elif module == '层次分析模糊综合评估':
                outcome = self.health_evaluation(input_data, algorithm='FAHP')
            elif module == '层次朴素贝叶斯评估':
                outcome = self.health_evaluation(input_data, algorithm='BHM')
            elif module == '层次逻辑回归评估':
                outcome = self.health_evaluation(input_data, algorithm='AHP')
            elif module == '健康评估':
                outcome = self.health_evaluation(input_data, algorithm='extra_health_evaluation')
            elif module == '特征选择':
                outcome = self.feature_selection(input_data)
            elif module == '无量纲化':
                outcome = self.dimensionless(input_data)
            elif module == '故障诊断':
                outcome = self.fault_diagnosis(input_data)
            elif module == '小波变换':
                outcome = self.wavelet_transform_denoise(input_data)
            elif module == '故障预测':
                outcome = self.fault_prediction(input_data)
            else:
                # 模型中所有模块都运行结束，程序结束
                outcome = ''
            input_data = outcome
        # print(self.results_to_response)
        # print(queue)
        if queue is not None:
            try:
                # 通过队列向启动模型运行任务子线程的父进程返回运行结果
                queue.put(self.results_to_response)
            except Exception as e:
                print(str(e))
