import pickle
import joblib
import argparse

if __name__ == '__main__':
    
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--model-filepath', type=str, default = None)  # 模型参数
    parser.add_argument('--input-filepath', type=str, default = None)  # 输入数据

    # 解析命令行参数
    args = parser.parse_args()
    model_filepath = args.model_filepath
    input_filepath = args.input_filepath
    
    # 加载模型
    model = joblib.load(model_filepath)
    
    # 模型推理
    example = pickle.load(open(input_filepath, 'rb'))  # 读取输入数据
    # 提取所需特征进行模型预测
    choose_features_eng = ['std', 'rms', 'var', 'rectified_mean', 'root_amplitude', 'peak_to_peak', 'cumulant_6th', 'mean',
                            'cumulant_4th', 'min']
    prediction = model.predict(example[choose_features_eng][0:1])
    
    # 以打印输出的形式返回故障预测结果（可能出现故障的时间）
    print(f'{prediction.item()}#', end='')  # 打印预测结果时，需要添加#号，否则无法识别