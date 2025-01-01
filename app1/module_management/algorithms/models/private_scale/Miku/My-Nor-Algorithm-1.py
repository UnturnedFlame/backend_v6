import pickle
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd



if __name__ == '__main__':
        
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--model-filepath', type=str, default = None)  # 模型参数
    parser.add_argument('--input-filepath', type=str, default = None)  # 输入数据
    parser.add_argument('--output-filepath', type=str, default = None)  # 输出结果

    # 解析命令行参数
    args = parser.parse_args()
    model_filepath = args.model_filepath
    input_filepath = args.input_filepath
    output_filepath = args.output_filepath
    
    # 加载数据
    filetype = input_filepath.split('.')[-1]
    if filetype == 'pkl':
        # 处理数据为提取的特征
        # 加载保存为.pkl类型的数据
        raw_data = pickle.load(open(input_filepath, 'rb'))
    else:
        # 处理数据为信号序列
        # 加载保存为.npy类型的数据
        raw_data = np.load(input_filepath)
        
    # 对输入数据进行无量纲化，并将无量纲化后的数据保存为文件
    if model_filepath is not None:
        # (1)使用保存的训练数据的参数进行无量纲化，以scikit-learn的MinMaxScaler为例
        choose_features = ['标准差', '均方根', '方差', '整流平均值', '方根幅值', 
                           '峰峰值', '六阶累积量', '均值', '四阶累积量', '最小值']  # 选择训练时的模型使用的特征
        scaler = joblib.load(model_filepath)  # 加载保存的训练数据的无量纲化参数
        data_scaled: pd.DataFrame = raw_data.copy()
        data_scaled[choose_features] = scaler.transform(raw_data[choose_features]) 
        data_scaled.to_pickle(output_filepath) 
     
    else:
        # (2)对信号序列进行无量纲化，以scikit-learn的MinMaxScaler为例
        scaler = MinMaxScaler()
        raw_data = raw_data.reshape(-1, 1)
        data_scaled: np.ndarray = scaler.fit_transform(raw_data)
        np.save(output_filepath, data_scaled)
   
    
