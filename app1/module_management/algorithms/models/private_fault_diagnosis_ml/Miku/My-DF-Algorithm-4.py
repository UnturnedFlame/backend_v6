import argparse
import joblib
import pickle

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
    choose_features = ['标准差', '均方根', '方差', '整流平均值', '方根幅值', '峰峰值', '六阶累积量', '均值',
                      '四阶累积量', '最小值']
    predicted = model.predict(example[choose_features][0:1])
    
    # 根据阈值判断有无故障
    if predicted.reshape(-1).item() < 0:
        prediction = 0
    else:
        prediction = 1
        
    # 以打印输出的形式返回故障诊断结果
    print(prediction)
