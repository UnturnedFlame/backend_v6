import argparse
import joblib
import pickle
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--model-filepath', type=str, default = None)  # 模型参数
    parser.add_argument('--input-filepath', type=str, default = None)  # 输入数据
    parser.add_argument('--output-filepath', type=str, default = None)

    # 解析命令行参数
    args = parser.parse_args()
    model_filepath = args.model_filepath
    input_filepath = args.input_filepath
    output_filepath = args.output_filepath
    
    # 加载模型
    model = joblib.load(model_filepath)
    
    # 模型推理
    example = pickle.load(open(input_filepath, 'rb'))  # 读取输入数据
    # 提取所需特征进行模型预测
    choose_features = ['标准差', '均方根', '方差', '整流平均值', '方根幅值', '峰峰值', '六阶累积量', '均值',
                      '四阶累积量', '最小值']
    
        
    # 以打印输出的形式返回故障诊断结果
    # print(prediction)

    num_examples = example.shape[0]
    num_has_fault = 0  # 记录有故障的样本的数量
    x_axis = []  # 横坐标，即样本的索引
    predictions = [] # 预测结果
    for i in range(num_examples):
        # if not multiple_sensor:
        #     # 单传感器的模型预测
        #     random_forest_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/random_forest'
        #                                         '/random_forest_model_2.pkl')
        #     random_forest_predictions = random_forest_model.predict(example[choose_features][i:i+1])
        # else:
        #     # 多传感器的模型预测
        #     random_forest_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/random_forest'
        #                                         '/mutli_random_forest_model.pkl')
        #     random_forest_predictions = random_forest_model.predict(example[choose_features_multiple][i:i+1])
        prediction = model.predict(example[choose_features][i:i+1])

        # 统计有故障的样本的数量，同时记录下有故障样本的索引
        if prediction[0] == 1:
            num_has_fault += 1
            x_axis.append(f'样本{i+1}（有故障）')
            predictions.append(1)
            # print(f'有故障的样本索引：{i+1}')
        else:
            x_axis.append(f'样本{i+1}（无故障）')
            predictions.append(0)
            # print(f'无故障的样本索引：{i+1}')
    num_has_not_fault = num_examples - num_has_fault
    indicator = {}

    # 将example按列划分保存到indicator中，其中indicator中的key为列名，value为该列的值
    # for col in example.columns:
    #     indicator[col] = example[col].to_list()
    scaler = StandardScaler()
    for col in choose_features:
        # 将列转换为二维数组，因为 MinMaxScaler 需要二维输入
        column_data = example[[col]].values
        # 进行归一化
        scaled_column = scaler.fit_transform(column_data)
        # 精确到小数点后三位
        indicator[col] = [round(num, 3) for num in scaled_column.flatten()]
        # indicator[col] = example[col].to_list()
        # 将归一化后的结果转换为列表并存储到 indicator 中
        indicator[col] = scaled_column.flatten().tolist()
    
    results = {
        'indicator': indicator,
        'x_axis': x_axis,
        'num_has_fault': num_has_fault,
        'num_has_not_fault': num_has_not_fault,
        'predictions': predictions
    }
    print(f'num_has_fault: {num_has_fault}\t num_has_not_fault: {num_has_not_fault}')
    # 保存为pickle
    pickle.dump(results, open(output_filepath, 'wb'))
