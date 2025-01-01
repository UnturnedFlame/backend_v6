import pickle
import numpy as np
import argparse
import torch.nn as nn
import torch

# LSTM网络
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq)  # output(40, 2048, 100)
        pred = self.linear(output)  # (40, 2048, 1)
        pred = pred[:, -1, :]  # (40, 1)
        return pred.squeeze(1)


if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--model-filepath', type=str, default = None)  # 模型参数
    parser.add_argument('--input-filepath', type=str, default = None)  # 输入数据

    # 解析命令行参数
    args = parser.parse_args()
    model_filepath = args.model_filepath
    input_filepath = args.input_filepath
    
    # 加载lstm模型
    lstm_model = LSTM(input_size=7, hidden_size=64, num_layers=4, output_size=1).to(device)
    lstm_model.load_state_dict(torch.load(model_filepath))
    lstm_model.eval()
    
    # 模型推理
    example = np.load(input_filepath)
    inputs = torch.from_numpy(example).type(torch.FloatTensor).reshape((-1, 2048, 7)).to(device)
    predicted = lstm_model.forward(inputs)
    
    # 根据阈值判断有无故障
    if predicted.reshape(-1).item() < 0:
        prediction = 0
    else:
        prediction = 1
        
    # 以打印输出的形式返回故障诊断结果
    print(prediction)
    
