import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np


"""
基于单传器的多域深度特征融合故障检测与诊断技术.增加信号的时频表征 的 2D 深度学习，共同构成时域-频域-时频域的深度特征融合，
增加其故障检 测与预测的可靠性.
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()

    def forward(self, x):
        # Extracting statistical features from time and frequency domain
        max_value, _ = torch.max(x, dim=-1, keepdim=True)
        min_value, _ = torch.min(x, dim=-1, keepdim=True)
        mean_value = torch.mean(x, dim=-1, keepdim=True)
        std_value = torch.std(x, dim=-1, keepdim=True)

        features = torch.cat([max_value, min_value, mean_value, std_value], dim=-1)
        return features


class OneDConvNet(nn.Module):
    def __init__(self, in_channels, sequence_length):
        super(OneDConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)

        # Dynamically compute the size after convolution
        conv_output_size = sequence_length
        self.fc1 = nn.Linear(32 * conv_output_size, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = F.relu(self.fc1(x))
        return x


class TwoDConvNet(nn.Module):
    def __init__(self, in_channels, tf_size):
        super(TwoDConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # 静态计算卷积输出特征数量
        self.fc1 = nn.Linear(68640, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = F.relu(self.fc1(x))
        return x


class FourBranchModel(nn.Module):
    def __init__(self, sequence_length=2048, tf_size=65):  # Adjust tf_size based on n_fft and hop_length
        super(FourBranchModel, self).__init__()
        self.feature_extraction = FeatureExtraction()
        self.oned_conv_net_time = OneDConvNet(in_channels=1, sequence_length=sequence_length)
        self.oned_conv_net_freq = OneDConvNet(in_channels=1, sequence_length=sequence_length)
        self.twod_conv_net = TwoDConvNet(in_channels=2, tf_size=tf_size)  # Adjust in_channels to 2 (real + imag)

        # Adjust the size of the merged features
        self.fc_merge = nn.Linear(4 + 128 + 128 + 128, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        # Branch 1: Time and Frequency Domain Feature Extraction
        features_time_freq = self.feature_extraction(x)

        # Branch 2: 1D ConvNet for Time Domain Feature Extraction
        features_time = self.oned_conv_net_time(x.unsqueeze(1))

        # Branch 3: 1D ConvNet for Frequency Domain Feature Extraction
        x_freq = torch.fft.fft(x)
        features_freq = self.oned_conv_net_freq(x_freq.real.unsqueeze(1))

        # Branch 4: 2D ConvNet for Time-Frequency Representation
        x_tf = torch.stft(x, n_fft=128, hop_length=64, return_complex=True)
        x_tf = torch.view_as_real(x_tf)  # Convert complex to real-valued tensor
        x_tf = x_tf.permute(0, 3, 1, 2)  # Adjust dimensions to [batch, channels, height, width]
        features_tf = self.twod_conv_net(x_tf)

        # Merge all features
        merged_features = torch.cat([features_time_freq, features_time, features_freq, features_tf], dim=-1)
        merged_features = F.relu(self.fc_merge(merged_features))
        merged_features = F.relu(self.fc1(merged_features))
        merged_features = F.relu(self.fc2(merged_features))

        # Classification
        output = self.classifier(merged_features)
        return output


def train_model(model, dataset, num_epochs, learning_rate, save_path):
    """
    训练 FaultPredictionModel。

    Args:
        model (nn.Module): 模型实例。
        dataset (Dataset): PyTorch 数据集。
        num_epochs (int): 训练轮数。
        learning_rate (float): 学习率。
        save_path (str): 模型保存路径。

    Returns:
        dict: 训练历史（每轮的损失值）。
    """
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    history = {'loss': []}

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        history['loss'].append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # 保存最优模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} with loss {epoch_loss:.4f}")

    return history


def predict(model, signal, mean, std):
    """
    使用训练好的模型对单个信号进行预测，包含标准化。

    Args:
        model (nn.Module): 训练好的 FaultPredictionModel。
        signal (numpy.ndarray): 输入信号，形状为 (2048,)。
        mean (numpy.ndarray): 用于标准化的均值，形状为 (2048,)。
        std (numpy.ndarray): 用于标准化的标准差，形状为 (2048,)。

    Returns:
        int: 预测的类别 (0 或 1)。
    """
    model.eval()
    with torch.no_grad():
        # 检查输入信号的长度是否为 2048
        if signal.shape[0] != 2048:
            raise ValueError("Input signal must have length 2048.")

        # 标准化输入信号
        signal_normalized = (signal - mean) / std

        # 转换 numpy 数组为 Tensor，并添加批次维度
        signal_tensor = torch.tensor(signal_normalized, dtype=torch.float32).unsqueeze(0).to(device)

        # 模型前向传播
        output = model(signal_tensor)

        # 获取预测类别
        _, predicted_class = torch.max(output, dim=1)

    return predicted_class.item()


# 模型推理故障诊断接口
def fault_diagnose(input_signal):
    """
    :param input_signal: 输入信号
    :return: 预测故障类型，0表示无故障，1表示有故障
    """

    model_save_path = 'app1/module_management/algorithms/models/fault_diagnosis/single_sensor_3_best_fault_model.pth'
    params_path = 'app1/module_management/algorithms/models/resources/single_sensor_means_stds.npz'

    # 加载模型
    model = FourBranchModel(sequence_length=2048, tf_size=64)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model = model.to(device)

    additional_params = np.load(params_path)
    mean = additional_params['mean']
    std = additional_params['std']
    predicted_class = predict(model, input_signal, mean, std)  # 预测故障类型

    return predicted_class
