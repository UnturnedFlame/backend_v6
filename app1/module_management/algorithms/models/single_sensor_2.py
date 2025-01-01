import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

# 基于单传感器的时域和频域协同注意学习故障诊断技术

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Deep FeatureExtractor
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)  # Adjusted channels
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # Adjusted channels
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.mean(x, dim=-1)  # Global average pooling
        x = self.fc(x)
        return x


# CoAttention
class CoAttention(nn.Module):
    def __init__(self, input_dim):
        super(CoAttention, self).__init__()
        self.attn = nn.Linear(input_dim, input_dim)

    def forward(self, x1, x2):
        attn_weights1 = torch.sigmoid(self.attn(x1))
        attn_weights2 = torch.sigmoid(self.attn(x2))
        x1_attn = x1 * attn_weights1
        x2_attn = x2 * attn_weights2
        return x1_attn + x2_attn  # Residual connection


# TimeFrequencyModel
class TimeFrequencyModel(nn.Module):
    def __init__(self, input_length, feature_dim, num_classes):
        super(TimeFrequencyModel, self).__init__()
        self.time_feature_extractor = FeatureExtractor(input_dim=1, output_dim=feature_dim)
        self.freq_feature_extractor = FeatureExtractor(input_dim=1, output_dim=feature_dim)
        self.co_attention = CoAttention(feature_dim)
        # Adjusted input size for concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + 6, 256),  # Updated to match concatenated feature size
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Extract time-domain and frequency-domain inputs
        time_input = x.unsqueeze(1)  # Add channel dimension for CNN
        freq_input = x.unsqueeze(1)  # Same for frequency

        # Extract knowledge-driven features
        time_knowledge_features = extract_time_domain_features(x)
        freq_knowledge_features = extract_frequency_domain_features(x)

        # Extract deep features using CNN
        time_features = self.time_feature_extractor(time_input)
        freq_features = self.freq_feature_extractor(freq_input)

        # Apply co-attention mechanism
        fused_features = self.co_attention(time_features, freq_features)

        # Concatenate all features
        combined_features = torch.cat((time_knowledge_features, freq_knowledge_features, fused_features), dim=-1)

        # Classification
        out = self.classifier(combined_features)
        return out


# Feature extraction functions
def extract_time_domain_features(signal):
    mean = torch.mean(signal, dim=-1, keepdim=True)
    std = torch.std(signal, dim=-1, keepdim=True)
    max_val, _ = torch.max(signal, dim=-1, keepdim=True)
    min_val, _ = torch.min(signal, dim=-1, keepdim=True)
    return torch.cat((mean, std, max_val, min_val), dim=-1)


def extract_frequency_domain_features(signal):
    fft_signal = torch.fft.rfft(signal, dim=-1)
    power = torch.abs(fft_signal) ** 2
    mean_power = torch.mean(power, dim=-1, keepdim=True)
    max_power, _ = torch.max(power, dim=-1, keepdim=True)
    return torch.cat((mean_power, max_power), dim=-1)


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
    input_length = 2048  # 输入长度
    deep_feature_dim = 16  # 深度特征维度

    model_save_path = 'app1/module_management/algorithms/models/fault_diagnosis/single_sensor_2_best_fault_model.pth'
    params_path = 'app1/module_management/algorithms/models/resources/single_sensor_means_stds.npz'

    # 加载模型
    model = TimeFrequencyModel(input_length=input_length, feature_dim=deep_feature_dim, num_classes=2)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model = model.to(device)

    additional_params = np.load(params_path)
    mean = additional_params['mean']
    std = additional_params['std']
    predicted_class = predict(model, input_signal, mean, std)  # 预测故障类型

    return predicted_class
