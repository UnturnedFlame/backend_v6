import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import scipy.io

# 基于单传感器的知识型 1D 时域深度学习故障诊断技术.结合传统机器学习故障检测与诊断的信号特征知识图谱
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureExtractor:
    @staticmethod
    def knowledge_driven_features(signal):
        mean = np.mean(signal, axis=1, keepdims=True)
        variance = np.var(signal, axis=1, keepdims=True)
        max_value = np.max(signal, axis=1, keepdims=True)
        min_value = np.min(signal, axis=1, keepdims=True)
        return np.concatenate([mean, variance, max_value, min_value], axis=1)


class DeepFeatureExtractor(nn.Module):
    def __init__(self, input_channels, feature_dim):
        super(DeepFeatureExtractor, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, feature_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.squeeze(-1)


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Binary classification
        )

    def forward(self, x):
        return self.fc_layers(x)


class FaultPredictionModel(nn.Module):
    def __init__(self, input_length, deep_feature_dim):
        super(FaultPredictionModel, self).__init__()
        self.deep_extractor = DeepFeatureExtractor(input_channels=1, feature_dim=deep_feature_dim)
        self.classifier = Classifier(input_dim=deep_feature_dim + 4)  # Deep features + knowledge-driven features

    def forward(self, x):
        # Knowledge-driven features
        batch_size = x.shape[0]
        x_np = x.cpu().numpy() if x.is_cuda else x.numpy()
        kd_features = FeatureExtractor.knowledge_driven_features(x_np)
        kd_features = torch.tensor(kd_features, dtype=torch.float32, device=x.device)

        # 1D Deep learning features
        x_deep_features = self.deep_extractor(x.unsqueeze(1))

        # Feature Fusion
        fused_features = torch.cat((x_deep_features, kd_features), dim=1)

        # Classification
        output = self.classifier(fused_features)
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
    input_length = 2048  # 输入长度
    deep_feature_dim = 16  # 深度特征维度

    model_save_path = 'app1/module_management/algorithms/models/fault_diagnosis/single_sensor_1_best_fault_model.pth'
    params_path = 'app1/module_management/algorithms/models/resources/single_sensor_means_stds.npz'

    # 加载模型
    model = FaultPredictionModel(input_length=input_length, deep_feature_dim=deep_feature_dim)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model = model.to(device)

    additional_params = np.load(params_path)
    mean = additional_params['mean']
    std = additional_params['std']
    predicted_class = predict(model, input_signal, mean, std)  # 预测故障类型

    return predicted_class
