import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim

# 基于多传感器特征级融合的深度学习故障检测与诊断技术

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiSensorFaultPrediction(nn.Module):
    def __init__(self, num_sensors, num_classes=2):
        super(MultiSensorFaultPrediction, self).__init__()
        self.num_sensors = num_sensors
        # Define individual feature extractors for each sensor
        self.feature_extractors = nn.ModuleList(
            [nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1) for _ in range(num_sensors)])
        self.feature_extractors_bn = nn.ModuleList([nn.BatchNorm1d(16) for _ in range(num_sensors)])
        # Feature alignment and fusion layer
        self.fusion_conv = nn.Conv1d(16 * num_sensors, 64, kernel_size=3, stride=1, padding=1)
        self.fusion_bn = nn.BatchNorm1d(64)
        # Classifier layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: input tensor of shape (batch_size, sequence_length, num_sensors)
        # Transpose to (batch_size, num_sensors, sequence_length)
        x = x.permute(0, 2, 1)

        # Process each sensor separately
        features = []
        for i in range(self.num_sensors):
            # Extract the i-th sensor's data (shape: batch_size, 1, sequence_length)
            sensor_data = x[:, i:i + 1, :]
            feature = F.relu(self.feature_extractors_bn[i](self.feature_extractors[i](sensor_data)))
            features.append(feature)

        # Align features along the channel dimension
        aligned_features = torch.cat(features, dim=1)  # shape: (batch_size, 16 * num_sensors, sequence_length)

        # Feature fusion
        fusion_output = F.relu(
            self.fusion_bn(self.fusion_conv(aligned_features)))  # shape: (batch_size, 64, sequence_length)

        # Global average pooling
        fusion_output = torch.mean(fusion_output, dim=2)  # shape: (batch_size, 64)

        # Classification
        x = F.relu(self.fc1(fusion_output))  # shape: (batch_size, 128)
        x = F.relu(self.fc2(x))  # shape: (batch_size, 64)
        x = F.relu(self.fc3(x))  # shape: (batch_size, 32)
        x = self.fc4(x)  # shape: (batch_size, num_classes)

        return x


def train_model(model, dataloader, num_epochs, learning_rate, save_path):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    history = {'loss': []}

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        history['loss'].append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with loss {epoch_loss:.4f}")

    return history


def predict(model, signal, mean, std):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # Normalize input signal
        signal_normalized = (signal - mean) / std

        # Convert to tensor and add batch dimension
        signal_tensor = torch.tensor(signal_normalized, dtype=torch.float32).unsqueeze(0).to(device)

        # Forward pass
        output = model(signal_tensor)

        # Get predicted class
        _, predicted_class = torch.max(output, dim=1)

    return predicted_class.item()


# 模型推理故障诊断接口
def fault_diagnose(input_signal):
    """
    :param input_signal: 输入信号
    :return: 预测故障类型，0表示无故障，1表示有故障
    """
    seq_length = 2048
    num_sensors = 7
    num_classes = 2

    model_save_path = 'app1/module_management/algorithms/models/fault_diagnosis/mutli_sensor_3_fault_model_best.pth'
    params_path = 'app1/module_management/algorithms/models/resources/mutli_sensor_means_stds.npz'

    # 加载模型
    model = MultiSensorFaultPrediction(num_sensors=num_sensors, num_classes=num_classes)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

    additional_params = np.load(params_path)
    mean = additional_params['mean']
    std = additional_params['std']
    predicted_class = predict(model, input_signal, mean, std)  # 预测故障类型

    return predicted_class


