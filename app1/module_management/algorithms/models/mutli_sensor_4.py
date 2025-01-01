import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

# 多传感器决策级融合的深度学习故障检测与诊断是在每个传感器的深度学
# 习故障检测与诊断基础上，根据每个传感器的故障分类匹配得分，

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define 1D CNN module for feature extraction
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x


# Define classifier module for each sensor
class SensorClassifier(nn.Module):
    def __init__(self, input_features):
        super(SensorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define complete model
class MultiSensorModel(nn.Module):
    def __init__(self, input_channels, sequence_length, num_sensors):
        super(MultiSensorModel, self).__init__()
        # Each sensor is processed individually, so input_channels=1 for each CNN
        self.feature_extractors = nn.ModuleList([CNNFeatureExtractor(1) for _ in range(num_sensors)])
        flattened_feature_length = (sequence_length // 8) * 64  # Adjust based on pooling
        self.classifiers = nn.ModuleList([SensorClassifier(flattened_feature_length) for _ in range(num_sensors)])

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Shape: (batch_size, n_sensors, sequence_length)
        outputs = []
        for i in range(len(self.feature_extractors)):
            sensor_input = x[:, i, :].unsqueeze(1)  # Shape: (batch_size, 1, sequence_length)
            features = self.feature_extractors[i](sensor_input)
            features = features.view(features.size(0), -1)  # Flatten for the linear layer
            output = self.classifiers[i](features)
            outputs.append(output)
        final_predictions = self.get_final_prediction(outputs)
        return outputs, final_predictions

    def get_final_prediction(self, predictions):
        # predictions: list of tensors from each classifier
        predictions = torch.stack(predictions, dim=0)  # Shape: (n_sensors, batch_size, n_classes)
        predictions = torch.argmax(predictions, dim=2)  # Get class indices, Shape: (n_sensors, batch_size)
        predictions = predictions.transpose(0, 1)  # Shape: (batch_size, n_sensors)
        final_predictions = []
        for batch in predictions:
            values, counts = torch.unique(batch, return_counts=True)
            final_predictions.append(values[torch.argmax(counts)])
        return torch.stack(final_predictions)


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
            optimizer.zero_grad()
            outputs, final_predictions = model(batch_x)
            loss = 0
            for pred in outputs:
                loss += criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        history['loss'].append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} with loss {epoch_loss:.4f}")

    return history


def predict(model, signal, mean, std):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # Normalize input signal
        signal_normalized = (signal - mean) / std

        # Convert to tensor and add batch dimension
        signal_tensor = torch.tensor(signal_normalized, dtype=torch.float32).unsqueeze(0).to(device)

        # Get predicted class
        _, final_predictions = model(signal_tensor)
    return final_predictions.item()


# 模型推理故障诊断接口
def fault_diagnose(input_signal):
    """
    :param input_signal: 输入信号
    :return: 预测故障类型，0表示无故障，1表示有故障
    """
    seq_length = 2048
    num_sensors = 7
    num_classes = 2

    model_save_path = 'app1/module_management/algorithms/models/fault_diagnosis/mutli_sensor_4_fault_model_best.pth'
    params_path = 'app1/module_management/algorithms/models/resources/mutli_sensor_means_stds.npz'

    # 加载模型
    model = MultiSensorModel(input_channels=1, sequence_length=seq_length, num_sensors=num_sensors)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

    additional_params = np.load(params_path)
    mean = additional_params['mean']
    std = additional_params['std']
    predicted_class = predict(model, input_signal, mean, std)  # 预测故障类型

    return predicted_class
