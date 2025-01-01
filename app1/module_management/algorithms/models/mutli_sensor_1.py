import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 基于多传感器信号级加权融合的故障检测与诊断技术

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureExtractor:
    @staticmethod
    def knowledge_driven_features(signal):
        mean = np.mean(signal, axis=2, keepdims=True)
        variance = np.var(signal, axis=2, keepdims=True)
        max_value = np.max(signal, axis=2, keepdims=True)
        min_value = np.min(signal, axis=2, keepdims=True)
        return np.concatenate([mean, variance, max_value, min_value], axis=2)


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
    def __init__(self, num_sensors, input_length, deep_feature_dim):
        super(FaultPredictionModel, self).__init__()
        self.num_sensors = num_sensors
        self.deep_extractor = DeepFeatureExtractor(input_channels=1, feature_dim=deep_feature_dim)
        self.sensor_weights = nn.Parameter(torch.ones(num_sensors))  # Learnable sensor weights
        self.classifier = Classifier(
            input_dim=num_sensors * (deep_feature_dim + 4))  # Deep features + knowledge-driven features

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # Apply learnable weights to each sensor's input
        weights = torch.softmax(self.sensor_weights, dim=0)
        x_weighted = x * weights.view(1, -1, 1)

        # Knowledge-driven features
        batch_size = x.shape[0]
        x_np = x_weighted.detach().cpu().numpy()
        kd_features = FeatureExtractor.knowledge_driven_features(x_np)
        kd_features = torch.tensor(kd_features, dtype=torch.float32, device=device)

        # 1D Deep learning features for each sensor
        deep_features = []
        for i in range(self.num_sensors):
            sensor_input = x_weighted[:, i, :]
            deep_feature = self.deep_extractor(sensor_input.unsqueeze(1))
            deep_features.append(deep_feature)
        deep_features = torch.stack(deep_features, dim=1)

        # Concatenate knowledge-driven and deep features for each sensor
        kd_features = kd_features.view(batch_size, self.num_sensors, -1)
        fused_features = torch.cat((deep_features, kd_features), dim=2)
        fused_features = fused_features.view(batch_size, -1)

        # Classification
        output = self.classifier(fused_features)
        return output


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
    deep_feature_dim = 16

    model_save_path = 'app1/module_management/algorithms/models/fault_diagnosis/mutli_sensor_1_fault_model_best.pth'
    params_path = 'app1/module_management/algorithms/models/resources/mutli_sensor_means_stds.npz'

    # 加载模型
    model = FaultPredictionModel(num_sensors=num_sensors, input_length=seq_length, deep_feature_dim=deep_feature_dim)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

    additional_params = np.load(params_path)
    mean = additional_params['mean']
    std = additional_params['std']
    predicted_class = predict(model, input_signal, mean, std)  # 预测故障类型

    return predicted_class


