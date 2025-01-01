import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.signal import stft
# from mutli_dataset import RandomSequenceDataset

# 基于多传感器信号时频表征自适应加权融合的故障检测与诊断技术
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class STFTFusionClassifier(nn.Module):
    def __init__(self, num_sensors, seq_length, nperseg, hidden_dim, num_classes):
        super(STFTFusionClassifier, self).__init__()
        self.num_sensors = num_sensors
        self.seq_length = seq_length
        self.nperseg = nperseg

        # Learnable weights for sensor fusion
        self.sensor_weights = nn.Parameter(torch.ones(num_sensors))

        # Fully connected layers for classification
        # input_dim = stft_height * stft_width
        input_dim = 2145

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, num_sensors)
        batch_size, seq_length, num_sensors = x.shape
        assert num_sensors == self.num_sensors, "Input sensor count does not match model configuration"

        # Compute STFT for each sensor
        stft_results = []
        for i in range(self.num_sensors):
            sensor_data = x[:, :, i]  # shape: (batch_size, seq_length)
            sensor_stft = torch.stack(
                [torch.from_numpy(np.abs(stft(signal.cpu().numpy(), nperseg=self.nperseg)[2])) for signal in
                 sensor_data],
                dim=0
            )  # shape: (batch_size, stft_height, stft_width)
            stft_results.append(sensor_stft)

        # Stack STFT results along the sensor axis
        stft_results = torch.stack(stft_results, dim=1)  # shape: (batch_size, num_sensors, stft_height, stft_width)
        stft_results = stft_results.to(device)
        # Fusion along the sensor axis using learnable weights
        self.sensor_weights = self.sensor_weights.to(device)
        sensor_weights = F.softmax(self.sensor_weights, dim=0)  # Normalize weights
        fused_stft = torch.sum(sensor_weights.view(1, -1, 1, 1) * stft_results,
                               dim=1)  # shape: (batch_size, stft_height, stft_width)

        # Flatten and pass through the classifier
        fused_flat = fused_stft.view(batch_size, -1)  # shape: (batch_size, stft_height * stft_width)
        x = F.relu(self.fc1(fused_flat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
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
    nperseg = 64
    hidden_dim = 128
    num_classes = 2

    model_save_path = 'app1/module_management/algorithms/models/fault_diagnosis/mutli_sensor_2_fault_model_best.pth'
    params_path = 'app1/module_management/algorithms/models/resources/mutli_sensor_means_stds.npz'

    # 加载模型
    model = STFTFusionClassifier(
        num_sensors=num_sensors,
        seq_length=seq_length,
        nperseg=nperseg,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

    additional_params = np.load(params_path)
    mean = additional_params['mean']
    std = additional_params['std']
    predicted_class = predict(model, input_signal, mean, std)  # 预测故障类型

    return predicted_class

