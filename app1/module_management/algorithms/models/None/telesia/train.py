import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.residualMoudle = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
        )

    def forward(self, input):
        x0 = input
        x = self.residualMoudle(x0)
        x = x + x0
        return x


class MCL_DRL(nn.Module):
    def __init__(self, num_classes):
        super(MCL_DRL, self).__init__()

        self.conv1d_0 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU(inplace=False)
        )

        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=64, stride=7),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.ReLU(inplace=False)
        )

        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=32, stride=3, padding=14),
            nn.BatchNorm1d(32),

        )
        self.lstm_1 = nn.LSTM(batch_first=True, input_size=32, hidden_size=32, num_layers=2)

        self.conv1d_3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=16, stride=3, padding=6),
            nn.BatchNorm1d(32),
        )
        self.lstm_2 = nn.LSTM(batch_first=True, input_size=32, hidden_size=32, num_layers=2)

        self.conv1d_4 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=3, padding=2),
            nn.BatchNorm1d(32),
        )
        self.lstm_3 = nn.LSTM(batch_first=True, input_size=32, hidden_size=32, num_layers=2)

        # max-pooling after depth concat
        self.maxpool1d = nn.MaxPool1d(kernel_size=3, stride=1)

        # residual module
        self.residualMoudle = ResBlock()
        # self.residualMoudle = nn.Sequential(
        #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(inplace=False),
        #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm1d(32),
        # )

        self.lstm_4 = nn.LSTM(batch_first=True, input_size=32, hidden_size=32, num_layers=2)

        # full connect
        self.fc = nn.Sequential(
            nn.Linear(in_features=4256, out_features=64),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x):
        # unsqueeze
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)

        # multi-scale convolution
        x1 = self.conv1d_2(x)
        x1, _ = self.lstm_1(x1.view(x1.size(0), x1.size(2), x1.size(1)))
        x1 = x1.reshape(x1.size(0), x1.size(2), x1.size(1))
        x1 = F.relu(x1)

        x2 = self.conv1d_3(x)
        x2, _ = self.lstm_2(x2.view(x2.size(0), x2.size(2), x2.size(1)))
        x2 = x2.reshape(x2.size(0), x2.size(2), x2.size(1))
        x2 = F.relu(x2)

        x3 = self.conv1d_4(x)
        x3, _ = self.lstm_3(x3.view(x3.size(0), x3.size(2), x3.size(1)))
        x3 = x3.reshape(x3.size(0), x3.size(2), x3.size(1))
        x3 = F.relu(x3)

        # depth concat
        x4 = torch.cat([x1, x2, x3], dim=2)
        x4 = self.maxpool1d(x4)

        x0 = self.residualMoudle(x4)
        x0, _ = self.lstm_4(x0.view(x0.size(0), x0.size(2), x0.size(1)))
        # depth concat and residual connection
        x4 = x4 + x0.reshape(x0.size(0), x0.size(2), x0.size(1))

        # full connect
        x5 = self.fc(x4.view(x4.size(0), -1))

        return x5


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用无噪声数据训练
# examples_all = np.loadtxt('data/CWRU/12k Drive End Bearing Fault Data/normalized_examples.txt')
# labels_all = np.loadtxt('data/CWRU/12k Drive End Bearing Fault Data/labels_abundant.txt')

# 使用有噪声数据训练
examples_all = np.loadtxt(r'/root/autodl-tmp/dataset/CWRU/12k Driven End/no_overlap/0hp_examples_noised_snr_n8.txt')
labels_all = np.loadtxt(r'/root/autodl-tmp/dataset/CWRU/12k Driven End/no_overlap/0hp_labels_noised_snr_n8.txt')

train_dataset = data.DataLoader(
    data.TensorDataset(torch.from_numpy(examples_all).requires_grad_().type(torch.FloatTensor).to(device=device),
                       torch.from_numpy(labels_all).requires_grad_().type(torch.FloatTensor).to(device)), batch_size=32,
    shuffle=True)

model = MCL_DRL(10).to(device)  # 10 classes classification problem

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = 20

with torch.autograd.set_detect_anomaly(True):
    for i in range(epoch):

        for examples, labels in train_dataset:
            # print(examples.shape)
            # print(examples.requires_grad)

            y_hat = model(examples.unsqueeze(1))  # 添加通道维度
            loss = loss_fn(y_hat, labels)

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

        with torch.no_grad():
            loss_sum = 0
            accuracy = 0
            for examples, labels in train_dataset:
                pre = model(examples.unsqueeze(1))
                loss_sum += loss_fn(pre, labels).item()
                accuracy += (pre.argmax(dim=1) == labels.argmax(dim=1)).sum().item()

        print(f'epoch: {i + 1}, loss: {loss_sum}, accuracy: {accuracy / labels_all.shape[0]}')