import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


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