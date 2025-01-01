import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init


class channel_shuffle(nn.Module):
    def __init__(self, groups=2):
        super(channel_shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, L = x.size()
        g = self.groups
        return x.view(N, g, C // g, L).permute(0, 2, 1, 3).reshape(N, C, L)


class channelattention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(channelattention, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, T = x.size()
        # Average along each channel
        x_gap = self.gap(x).squeeze()
        x_gmp = self.gmp(x).squeeze()

        # channel excitation
        fc_out_gap = self.fc2(self.relu(self.fc1(x_gap)))
        fc_out_gmp = self.fc2(self.relu(self.fc1(x_gmp)))
        att_map = self.sigmoid(fc_out_gap + fc_out_gmp)
        output_tensor = torch.mul(x, att_map.view(b, c, 1))
        return output_tensor


class dwconv_mobile(nn.Module):
    def __init__(self, in_c=32, out_c=32, kz=5):
        super(dwconv_mobile, self).__init__()
        self.dw = nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=kz, stride=2, padding=(kz-1)//2)
        self.pw = nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        self.shuffle = channel_shuffle()
        self.att = channelattention(out_c)

    def forward(self, x):
        return self.att(self.shuffle(self.relu(self.bn(self.pw(self.dw(x))))))


class ULCNN(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 feature_dim: int = 3,
                 num_layers: int = 6,
                 encoder_step: int = 32,
                 conv_kernel_size: int = 5,
                 se_reduction_ratio: int = 16,
                 conv_dropout_p: float = 0.1,
                 ):
        super(ULCNN, self).__init__()

        self.input_pro = nn.Conv1d(in_channels=feature_dim, out_channels=encoder_step, kernel_size=1)
        self.layers = nn.ModuleList(
            dwconv_mobile(in_c=encoder_step, out_c=encoder_step, kz=conv_kernel_size
            ) for _ in range(num_layers)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifi = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_features=encoder_step, out_features=num_classes, bias=False),
            nn.Sigmoid()
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x.size() = b,2,128  B, C, L
        # B,_ = x.size()
        # x = x.view(B,1,-1)
        x = self.input_pro(x)    # b, 64, 128

        n = len(self.layers)  # n=6
        out = 0
        for layer in self.layers:
            x = layer(x)
            if n < 4:
                out += self.avgpool(x)
            n = n - 1
        out = self.classifi(out)
        return out


if __name__ == '__main__':
    ULCNN_model = ULCNN(num_classes=1, feature_dim=1, num_layers=6, encoder_step=32, conv_kernel_size=3)
    x = torch.randn(12, 1, 32)
    f = ULCNN_model(x)
    print(f)