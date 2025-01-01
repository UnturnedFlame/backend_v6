import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torchaudio
import torchsummary


class SEBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel=3, padding=1):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=1,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


# 针对单传感器的时频图的卷积神经网络
class SimModelSingle(nn.Module):
    def __init__(self):
        super().__init__()
        n_mels = 40

        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=50000, n_fft=512, win_length=400,
                                                            hop_length=200, window_fn=torch.hamming_window,
                                                            n_mels=n_mels)

        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.hidden1 = SEBottleneck(inplanes=16, planes=32, kernel=3)
        self.hidden2 = SEBottleneck(inplanes=32, planes=64, kernel=3)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.log_input = True
        self.fc = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        B, _ = x.size()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x) + 1e-6
                if self.log_input: x = x.log()
                x = self.instancenorm(x).unsqueeze(1).detach()

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.hidden2(x)

        x = self.avg(x)
        x = x.view(B, -1)
        x = self.fc(x)
        x = self.sig(x)
        return x


# 针对多传感器的时频图卷积神经网络
class SimModelMultiple(nn.Module):
    def __init__(self):
        super().__init__()
        n_mels = 40

        # sample_rate = 16000
        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=50000, n_fft=512, win_length=400,
                                                            hop_length=200, window_fn=torch.hamming_window,
                                                            n_mels=n_mels)

        self.conv = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.hidden1 = SEBottleneck(inplanes=16, planes=32, kernel=3)
        self.hidden2 = SEBottleneck(inplanes=32, planes=64, kernel=3)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.log_input = True
        self.fc = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        B, _, _ = x.size()
        x = torch.split((x), 1, dim=1)
        x1 = x[0].view(B, -1)
        x2 = x[1].view(B, -1)
        x3 = x[2].view(B, -1)
        x4 = x[3].view(B, -1)
        x5 = x[4].view(B, -1)
        x6 = x[5].view(B, -1)
        x7 = x[6].view(B, -1)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x1 = self.torchfb(x1) + 1e-6
                x2 = self.torchfb(x2) + 1e-6
                x3 = self.torchfb(x3) + 1e-6
                x4 = self.torchfb(x4) + 1e-6
                x5 = self.torchfb(x5) + 1e-6
                x6 = self.torchfb(x6) + 1e-6
                x7 = self.torchfb(x7) + 1e-6
                if self.log_input:
                    x1 = x1.log()
                    x2 = x2.log()
                    x3 = x3.log()
                    x4 = x4.log()
                    x5 = x5.log()
                    x6 = x6.log()
                    x7 = x7.log()
                x1 = self.instancenorm(x1).unsqueeze(1).detach()
                x2 = self.instancenorm(x2).unsqueeze(1).detach()
                x3 = self.instancenorm(x3).unsqueeze(1).detach()
                x4 = self.instancenorm(x4).unsqueeze(1).detach()
                x5 = self.instancenorm(x5).unsqueeze(1).detach()
                x6 = self.instancenorm(x6).unsqueeze(1).detach()
                x7 = self.instancenorm(x7).unsqueeze(1).detach()
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.hidden2(x)

        x = self.avg(x)
        x = x.view(B, -1)
        x = self.fc(x)
        x = self.sig(x)
        return x
