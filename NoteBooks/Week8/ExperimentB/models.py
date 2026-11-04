import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MultiScaleConv1D, self).__init__()
        # To capture multiple temporal scales (e.g., QRS ~ tens of ms, ST/T ~ hundreds of ms)
        branch_out = out_channels // 4
        out1 = branch_out
        out2 = branch_out
        out3 = branch_out
        out4 = out_channels - (out1 + out2 + out3)
        
        # 1x1 bottleneck to reduce parameters before large kernels
        bottleneck_channels = in_channels if in_channels <= 32 else in_channels // 2
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        
        self.branch1 = nn.Conv1d(bottleneck_channels, out1, kernel_size=7, stride=stride, padding=3, bias=False)
        self.branch2 = nn.Conv1d(bottleneck_channels, out2, kernel_size=15, stride=stride, padding=7, bias=False)
        self.branch3 = nn.Conv1d(bottleneck_channels, out3, kernel_size=31, stride=stride, padding=15, bias=False)
        
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=stride, padding=1) if stride > 1 else nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out4, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        bottleneck_x = self.bottleneck(x)
        b1 = self.branch1(bottleneck_x)
        b2 = self.branch2(bottleneck_x)
        b3 = self.branch3(bottleneck_x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class MultiScaleResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(MultiScaleResidualBlock1D, self).__init__()
        self.conv1 = MultiScaleConv1D(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MultiScaleConv1D(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = out + identity
        out = self.relu(out)
        return out

class MultiScaleResNet1D(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiScaleResNet1D, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv1d(12, self.in_channels, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(32, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            
        layers = []
        layers.append(MultiScaleResidualBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(MultiScaleResidualBlock1D(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
