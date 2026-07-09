import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    '''Residual блок (models.py)'''
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleCNN(nn.Module):
    '''Простая CNN для MNIST'''
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNWithResidual(nn.Module):
    '''CNN с Residual блоками для MNIST'''
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CIFARCNN(nn.Module):
    '''CNN для CIFAR-10'''
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFARResNet(nn.Module):
    '''CNN с регуляризацией и Residual блоками для CIFAR-10'''
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 32, 2)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(self.dropout(x))

class KernelSizeCNN(nn.Module):
    '''Модель для эксперимента с размером ядра свертки'''
    def __init__(self, in_ch=1, num_classes=10, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # Подбираем каналы для примерно равного числа параметров
        ch1, ch2 = 64, 128
        if kernel_size == 5:
            ch1, ch2 = 40, 80
        elif kernel_size == 7:
            ch1, ch2 = 28, 56

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, ch1, kernel_size, padding=padding), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(ch1, ch2, kernel_size, padding=padding), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(ch2 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

class DepthCNN(nn.Module):
    '''Модель для эксперимента с глубиной CNN'''
    def __init__(self, in_ch=1, num_classes=10, depth=2, use_residual=False):
        super().__init__()
        layers = []
        current_ch = in_ch
        out_ch = 32
        for i in range(depth):
            if use_residual and i > 0:
                layers.append(ResidualBlock(current_ch, out_ch, stride=2 if i % 2 == 1 else 1))
            else:
                stride = 2 if i > 0 and not use_residual else 1
                layers.extend([
                    nn.Conv2d(current_ch, out_ch, 3, stride=stride, padding=1),
                    nn.BatchNorm2d(out_ch), nn.ReLU()
                ])
            current_ch = out_ch
            if not use_residual and i % 2 == 1:
                out_ch *= 2

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(current_ch * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))