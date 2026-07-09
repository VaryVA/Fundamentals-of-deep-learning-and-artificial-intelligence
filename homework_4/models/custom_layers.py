import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class CustomActivation(Function):
    '''Функция активации с ручной реализацией backward'''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * torch.sigmoid(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigmoid = torch.sigmoid(input)
        grad_input = grad_output * (sigmoid + input * sigmoid * (1 - sigmoid))
        return grad_input

class CustomPool(nn.Module):
    '''Смесь Max и Average pooling'''
    def __init__(self, kernel_size=2, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.max_pool = nn.MaxPool2d(kernel_size, kernel_size)
        self.avg_pool = nn.AvgPool2d(kernel_size, kernel_size)

    def forward(self, x):
        return self.alpha * self.max_pool(x) + (1 - self.alpha) * self.avg_pool(x)

class CNNAttention(nn.Module):
    '''Простой Spatial Attention механизм для CNN'''
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn

class CustomConv2d(nn.Module):
    '''Сверточный слой с learnable temperature'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = self.bn(self.conv(x))
        return F.relu(out) * self.temperature

class BasicBlock(nn.Module):
    '''Базовый Residual блок'''
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class BottleneckBlock(nn.Module):
    '''Bottleneck Residual блок'''
    def __init__(self, in_ch, out_ch, stride=1, expansion=4):
        super().__init__()
        mid_ch = out_ch // expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class WideBlock(nn.Module):
    '''Wide Residual блок'''
    def __init__(self, in_ch, out_ch, stride=1, widen_factor=2):
        super().__init__()
        mid_ch = out_ch * widen_factor
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)