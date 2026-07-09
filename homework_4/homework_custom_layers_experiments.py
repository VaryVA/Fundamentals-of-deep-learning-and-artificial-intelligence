import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_mnist_loaders
from models.custom_layers import (
    CustomActivation, CustomPool, CNNAttention, CustomConv2d,
    BasicBlock, BottleneckBlock, WideBlock
)
from utils.training_utils import train_model, setup_logging
from utils.comparison_utils import count_parameters, compare_models_table

os.makedirs('results/custom_layers', exist_ok=True)
setup_logging('results/custom_layers.log')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_mnist_loaders(64)

# 3.1 Тест кастомных слоев
print("3.1 Тест кастомных слоев")

# 1. Модель с кастомной активацией
class TestCustomAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(CustomActivation.apply(self.conv(x)))
        return self.fc(x.view(x.size(0), -1))

# 2. Модель с кастомным пулингом
class TestCustomPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = CustomPool(kernel_size=2, alpha=0.6)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        return self.fc(x.view(x.size(0), -1))

# 3. Модель с Attention
class TestAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.attn = CNNAttention(16)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.attn(x)
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))

# 4. Модель с кастомным Conv2d
class TestCustomConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = CustomConv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.conv(x))
        return self.fc(x.view(x.size(0), -1))

custom_results = {}
custom_models = {
    "Custom Activation (Swish)": TestCustomAct().to(device),
    "Custom Pool (Max+Avg)": TestCustomPool().to(device),
    "CNN Attention": TestAttention().to(device),
    "Custom Conv2d (Temp)": TestCustomConv().to(device)
}

for name, model in custom_models.items():
    history = train_model(model, train_loader, test_loader, epochs=3, device=str(device))
    custom_results[name] = {
        "Params": count_parameters(model),
        "Test Acc": f"{history['test_accs'][-1]:.4f}",
        "Train Loss": f"{history['train_losses'][-1]:.4f}"
    }

compare_models_table(custom_results,
                     title="CUSTOM LAYERS COMPARISON",
                     save_path="results/custom_layers/custom_comparison.png")
# 3.2 Варианты Residual блоков
print("3.2 Варианты Residual блоков")

class ResNetVariant(nn.Module):
    def __init__(self, block_class, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.layer1 = block_class(32, 32)
        self.layer2 = block_class(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 16, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.layer2(self.layer1(x))
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))

res_results = {}
res_blocks = {
    "Basic Block": ResNetVariant(BasicBlock).to(device),
    "Bottleneck Block": ResNetVariant(BottleneckBlock).to(device),
    "Wide Block": ResNetVariant(WideBlock).to(device)
}

for name, model in res_blocks.items():
    history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))
    res_results[name] = {
        "Params": count_parameters(model),
        "Test Acc": f"{history['test_accs'][-1]:.4f}",
        "Final Train Loss": f"{history['train_losses'][-1]:.4f}"
    }

compare_models_table(res_results,
                     title="RESIDUAL BLOCKS COMPARISON",
                     save_path="results/custom_layers/residual_comparison.png")
print("\nЗадание 3 завершено!")