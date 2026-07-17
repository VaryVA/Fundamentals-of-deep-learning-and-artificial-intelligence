import os
import time
import copy
import warnings

import matplotlib

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from datasets import CustomImageDataset

# Создаем папку для результатов
os.makedirs('results', exist_ok=True)

# Определяем устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# 1. Подготовка датасета
print("\n Подготовка датасета...")

# Нормализация ImageNet (ОБЯЗАТЕЛЬНА для предобученных моделей)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Для train: аугментации + нормализация
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Для val: ТОЛЬКО ресайз + нормализация (без аугментаций!)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

train_dataset = CustomImageDataset('data/train', transform=train_transform, target_size=(224, 224))
val_dataset = CustomImageDataset('data/val', transform=val_transform, target_size=(224, 224))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

num_classes = len(train_dataset.get_class_names())
print(f"   Классов: {num_classes}")
print(f"   Train изображений: {len(train_dataset)}")
print(f"   Val изображений: {len(val_dataset)}")

# 2. Загрузка предобученной модели
print("\n Загрузка предобученной модели ResNet18...")
model = models.resnet18(weights='IMAGENET1K_V1')

# Замораживаем все слои, кроме последнего
for param in model.parameters():
    param.requires_grad = False

# Заменяем последний слой (fc) под количество наших классов
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)
print(f"   Последний слой заменен: Linear({num_ftrs} -> {num_classes})")

# 3. Настройка обучения
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()

epochs = 10
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': []
}

best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

print(f"\n Начало обучения ({epochs} эпох)...")

start_training = time.time()

for epoch in range(epochs):
    epoch_start = time.time()

    # TRAIN PHASE
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, predicted = torch.max(out, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # VAL PHASE
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)

            val_running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(out, 1)
            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total

    # Обновляем learning rate
    scheduler.step()

    # Сохраняем историю
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    # Сохраняем лучшую модель
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        is_best = "BEST"
    else:
        is_best = ""

    epoch_time = time.time() - epoch_start
    lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch + 1:2d}/{epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {lr:.5f} | {epoch_time:.1f}s {is_best}")

total_time = time.time() - start_training
print("-" * 70)
print(f"Обучение завершено за {total_time:.1f} сек")
print(f"Лучшая Val Accuracy: {best_val_acc:.4f}")

# Восстанавливаем лучшие веса
model.load_state_dict(best_model_wts)

# 4. Сохранение модели
model_path = 'results/best_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes,
    'class_names': train_dataset.get_class_names(),
    'val_accuracy': best_val_acc,
    'epochs_trained': epochs
}, model_path)
print(f"Лучшая модель сохранена: {model_path}")

# 5. Визуализация
print("\n Построение графиков обучения...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

epochs_range = range(1, epochs + 1)

# График Loss
ax1.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss', color='royalblue', linewidth=2, markersize=6)
ax1.plot(epochs_range, history['val_loss'], 's-', label='Val Loss', color='crimson', linewidth=2, markersize=6)
ax1.set_title('Динамика Loss', fontsize=13, fontweight='bold')
ax1.set_xlabel('Эпоха', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_xticks(epochs_range)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# График Accuracy
ax2.plot(epochs_range, history['train_acc'], 'o-', label='Train Accuracy', color='royalblue', linewidth=2, markersize=6)
ax2.plot(epochs_range, history['val_acc'], 's-', label='Val Accuracy', color='crimson', linewidth=2, markersize=6)
ax2.set_title('Динамика Accuracy', fontsize=13, fontweight='bold')
ax2.set_xlabel('Эпоха', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_xticks(epochs_range)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Добавляем значения на последнюю точку
ax2.annotate(f"{history['train_acc'][-1]:.3f}",
             (epochs, history['train_acc'][-1]),
             textcoords="offset points", xytext=(10, 0), ha='left', fontsize=9, color='royalblue')
ax2.annotate(f"{history['val_acc'][-1]:.3f}",
             (epochs, history['val_acc'][-1]),
             textcoords="offset points", xytext=(10, 0), ha='left', fontsize=9, color='crimson')

plt.tight_layout()
plot_path = 'results/training_history.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close('all')
print(f"Графики сохранены: {plot_path}")

# 6. Сохранение отчёта
report_path = 'results/training_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 50 + "\n")
    f.write("       ОТЧЁТ ОБ ОБУЧЕНИИ МОДЕЛИ\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Модель: ResNet18 (ImageNet pretrained)\n")
    f.write(f"Устройство: {device}\n")
    f.write(f"Количество эпох: {epochs}\n")
    f.write(f"Batch size: 32\n")
    f.write(f"Optimizer: Adam (lr=1e-3, StepLR step=3, gamma=0.1)\n\n")
    f.write(f"Train изображений: {len(train_dataset)}\n")
    f.write(f"Val изображений: {len(val_dataset)}\n")
    f.write(f"Количество классов: {num_classes}\n")
    f.write(f"Классы: {', '.join(train_dataset.get_class_names())}\n\n")

    f.write("История обучения \n")
    f.write(f"{'Эпоха':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}\n")
    f.write("-" * 56 + "\n")
    for i in range(epochs):
        f.write(f"{i + 1:<8} {history['train_loss'][i]:<12.4f} {history['train_acc'][i]:<12.4f} "
                f"{history['val_loss'][i]:<12.4f} {history['val_acc'][i]:<12.4f}\n")

    f.write(f"\n Лучшая Val Accuracy: {best_val_acc:.4f}\n")
    f.write(f"Общее время обучения: {total_time:.1f} сек\n")

print(f"Отчёт сохранён: {report_path}")
print("\n Задание 6 выполнено полностью!")