import os
import time
import warnings
import tracemalloc

import matplotlib

matplotlib.use('Agg')  # Отключаем показ окон
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from datasets import CustomImageDataset

# Создаем папку для результатов
os.makedirs('results', exist_ok=True)

# Размеры для эксперимента
sizes = [64, 128, 224, 512]
times = []
memories = []

print("Запуск эксперимента с размерами изображений...")
print("-" * 50)

for size in sizes:
    print(f"Тестирование размера: {size}x{size}")

    # Пайплайн: ресайз + простая аугментация + тензор
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor()
    ])

    # Создаем датасет с нужным target_size
    dataset = CustomImageDataset('data/train', transform=transform, target_size=(size, size))

    # Берем ровно 100 изображений для чистоты эксперимента
    subset = Subset(dataset, range(min(100, len(dataset))))

    # num_workers=0 важен для корректного замера памяти в главном процессе
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)

    # Запускаем замер памяти
    tracemalloc.start()
    start_time = time.time()

    # Эмуляция загрузки и применения аугментаций
    for batch in loader:
        images, labels = batch
        # images уже прошли через аугментации благодаря transform в Dataset

    end_time = time.time()

    # Получаем пиковое потребление памяти и останавливаем замер
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Переводим байты в мегабайты
    mem_mb = peak_mem / (1024 * 1024)
    elapsed_time = end_time - start_time

    times.append(elapsed_time)
    memories.append(mem_mb)

    print(f"Время: {elapsed_time:.4f} сек | Память: {mem_mb:.2f} МБ")

print("-" * 50)
print("Построение графиков зависимости...")

# Создаем фигуру с двумя графиками
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# График 1: Зависимость времени от размера
axes[0].plot(sizes, times, marker='o', color='royalblue', linewidth=2, markersize=8)
axes[0].set_title('Зависимость времени обработки от размера', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Размер изображения (px)', fontsize=10)
axes[0].set_ylabel('Время обработки 100 изображений (сек)', fontsize=10)
axes[0].set_xticks(sizes)
axes[0].grid(True, alpha=0.3)

# Добавляем значения на точки
for i, txt in enumerate(times):
    axes[0].annotate(f'{txt:.2f}s', (sizes[i], times[i]), textcoords="offset points", xytext=(0, 10), ha='center')

# График 2: Зависимость памяти от размера
axes[1].plot(sizes, memories, marker='s', color='crimson', linewidth=2, markersize=8)
axes[1].set_title('Зависимость потребления памяти от размера', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Размер изображения (px)', fontsize=10)
axes[1].set_ylabel('Пиковая память (МБ)', fontsize=10)
axes[1].set_xticks(sizes)
axes[1].grid(True, alpha=0.3)

# Добавляем значения на точки
for i, txt in enumerate(memories):
    axes[1].annotate(f'{txt:.1f} МБ', (sizes[i], memories[i]), textcoords="offset points", xytext=(0, 10), ha='center')

# Сохранение и очистка
plt.tight_layout()
output_path = 'results/size_experiment.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close('all')

print(f"Графики успешно сохранены в: {output_path}")