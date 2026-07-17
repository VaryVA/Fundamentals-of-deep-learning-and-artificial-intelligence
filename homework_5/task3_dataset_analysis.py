import os
import warnings


import matplotlib

matplotlib.use('Agg')  # Отключаем показ окон
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

# Создаем папку для результатов
os.makedirs('results', exist_ok=True)

root_dir = 'data/train'


classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

class_counts = defaultdict(int)
widths = []
heights = []
areas = []

print("Анализ датасета... Это может занять некоторое время.")

for cls in classes:
    cls_dir = os.path.join(root_dir, cls)
    for img_name in os.listdir(cls_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            class_counts[cls] += 1
            img_path = os.path.join(cls_dir, img_name)

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    areas.append(w * h)
            except Exception as e:
                print(f"⚠ Не удалось открыть изображение {img_path}: {e}")

# Статистика
total_images = sum(class_counts.values())

print(f"\n Статистика датасета:")
print(f"Всего классов: {len(classes)}")
print(f"Всего изображений: {total_images}")

if areas:
    min_area = min(areas)
    max_area = max(areas)
    avg_area = sum(areas) / len(areas)

    print(f"Минимальный размер (площадь): {min_area} пикселей")
    print(f"Максимальный размер (площадь): {max_area} пикселей")
    print(f"Средний размер (площадь): {avg_area:.0f} пикселей")
else:
    print("Изображения не найдены!")
    exit(1)

# Сохраняем статистику
stats_file = 'results/dataset_statistics.txt'
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("=" * 40 + "\n")
    f.write("       СТАТИСТИКА ДАТАСЕТА\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Корневая папка: {root_dir}\n")
    f.write(f"Всего классов: {len(classes)}\n")
    f.write(f"Всего изображений: {total_images}\n\n")

    f.write("Размеры изображений (площадь в пикселях) \n")
    f.write(f"Минимальный размер: {min_area} px\n")
    f.write(f"Максимальный размер: {max_area} px\n")
    f.write(f"Средний размер:     {avg_area:.0f} px\n\n")

    f.write(" Распределение по классам \n")
    # Сортируем по количеству изображений
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_classes:
        percentage = (count / total_images) * 100
        f.write(f"• {cls}: {count} изобр. ({percentage:.1f}%)\n")

print(f"Статистика успешно сохранена в: {stats_file}")

# --- Визуализация ---
print("\n Построение графиков...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Гистограмма по классам
axes[0].bar(class_counts.keys(), class_counts.values(), color='skyblue', edgecolor='black')
axes[0].set_title('Распределение изображений по классам', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Классы (герои)', fontsize=10)
axes[0].set_ylabel('Количество изображений', fontsize=10)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
max_count = max(class_counts.values()) if class_counts.values() else 1
for i, v in enumerate(class_counts.values()):
    axes[0].text(i, v + (max_count * 0.01), str(v), ha='center', fontsize=9)

# 2. Распределение размеров (Scatter plot Width vs Height)
axes[1].scatter(widths, heights, alpha=0.4, c='green', edgecolors='none')
axes[1].set_title('Распределение исходных размеров изображений', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Ширина (px)', fontsize=10)
axes[1].set_ylabel('Высота (px)', fontsize=10)
axes[1].grid(True, alpha=0.3)

# Сохранение и очистка
plt.tight_layout()
output_path = 'results/dataset_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close('all')

print(f"Графики успешно сохранены в: {output_path}")