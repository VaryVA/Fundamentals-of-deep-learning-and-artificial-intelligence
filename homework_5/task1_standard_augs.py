import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from datasets import CustomImageDataset
from utils import show_single_augmentation, show_multiple_augmentations

# Создаем папку для результатов
os.makedirs('results', exist_ok=True)

# 1. Загрузка датасета
dataset = CustomImageDataset('data/train', transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Берем 5 изображений из РАЗНЫХ классов
selected_imgs = []
selected_titles = []
for i, cls in enumerate(class_names[:5]):
    idx = dataset.labels.index(dataset.class_to_idx[cls])
    img_pil, _ = dataset[idx]
    img_tensor = transforms.ToTensor()(img_pil)
    selected_imgs.append(img_tensor)
    selected_titles.append(cls)

# 2. Стандартные аугментации
augs = {
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=1.0),
    "RandomCrop": transforms.RandomCrop(200, padding=20),
    "ColorJitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    "RandomRotation": transforms.RandomRotation(degrees=30),
    "RandomGrayscale": transforms.RandomGrayscale(p=1.0)
}

# 3. Визуализация для каждого из 5 изображений
for img_tensor, title in zip(selected_imgs, selected_titles):
    print(f"\n Класс: {title} ")

    # Отдельные аугментации
    aug_imgs = []
    aug_titles = []
    for name, aug in augs.items():
        img_pil = transforms.ToPILImage()(img_tensor)
        aug_img_pil = aug(img_pil)
        aug_img_tensor = transforms.ToTensor()(aug_img_pil)

        aug_imgs.append(aug_img_tensor)
        aug_titles.append(name)

    # Показываем и сохраняем стандартные аугментации
    show_multiple_augmentations(img_tensor, aug_imgs, aug_titles)
    plt.savefig(f'results/{title}_standard_augs.png', dpi=150, bbox_inches='tight')
    print(f"Сохранено: results/{title}_standard_augs.png")

    # Комбинированная аугментация
    combined = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(200, padding=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor()
    ])
    img_pil = transforms.ToPILImage()(img_tensor)
    combined_img = combined(img_pil)

    # Показываем и сохраняем комбинированную аугментацию
    show_single_augmentation(img_tensor, combined_img, "Combined Augs")
    plt.savefig(f'results/{title}_combined.png', dpi=150, bbox_inches='tight')
    print(f"Сохранено: results/{title}_combined.png")

    # Закрываем figure, чтобы не было утечки памяти
    plt.close('all')

print("\n Все изображения сохранены в папку results/")