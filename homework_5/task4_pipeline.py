import os
import warnings


import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from datasets import CustomImageDataset
from extra_augs import AddGaussianNoise, CutOut, Solarize, AutoContrast

# Создаем папку для результатов
os.makedirs('results', exist_ok=True)


# Реализация класса AugmentationPipeline 
class AugmentationPipeline:
    '''Класс для управления пайплайном аугментаций.'''

    def __init__(self):
        self.augmentations = {}

    def add_augmentation(self, name, aug):
        '''Добавляет аугментацию в пайплайн.'''
        self.augmentations[name] = aug

    def remove_augmentation(self, name):
        '''Удаляет аугментацию из пайплайна.'''
        if name in self.augmentations:
            del self.augmentations[name]
            print(f"Аугментация '{name}' удалена")
        else:
            print(f"Аугментация '{name}' не найдена")

    def get_augmentations(self):
        '''Возвращает список названий аугментаций.'''
        return list(self.augmentations.keys())

    def apply(self, image):
        '''
        Применяет все аугментации к изображению.
        image: PIL Image
        возвращает: тензор
        '''
        # image - PIL Image
        for name, aug in self.augmentations.items():
            # Проверяем, требует ли аугментация тензор (как в extra_augs)
            if isinstance(aug, (AddGaussianNoise, CutOut, Solarize, AutoContrast)):
                img_tensor = transforms.ToTensor()(image)
                img_tensor = aug(img_tensor)
                image = transforms.ToPILImage()(img_tensor)
            else:
                # Стандартные аугментации torchvision работают с PIL
                image = aug(image)

        # Возвращаем финальный тензор
        return transforms.ToTensor()(image)


# Создание конфигураций
print("Создание конфигураций пайплайнов...")

# 1. LIGHT - минимальные аугментации
light_pipeline = AugmentationPipeline()
light_pipeline.add_augmentation("HFlip", transforms.RandomHorizontalFlip(p=0.5))
light_pipeline.add_augmentation("ColorJitter", transforms.ColorJitter(brightness=0.1, contrast=0.1))

# 2. MEDIUM - средние аугментации
medium_pipeline = AugmentationPipeline()
medium_pipeline.add_augmentation("HFlip", transforms.RandomHorizontalFlip(p=0.5))
medium_pipeline.add_augmentation("ColorJitter", transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
medium_pipeline.add_augmentation("Rotation", transforms.RandomRotation(degrees=15))
medium_pipeline.add_augmentation("Noise", AddGaussianNoise(0., 0.05))

# 3. HEAVY - агрессивные аугментации
heavy_pipeline = AugmentationPipeline()
heavy_pipeline.add_augmentation("HFlip", transforms.RandomHorizontalFlip(p=0.5))
heavy_pipeline.add_augmentation("Rotation", transforms.RandomRotation(degrees=30))
heavy_pipeline.add_augmentation("ColorJitter",
                                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1))
heavy_pipeline.add_augmentation("Noise", AddGaussianNoise(0., 0.1))
heavy_pipeline.add_augmentation("CutOut", CutOut(p=0.5, size=(32, 32)))
heavy_pipeline.add_augmentation("Solarize", Solarize(threshold=128))

# Выводим информацию о конфигурациях
print(f"\nLIGHT pipeline: {light_pipeline.get_augmentations()}")
print(f"MEDIUM pipeline: {medium_pipeline.get_augmentations()}")
print(f"HEAVY pipeline: {heavy_pipeline.get_augmentations()}")

# Применение к изображениям
print("\n Применение пайплайнов к изображениям...")

# Загружаем датасет
dataset = CustomImageDataset('data/train', transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Берем изображения из 3 разных классов
test_images = []
for i, cls in enumerate(class_names[:3]):
    idx = dataset.labels.index(dataset.class_to_idx[cls])
    img_pil, _ = dataset[idx]
    test_images.append((img_pil, cls))

pipelines = {
    "light": light_pipeline,
    "medium": medium_pipeline,
    "heavy": heavy_pipeline
}

# Применяем каждый pipeline к каждому изображению
for img_pil, class_name in test_images:
    print(f"\n Класс: {class_name} ")

    for pipe_name, pipe in pipelines.items():
        # Применяем пайплайн
        result_tensor = pipe.apply(img_pil)

        # Сохраняем результат
        img_to_save = transforms.ToPILImage()(result_tensor)
        filename = f'results/{class_name}_pipeline_{pipe_name}.jpg'
        img_to_save.save(filename, quality=95)

        print(f" Pipeline '{pipe_name}' применен и сохранен: {filename}")

print("\n Все результаты pipeline сохранены в папке results/")