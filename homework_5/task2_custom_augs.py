import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
from datasets import CustomImageDataset
from utils import show_single_augmentation
from extra_augs import AddGaussianNoise, CutOut, Solarize

os.makedirs('results', exist_ok=True)


# Реализация кастомных аугментаций
class RandomGaussianBlur:
    '''Случайное гауссово размытие.'''

    def __init__(self, p=0.5, radius_range=(1, 3)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() > self.p: return img
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomSaltAndPepper:
    '''Шум соль и перец.'''

    def __init__(self, p=0.5, amount=0.05):
        self.p = p
        self.amount = amount

    def __call__(self, img):
        if random.random() > self.p: return img
        img_np = np.array(img)
        h, w, c = img_np.shape
        num_salt = np.ceil(self.amount * h * w * 0.5).astype(int)
        num_pepper = np.ceil(self.amount * h * w * 0.5).astype(int)

        # Salt
        coords = [np.random.randint(0, i - 1, num_salt) for i in img_np.shape[:2]]
        img_np[coords[0], coords[1], :] = 255
        # Pepper
        coords = [np.random.randint(0, i - 1, num_pepper) for i in img_np.shape[:2]]
        img_np[coords[0], coords[1], :] = 0

        return Image.fromarray(img_np)


class RandomBrightnessContrast:
    '''Случайное изменение яркости и контраста через PIL.'''

    def __init__(self, p=0.5, brightness=(0.7, 1.3), contrast=(0.7, 1.3)):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        if random.random() > self.p: return img
        b_factor = random.uniform(*self.brightness)
        c_factor = random.uniform(*self.contrast)
        img = ImageEnhance.Brightness(img).enhance(b_factor)
        img = ImageEnhance.Contrast(img).enhance(c_factor)
        return img


# Применение и сравнение
dataset = CustomImageDataset('data/train', transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Берем изображения из 3 разных классов
test_images = []
for i, cls in enumerate(class_names[:3]):
    idx = dataset.labels.index(dataset.class_to_idx[cls])
    img_pil, _ = dataset[idx]
    test_images.append((img_pil, cls))

custom_augs = {
    "GaussianBlur": RandomGaussianBlur(p=1.0),
    "SaltAndPepper": RandomSaltAndPepper(p=1.0),
    "BrightnessContrast": RandomBrightnessContrast(p=1.0)
}

print("Применение кастомных аугментаций...")
for img_pil, class_name in test_images:
    print(f"\n Класс: {class_name} ")
    orig_tensor = transforms.ToTensor()(img_pil)

    for name, aug in custom_augs.items():
        # Аугментации работают с PIL Image
        aug_img_pil = aug(img_pil)
        aug_img_tensor = transforms.ToTensor()(aug_img_pil)

        show_single_augmentation(orig_tensor, aug_img_tensor, f"Custom: {name}")

        plt.suptitle(f'Класс: {class_name}', fontsize=16, fontweight='bold', y=1.05)
        plt.savefig(f'results/{class_name}_custom_{name}.png', dpi=150, bbox_inches='tight')
        print(f"Сохранено: results/{class_name}_custom_{name}.png")
        plt.close('all')

# Сравнение с готовыми аугментациями из extra_augs
print("\nСравнение с готовыми аугментациями из extra_augs...")
extra_augs_dict = {
    "AddGaussianNoise": transforms.Compose([AddGaussianNoise(0., 0.1)]),
    "CutOut": transforms.Compose([CutOut(p=1.0, size=(32, 32))]),
    "Solarize": transforms.Compose([Solarize(threshold=128)])
}

for img_pil, class_name in test_images:
    print(f"\n Класс: {class_name} ")
    orig_tensor = transforms.ToTensor()(img_pil)

    for name, aug in extra_augs_dict.items():
        aug_img_tensor = aug(orig_tensor)
        show_single_augmentation(orig_tensor, aug_img_tensor, f"Extra: {name}")
        # Добавляем название класса в заголовок
        plt.suptitle(f'Класс: {class_name}', fontsize=16, fontweight='bold', y=1.05)
        plt.savefig(f'results/{class_name}_extra_{name}.png', dpi=150, bbox_inches='tight')
        print(f"Сохранено: results/{class_name}_extra_{name}.png")
        plt.close('all')

print("\n Все кастомные аугментации сохранены в results/")