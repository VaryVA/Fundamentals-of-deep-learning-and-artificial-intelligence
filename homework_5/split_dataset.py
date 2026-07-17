import os
import shutil
import random


def split_dataset(src_dir='data/train', dst_dir='data/val', split_ratio=0.2):
    '''Делит датасет из src_dir на train и val'''
    os.makedirs(dst_dir, exist_ok=True)
    random.seed(42)

    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    for cls in classes:
        src_cls_dir = os.path.join(src_dir, cls)
        dst_cls_dir = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        val_images = images[:split_idx]

        for img_name in val_images:
            src_path = os.path.join(src_cls_dir, img_name)
            dst_path = os.path.join(dst_cls_dir, img_name)
            shutil.move(src_path, dst_path)

    print(f"Датасет успешно разделен! Папка {dst_dir} создана.")


if __name__ == "__main__":
    split_dataset()