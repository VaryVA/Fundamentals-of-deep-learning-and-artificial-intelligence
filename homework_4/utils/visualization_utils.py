import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pandas as pd


def plot_training_history(history, title="Training History", save_path=None, show=False):
    '''Визуализация истории обучения'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")

    plt.close() 
    if show:
        plt.show()


def plot_confusion_matrix(model, data_loader, device, classes, title="Confusion Matrix", save_path=None, show=False):
    '''Построение confusion matrix'''
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_targets.extend(target.numpy())

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix сохранена: {save_path}")

    plt.close() 
    if show:
        plt.show()


def plot_gradient_flow(grad_flow_history, title="Gradient Flow", save_path=None, show=False):
    '''Визуализация потока градиентов по слоям'''
    if not grad_flow_history:
        return
    last_epoch_grads = grad_flow_history[-1]
    names = list(last_epoch_grads.keys())
    values = list(last_epoch_grads.values())

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(names)), values, alpha=0.7, color='b')
    plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
    plt.ylabel('Mean Absolute Gradient')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gradient flow сохранен: {save_path}")

    plt.close() 
    if show:
        plt.show()


def plot_feature_maps(model, data_loader, device, layer_name="conv1", save_path=None, show=False):
    '''Визуализация feature maps указанного слоя'''
    model.eval()
    data, _ = next(iter(data_loader))
    data = data[0:1].to(device)

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    for name, module in model.named_modules():
        if layer_name in name:
            module.register_forward_hook(get_activation(name))
            break

    model(data)

    if activations:
        feats = list(activations.values())[0]
        n_feats = min(feats.shape[1], 16)
        cols = 8
        rows = (n_feats + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))
        axes = axes.flatten() if n_feats > 1 else [axes]
        for i, ax in enumerate(axes):
            if i < n_feats:
                ax.imshow(feats[0, i].cpu().numpy(), cmap='viridis')
                ax.axis('off')
            else:
                ax.axis('off')
        plt.suptitle(f"Feature Maps: {layer_name}")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature maps сохранены: {save_path}")

        plt.close() 
        if show:
            plt.show()


def compare_models(fc_history, cnn_history, title="FC vs CNN", save_path=None, show=False):
    '''Сравнение FC и CNN на одном графике'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(fc_history['test_accs'], label='FC Network', marker='o')
    ax1.plot(cnn_history['test_accs'], label='CNN', marker='s')
    ax1.set_title(f'{title} - Test Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(fc_history['test_losses'], label='FC Network', marker='o')
    ax2.plot(cnn_history['test_losses'], label='CNN', marker='s')
    ax2.set_title(f'{title} - Test Loss')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Сравнение сохранено: {save_path}")

    plt.close() 
    if show:
        plt.show()



def save_table_as_image(data_dict, title, save_path):
    '''Сохраняет таблицу сравнения моделей как изображение'''
    df = pd.DataFrame(data_dict).T

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(12, len(data_dict) * 0.5 + 2))
    ax.axis('off')

    # Создаем таблицу
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center',
                     colColours=['#f0f0f0'] * len(df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Стилизация заголовков
    for i in range(len(df.columns)):
        table[0, i].set_facecolor('#4CAF50')
        table[0, i].set_text_props(color='white', fontweight='bold')

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    # Сохраняем
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Таблица сохранена: {save_path}")

    plt.close()