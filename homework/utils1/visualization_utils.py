import matplotlib.pyplot as plt
import numpy as np
import os


def plot_multiple_histories(histories_dict, title, save_path, metric='acc'):
    '''
    Строиv графики для нескольких моделей
    histories_dict: {имя_модели: history_dict}
    metric: 'acc' или 'loss'
    '''
    fig, ax = plt.subplots(figsize=(10, 6))

    suffix = 'accs' if metric == 'acc' else 'losses'
    ylabel = 'Accuracy' if metric == 'acc' else 'Loss'

    for name, history in histories_dict.items():
        ax.plot(history[f'train_{suffix}'], label=f'Train {name}', linestyle='--')
        ax.plot(history[f'test_{suffix}'], label=f'Test {name}', linestyle='-')

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_heatmap(data, x_labels, y_labels, title, save_path):
    '''Строим heatmap для grid search'''
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap='viridis', aspect='auto')

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels)

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                           ha='center', va='center', color='white', fontsize=10)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Test Accuracy')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_weight_distribution(model, title, save_path):
    '''Визуализируем распределение весов и bias модели'''
    weights = []
    biases = []

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            weights.extend(param.detach().cpu().numpy().flatten())
        elif 'bias' in name:
            biases.extend(param.detach().cpu().numpy().flatten())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if weights:
        axes[0].hist(weights, bins=80, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title(f'Weights Distribution\n(mean={np.mean(weights):.4f}, std={np.std(weights):.4f})')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Count')

    if biases:
        axes[1].hist(biases, bins=80, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1].set_title(f'Biases Distribution\n(mean={np.mean(biases):.4f}, std={np.std(biases):.4f})')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Count')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_bar(names, values, title, save_path, ylabel='Test Accuracy'):
    '''Столбчатая диаграмма для сравнения финальных метрик'''
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(names)), values, color='steelblue', edgecolor='black')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x(), bar.get_height() + 0.005, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()