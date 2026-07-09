import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

def count_parameters(model):
    '''Подсчитывает количество обучаемых параметров модели'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models_table(results_dict, title="СРАВНЕНИЕ МОДЕЛЕЙ", save_path=None):
    '''Выводит таблицу сравнения моделей и сохраняет как картинку'''
    df = pd.DataFrame(results_dict).T

    print(title)
    print(df.to_string())

    if save_path:
        fig, ax = plt.subplots(figsize=(12, len(results_dict) * 0.5 + 2))
        ax.axis('off')

        table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         rowLabels=df.index,
                         cellLoc='center',
                         loc='center',
                         colColours=['#f0f0f0'] * len(df.columns))

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        for i in range(len(df.columns)):
            table[0, i].set_facecolor('#4CAF50')
            table[0, i].set_text_props(color='white', fontweight='bold')

        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Таблица сохранена: {save_path}")

        plt.close()

    return df


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model