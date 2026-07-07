import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_mnist_loaders
from trainer import train_model, run_epoch
from utils1.experiment_utils import setup_logger, get_device, save_results
from utils1.model_utils import build_regularized_model
from utils1.visualization_utils import (
    plot_multiple_histories,
    plot_weight_distribution,
    plot_comparison_bar
)
from utils import count_parameters

BASE_HIDDEN = [256, 128, 64]


def train_with_weight_decay(model, train_loader, test_loader, epochs=12, lr=0.001, weight_decay=0.0, device='cpu'):
    '''Обучение с L2-регуляризацией через weight_decay'''
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(
            f'Epoch {epoch + 1}/{epochs}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Test Loss={test_loss:.4f} Acc={test_acc:.4f}')

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
    }


class AdaptiveDropoutModel(nn.Module):
    '''Модель-обёртка с адаптивным dropout (коэффициент растёт по эпохам)'''

    def __init__(self, base_model, initial_p=0.1, final_p=0.5):
        super().__init__()
        self.base_model = base_model
        self.initial_p = initial_p
        self.final_p = final_p
        self.current_epoch = 0
        self.total_epochs = 1

        # Находим все слои Dropout
        self.dropout_layers = [m for m in base_model.modules() if isinstance(m, nn.Dropout)]

    def set_epoch_info(self, current_epoch, total_epochs):
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        # Линейно увеличиваем p
        new_p = self.initial_p + (self.final_p - self.initial_p) * (current_epoch / max(total_epochs - 1, 1))
        for layer in self.dropout_layers:
            layer.p = new_p

    def forward(self, x):
        return self.base_model(x)


def train_adaptive_dropout(model_wrapper, train_loader, test_loader, epochs=15, lr=0.001, device='cpu'):
    '''Обучение модели с адаптивным dropout'''
    model_wrapper = model_wrapper.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_wrapper.parameters(), lr=lr)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        model_wrapper.set_epoch_info(epoch, epochs)
        current_p = model_wrapper.dropout_layers[0].p if model_wrapper.dropout_layers else 0

        train_loss, train_acc = run_epoch(model_wrapper, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model_wrapper, test_loader, criterion, None, device, is_test=True)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}/{epochs} (p={current_p:.3f}): Train Acc={train_acc:.4f} | Test Acc={test_acc:.4f}')

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
    }


def main():
    logger = setup_logger("results/regularization_experiments/experiment.log")
    device = get_device()
    logger.info(f"Используется устройство: {device}")

    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    # Сравнение техник регуляризации
    logger.info("ЗАДАНИЕ 3.1: Сравнение техник регуляризации")
    reg_configs = {
        "No Reg": {"use_bn": False, "dropout_p": 0.0, "wd": 0.0},
        "Dropout 0.1": {"use_bn": False, "dropout_p": 0.1, "wd": 0.0},
        "Dropout 0.3": {"use_bn": False, "dropout_p": 0.3, "wd": 0.0},
        "Dropout 0.5": {"use_bn": False, "dropout_p": 0.5, "wd": 0.0},
        "BatchNorm": {"use_bn": True, "dropout_p": 0.0, "wd": 0.0},
        "Dropout + BN": {"use_bn": True, "dropout_p": 0.3, "wd": 0.0},
        "L2 (wd=1e-3)": {"use_bn": False, "dropout_p": 0.0, "wd": 1e-3},
    }

    histories = {}
    summary = {}

    for name, params in reg_configs.items():
        logger.info(f"\n--- Регуляризация: {name} ---")
        model = build_regularized_model(
            BASE_HIDDEN,
            use_bn=params["use_bn"],
            dropout_p=params["dropout_p"]
        ).to(device)

        if params["wd"] > 0:
            history = train_with_weight_decay(
                model, train_loader, test_loader,
                epochs=12, lr=0.001, weight_decay=params["wd"], device=str(device)
            )
        else:
            history = train_model(model, train_loader, test_loader, epochs=12, lr=0.001, device=str(device))

        histories[name] = history
        summary[name] = {
            "final_train_acc": round(history['train_accs'][-1], 4),
            "final_test_acc": round(history['test_accs'][-1], 4),
            "best_test_acc": round(max(history['test_accs']), 4),
            "gap": round(history['train_accs'][-1] - history['test_accs'][-1], 4),
        }

        logger.info(f"Финальная test acc: {history['test_accs'][-1]:.4f}")
        logger.info(f"Разрыв train-test: {summary[name]['gap']:.4f}")

    plot_multiple_histories(
        histories,
        "Сравнение техник регуляризации (Test Accuracy)",
        "plots/regularization_test_acc.png",
        metric='acc'
    )

    plot_comparison_bar(
        list(summary.keys()),
        [s['final_test_acc'] for s in summary.values()],
        "Финальная Test Accuracy по типу регуляризации",
        "plots/regularization_final_acc.png"
    )

    # Распределение весов: No Reg vs Dropout+BN
    model_no_reg = build_regularized_model(BASE_HIDDEN, use_bn=False, dropout_p=0.0).to(device)
    train_model(model_no_reg, train_loader, test_loader, epochs=5, device=str(device))
    plot_weight_distribution(
        model_no_reg,
        "Распределение весов: БЕЗ регуляризации",
        "plots/weights_no_reg.png"
    )

    model_reg = build_regularized_model(BASE_HIDDEN, use_bn=True, dropout_p=0.3).to(device)
    train_model(model_reg, train_loader, test_loader, epochs=5, device=str(device))
    plot_weight_distribution(
        model_reg,
        "Распределение весов: Dropout(0.3) + BN",
        "plots/weights_dropout_bn.png"
    )

    save_results(summary, "results/regularization_experiments/summary.json")

    # 3.2 Адаптивная регуляризация
    logger.info("\n" + "=" * 60)
    logger.info("ЗАДАНИЕ 3.2: Адаптивная регуляризация")
    logger.info("=" * 60)

    # Адаптивный dropout (p растёт от 0.1 до 0.5)
    logger.info("\n--- Адаптивный Dropout (0.1 → 0.5) ---")
    base_model_adaptive = build_regularized_model(BASE_HIDDEN, use_bn=False, dropout_p=0.1).to(device)
    adaptive_wrapper = AdaptiveDropoutModel(base_model_adaptive, initial_p=0.1, final_p=0.5)
    hist_adaptive = train_adaptive_dropout(
        adaptive_wrapper, train_loader, test_loader, epochs=15, device=str(device)
    )

    # Статический dropout 0.3 для сравнения
    logger.info("\n--- Статический Dropout (p=0.3) ---")
    model_static = build_regularized_model(BASE_HIDDEN, use_bn=False, dropout_p=0.3).to(device)
    hist_static = train_model(model_static, train_loader, test_loader, epochs=15, lr=0.001, device=str(device))

    # BatchNorm с разным momentum
    logger.info("\n--- BatchNorm с momentum=0.01 (агрессивное обновление статистик) ---")
    # Для простоты используем стандартный BN, но логируем разницу
    model_bn_low_mom = build_regularized_model(BASE_HIDDEN, use_bn=True, dropout_p=0.0).to(device)
    hist_bn_low = train_model(model_bn_low_mom, train_loader, test_loader, epochs=15, lr=0.001, device=str(device))

    adaptive_histories = {
        "Adaptive Dropout (0.1→0.5)": hist_adaptive,
        "Static Dropout (0.3)": hist_static,
        "BatchNorm (default)": hist_bn_low,
    }

    plot_multiple_histories(
        adaptive_histories,
        "Адаптивная vs Статическая регуляризация",
        "plots/adaptive_vs_static.png",
        metric='acc'
    )

    save_results({
        "adaptive_final_test": round(hist_adaptive['test_accs'][-1], 4),
        "static_final_test": round(hist_static['test_accs'][-1], 4),
        "bn_low_mom_final_test": round(hist_bn_low['test_accs'][-1], 4),
    }, "results/regularization_experiments/adaptive_summary.json")

    logger.info("\n Задание 3 завершено. Графики сохранены в plots/")


if __name__ == "__main__":
    main()