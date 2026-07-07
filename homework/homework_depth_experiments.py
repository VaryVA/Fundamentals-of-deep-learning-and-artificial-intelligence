import sys
import os
import time

# Добавляем путь к корню проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import get_mnist_loaders
from trainer import train_model
from utils1.experiment_utils import setup_logger, get_device, save_results
from utils1.model_utils import build_depth_model
from utils1.visualization_utils import plot_multiple_histories, plot_comparison_bar
from utils import count_parameters


def main():
    logger = setup_logger("results/depth_experiments/experiment.log")
    device = get_device()
    logger.info(f"Используется устройство: {device}")

    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    # Сравнение моделей разной глубины
    logger.info("=" * 60)
    logger.info("ЗАДАНИЕ 1.1: Сравнение моделей разной глубины")
    logger.info("=" * 60)

    depth_configs = {
        "1 layer (Linear)": 0,
        "2 layers": 1,
        "3 layers": 2,
        "5 layers": 4,
        "7 layers": 6,
    }

    histories = {}
    summary = {}

    for name, num_hidden in depth_configs.items():
        logger.info(f"\n--- Обучение модели: {name} ---")
        model = build_depth_model(num_hidden_layers=num_hidden, base_width=256).to(device)
        params = count_parameters(model)
        logger.info(f"Количество параметров: {params:,}")

        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=15, lr=0.001, device=str(device))
        elapsed = time.time() - start_time

        histories[name] = history
        summary[name] = {
            "params": params,
            "time_sec": round(elapsed, 2),
            "final_train_acc": round(history['train_accs'][-1], 4),
            "final_test_acc": round(history['test_accs'][-1], 4),
            "best_test_acc": round(max(history['test_accs']), 4),
        }

        logger.info(f"Время обучения: {elapsed:.2f} сек")
        logger.info(f"Финальная train acc: {history['train_accs'][-1]:.4f}")
        logger.info(f"Финальная test acc: {history['test_accs'][-1]:.4f}")

    # Визуализация
    plot_multiple_histories(
        histories,
        "Влияние глубины сети на Accuracy (MNIST)",
        "plots/depth_accuracy.png",
        metric='acc'
    )
    plot_multiple_histories(
        histories,
        "Влияние глубины сети на Loss (MNIST)",
        "plots/depth_loss.png",
        metric='loss'
    )

    # Bar chart финальных точностей
    plot_comparison_bar(
        list(summary.keys()),
        [s['final_test_acc'] for s in summary.values()],
        "Финальная Test Accuracy по глубине сети",
        "plots/depth_final_acc.png",
        ylabel='Test Accuracy'
    )

    save_results(summary, "results/depth_experiments/summary.json")

    # Анализ переобучения
    logger.info("\n" + "=" * 60)
    logger.info("ЗАДАНИЕ 1.2: Анализ переобучения (7 слоев)")
    logger.info("=" * 60)

    # Модель без регуляризации
    logger.info("\n--- 7 слоев БЕЗ регуляризации ---")
    model_base = build_depth_model(num_hidden_layers=6, base_width=256, use_bn=False, dropout_p=0.0).to(device)
    hist_base = train_model(model_base, train_loader, test_loader, epochs=20, lr=0.001, device=str(device))

    # Модель с Dropout + BatchNorm
    logger.info("\n--- 7 слоев С Dropout(0.3) + BatchNorm ---")
    model_reg = build_depth_model(num_hidden_layers=6, base_width=256, use_bn=True, dropout_p=0.3).to(device)
    hist_reg = train_model(model_reg, train_loader, test_loader, epochs=20, lr=0.001, device=str(device))

    overfitting_histories = {
        "7 layers (no reg)": hist_base,
        "7 layers (Dropout+BN)": hist_reg,
    }

    plot_multiple_histories(
        overfitting_histories,
        "Борьба с переобучением: 7-слойная сеть",
        "plots/overfitting_analysis.png",
        metric='acc'
    )

    # Анализ разрыва train-test
    gap_base = hist_base['train_accs'][-1] - hist_base['test_accs'][-1]
    gap_reg = hist_reg['train_accs'][-1] - hist_reg['test_accs'][-1]

    logger.info(f"\nРазрыв train-test (без рег.): {gap_base:.4f}")
    logger.info(f"Разрыв train-test (с рег.): {gap_reg:.4f}")

    save_results({
        "no_reg_gap": gap_base,
        "with_reg_gap": gap_reg,
        "no_reg_final_test": hist_base['test_accs'][-1],
        "with_reg_final_test": hist_reg['test_accs'][-1],
    }, "results/depth_experiments/overfitting_analysis.json")

    logger.info("\n Задание 1 завершено. Графики сохранены в plots/")


if __name__ == "__main__":
    main()