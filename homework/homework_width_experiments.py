import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import get_mnist_loaders
from trainer import train_model
from utils1.experiment_utils import setup_logger, get_device, save_results
from utils1.model_utils import build_width_model
from utils1.visualization_utils import plot_heatmap, plot_multiple_histories, plot_comparison_bar
from utils import count_parameters
import numpy as np


def main():
    logger = setup_logger("results/width_experiments/experiment.log")
    device = get_device()
    logger.info(f"Используется устройство: {device}")

    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    # Сравнение моделей разной ширины
    logger.info("=" * 60)
    logger.info("ЗАДАНИЕ 2.1: Сравнение моделей разной ширины (глубина = 3)")
    logger.info("=" * 60)

    width_configs = {
        "Narrow [64, 32, 16]": [64, 32, 16],
        "Medium [256, 128, 64]": [256, 128, 64],
        "Wide [1024, 512, 256]": [1024, 512, 256],
        "Very Wide [2048, 1024, 512]": [2048, 1024, 512],
    }

    histories = {}
    summary = {}

    for name, hidden_sizes in width_configs.items():
        logger.info(f"\n--- Обучение модели: {name} ---")
        # Добавляем BatchNorm, чтобы широкие сети стабильно сходились
        model = build_width_model(hidden_sizes, use_bn=True, dropout_p=0.0).to(device)
        params = count_parameters(model)
        logger.info(f"Количество параметров: {params:,}")

        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device=str(device))
        elapsed = time.time() - start_time

        histories[name] = history
        summary[name] = {
            "params": params,
            "time_sec": round(elapsed, 2),
            "final_train_acc": round(history['train_accs'][-1], 4),
            "final_test_acc": round(history['test_accs'][-1], 4),
        }

        logger.info(f"Время обучения: {elapsed:.2f} сек")
        logger.info(f"Финальная test acc: {history['test_accs'][-1]:.4f}")

    plot_multiple_histories(
        histories,
        "Влияние ширины сети на Accuracy (MNIST)",
        "plots/width_accuracy.png",
        metric='acc'
    )

    plot_comparison_bar(
        list(summary.keys()),
        [s['final_test_acc'] for s in summary.values()],
        "Финальная Test Accuracy по ширине сети",
        "plots/width_final_acc.png"
    )

    save_results(summary, "results/width_experiments/summary.json")

    # ===== 2.2 Оптимизация архитектуры (Grid Search) =====
    logger.info("\n" + "=" * 60)
    logger.info("ЗАДАНИЕ 2.2: Grid Search — ширина × схема изменения")
    logger.info("=" * 60)

    base_widths = [64, 256, 1024]
    schemes = {
        "Сужение (x0.5)": [1.0, 0.5, 0.25],
        "Постоянная (x1.0)": [1.0, 1.0, 1.0],
        "Расширение (x2.0)": [1.0, 2.0, 4.0],
    }

    results_matrix = np.zeros((len(base_widths), len(schemes)))
    grid_summary = {}

    for i, base_w in enumerate(base_widths):
        for j, (scheme_name, multipliers) in enumerate(schemes.items()):
            hidden_sizes = [max(int(base_w * m), 16) for m in multipliers]

            model = build_width_model(hidden_sizes, use_bn=True).to(device)
            history = train_model(model, train_loader, test_loader, epochs=8, lr=0.001, device=str(device))

            final_acc = history['test_accs'][-1]
            results_matrix[i, j] = final_acc

            key = f"base_{base_w}_{scheme_name}"
            grid_summary[key] = {
                "hidden_sizes": hidden_sizes,
                "final_test_acc": round(final_acc, 4),
            }

            logger.info(f"Base={base_w}, Scheme={scheme_name} | Sizes={hidden_sizes} | Acc={final_acc:.4f}")

    plot_heatmap(
        results_matrix,
        list(schemes.keys()),
        [f"Base {w}" for w in base_widths],
        "Heatmap: Test Accuracy по ширине и схеме",
        "plots/width_heatmap.png"
    )

    save_results(grid_summary, "results/width_experiments/grid_search.json")

    logger.info("\n Задание 2 завершено. Графики сохранены в plots/")


if __name__ == "__main__":
    main()