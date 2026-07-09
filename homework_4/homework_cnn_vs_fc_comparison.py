import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_mnist_loaders, get_cifar_loaders
from models.fc_models import SimpleFC, DeepFC
from models.cnn_models import SimpleCNN, CNNWithResidual, CIFARCNN, CIFARResNet
from utils.training_utils import train_model, measure_inference_time, setup_logging
from utils.visualization_utils import (
    plot_training_history, plot_confusion_matrix, compare_models
)
from utils.comparison_utils import count_parameters, compare_models_table


os.makedirs('results/mnist_comparison', exist_ok=True)
os.makedirs('results/cifar_comparison', exist_ok=True)
os.makedirs('plots', exist_ok=True)
setup_logging('results/experiment.log')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


print("1.1 Сравнение на MNIST")
train_loader, test_loader = get_mnist_loaders(64)
mnist_results = {}

models_mnist = {
    "Simple FC": SimpleFC().to(device),
    "Simple CNN": SimpleCNN().to(device),
    "CNN + Residual": CNNWithResidual().to(device)
}

histories_mnist = {}
for name, model in models_mnist.items():
    print(f"\nОбучение {name}...")
    params = count_parameters(model)
    inf_time = measure_inference_time(model, (64, 1, 28, 28), device)

    history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))
    histories_mnist[name] = history

    plot_training_history(history, title=f"MNIST - {name}",
                          save_path=f"plots/mnist_{name.replace(' ', '_')}_history.png")

    mnist_results[name] = {
        "Params": params,
        "Inf Time (ms)": f"{inf_time:.2f}",
        "Final Test Acc": f"{history['test_accs'][-1]:.4f}",
        "Final Train Acc": f"{history['train_accs'][-1]:.4f}"
    }

# Сравнение FC vs CNN
compare_models(histories_mnist["Simple FC"], histories_mnist["Simple CNN"],
               title="MNIST: FC vs CNN",
               save_path="plots/mnist_fc_vs_cnn.png")

compare_models_table(mnist_results,
                     title="MNIST COMPARISON",
                     save_path="results/mnist_comparison/mnist_comparison_table.png")

# 1.2 Сравнение на CIFAR-10
print("1.2 Сравнение на CIFAR-10")
train_loader, test_loader = get_cifar_loaders(64)
cifar_results = {}

models_cifar = {
    "Deep FC": DeepFC().to(device),
    "CIFAR CNN": CIFARCNN().to(device),
    "CIFAR ResNet": CIFARResNet().to(device)
}

classes = [str(i) for i in range(10)]
for name, model in models_cifar.items():
    print(f"\nОбучение {name}...")
    params = count_parameters(model)
    inf_time = measure_inference_time(model, (64, 3, 32, 32), device)

    # Трекаем градиенты для анализа vanishing/exploding
    history = train_model(model, train_loader, test_loader, epochs=10,
                          device=str(device), track_gradients=True)

    plot_training_history(history, title=f"CIFAR - {name}",
                          save_path=f"plots/cifar_{name.replace(' ', '_')}_history.png")

    # Confusion Matrix
    plot_confusion_matrix(model, test_loader, device, classes,
                          title=f"CM - {name}",
                          save_path=f"plots/cifar_{name.replace(' ', '_')}_cm.png")

    cifar_results[name] = {
        "Params": params,
        "Inf Time (ms)": f"{inf_time:.2f}",
        "Final Test Acc": f"{history['test_accs'][-1]:.4f}",
        "Final Train Acc": f"{history['train_accs'][-1]:.4f}",
        "Overfitting (Train-Test)": f"{history['train_accs'][-1] - history['test_accs'][-1]:.4f}"
    }

compare_models_table(cifar_results,
                     title="CIFAR-10 COMPARISON",
                     save_path="results/cifar_comparison/cifar_comparison_table.png")
print("\nЗадание 1 завершено!")