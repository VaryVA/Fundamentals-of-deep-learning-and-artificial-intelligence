import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_mnist_loaders
from models.cnn_models import KernelSizeCNN, DepthCNN
from utils.training_utils import train_model, measure_inference_time, setup_logging
from utils.visualization_utils import plot_feature_maps, plot_gradient_flow
from utils.comparison_utils import count_parameters, compare_models_table

os.makedirs('results/architecture_analysis', exist_ok=True)
os.makedirs('plots', exist_ok=True)
setup_logging('results/architecture.log')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_mnist_loaders(64)

# 2.1 Влияние размера ядра свертки
print("2.1 Влияние размера ядра свертки")
kernel_results = {}
kernels = [3, 5, 7]

for k in kernels:
    model = KernelSizeCNN(kernel_size=k).to(device)
    params = count_parameters(model)
    inf_time = measure_inference_time(model, (64, 1, 28, 28), device)

    history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))

    # Визуализация активаций первого слоя
    plot_feature_maps(model, train_loader, device, layer_name="0",
                      save_path=f"plots/kernel_{k}x{k}_features.png")

    kernel_results[f"Kernel {k}x{k}"] = {
        "Params": params,
        "Inf Time (ms)": f"{inf_time:.2f}",
        "Test Acc": f"{history['test_accs'][-1]:.4f}",
        "Receptive Field (approx)": f"{k * 2}"
    }

compare_models_table(kernel_results)

#  2.2 Влияние глубины CNN
print("2.2 Влияние глубины CNN")
depth_results = {}
depths = [2, 4, 6]

for d in depths:
    model = DepthCNN(depth=d, use_residual=False).to(device)
    params = count_parameters(model)
    history = train_model(model, train_loader, test_loader, epochs=5,
                          device=str(device), track_gradients=True)

    plot_gradient_flow(history['grad_flow'], title=f"Gradient Flow - Depth {d} (No Res)",
                       save_path=f"plots/depth_{d}_no_res_grad_flow.png")

    min_grad = min([min(v.values()) for v in history['grad_flow']]) if history['grad_flow'] else 0
    depth_results[f"Depth {d} (No Res)"] = {
        "Params": params,
        "Test Acc": f"{history['test_accs'][-1]:.4f}",
        "Min Grad": f"{min_grad:.6f}"
    }

# С Residual связями
model_res = DepthCNN(depth=6, use_residual=True).to(device)
params_res = count_parameters(model_res)
history_res = train_model(model_res, train_loader, test_loader, epochs=5,
                          device=str(device), track_gradients=True)
plot_gradient_flow(history_res['grad_flow'], title="Gradient Flow - Depth 6 (With Res)")
plt.savefig("plots/depth_6_with_res_grad_flow.png")

min_grad_res = min([min(v.values()) for v in history_res['grad_flow']]) if history_res['grad_flow'] else 0
depth_results["Depth 6 (With Res)"] = {
    "Params": params_res,
    "Test Acc": f"{history_res['test_accs'][-1]:.4f}",
    "Min Grad": f"{min_grad_res:.6f}"
}

compare_models_table(depth_results)
print("\nЗадание 2 завершено!")