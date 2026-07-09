import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from tqdm import tqdm

def setup_logging(log_file='experiment.log'):
    '''Настройка логирования'''
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False, track_gradients=False):
    '''Один эпос обучения, тестирования'''
    model.eval() if is_test else model.train()
    total_loss, correct, total = 0.0, 0, 0

    grads = {}
    if track_gradients and not is_test:
        grads = {name: [] for name, p in model.named_parameters()
                 if p.requires_grad and 'bn' not in name and 'shortcut' not in name}

    for data, target in tqdm(data_loader, desc="Epoch" if not is_test else "Test", leave=False):
        data, target = data.to(device), target.to(device)
        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            if track_gradients:
                for name, p in model.named_parameters():
                    if p.grad is not None and name in grads:
                        grads[name].append(p.grad.abs().mean().item())
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    avg_grads = {k: sum(v) / len(v) for k, v in grads.items()} if track_gradients else None
    return total_loss / len(data_loader), correct / total, avg_grads

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu', track_gradients=False):
    '''Полный цикл обучения модели'''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {
        'train_losses': [], 'train_accs': [],
        'test_losses': [], 'test_accs': [],
        'grad_flow': []
    }

    for epoch in range(epochs):
        train_loss, train_acc, grads = run_epoch(
            model, train_loader, criterion, optimizer, device, is_test=False, track_gradients=track_gradients
        )
        test_loss, test_acc, _ = run_epoch(
            model, test_loader, criterion, None, device, is_test=True
        )

        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc)
        if grads:
            history['grad_flow'].append(grads)

        logging.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}"
        )

    return history

def measure_inference_time(model, input_shape, device='cpu', runs=100):
    '''Измерение времени инференса'''
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    model.to(device)

    # Warmup
    for _ in range(10):
        model(dummy_input)

    if device == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(runs):
            model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) / runs
    else:
        start = time.time()
        for _ in range(runs):
            model(dummy_input)
        curr_time = (time.time() - start) * 1000 / runs

    return curr_time