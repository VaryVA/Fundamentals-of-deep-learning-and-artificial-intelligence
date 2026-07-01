import torch


def task_2_1():
    '''Градиенты простой функции'''

    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(4.0, requires_grad=True)

    f = x**2 + y**2 + z**2 + 2 * x * y * z
    f.backward()

    print("df/dx =", x.grad)
    print("df/dy =", y.grad)
    print("df/dz =", z.grad)


def mse(y_pred, y_true):
    '''Mean Squared Error'''
    return torch.mean((y_pred - y_true) ** 2)


def task_2_2():
    '''Градиенты MSE'''

    x = torch.tensor([1., 2., 3.])
    y_true = torch.tensor([2., 4., 6.])

    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    y_pred = w * x + b
    loss = mse(y_pred, y_true)

    loss.backward()

    print("Функция потерь:", loss.item())
    print("dw:", w.grad)
    print("db:", b.grad)


def task_2_3():
    '''Цепное правило'''

    x = torch.tensor(2.0, requires_grad=True)

    f = torch.sin(x**2 + 1)

    grad = torch.autograd.grad(f, x)[0]

    print("Градиент:", grad)


task_2_1()
task_2_2()
task_2_3()