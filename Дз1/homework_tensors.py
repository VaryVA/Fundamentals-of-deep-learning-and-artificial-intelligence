import torch

def task_1_1():
    '''Создание тензоров'''

    tensor_random = torch.rand(3, 4)
    tensor_zeros = torch.zeros(2, 3, 4)
    tensor_ones = torch.ones(5, 5)
    tensor_range = torch.arange(16).reshape(4, 4)

    print("Cлучайные числа:\n", tensor_random)
    print("Нули:\n", tensor_zeros)
    print("Еденички:\n", tensor_ones)
    print("От 0 до 15:\n", tensor_range)


def task_1_2():
    '''Операции с тензорами'''

    A = torch.rand(3, 4)
    B = torch.rand(4, 3)

    A_T = A.T
    matmul = torch.matmul(A, B)
    elementwise = A * B.T
    total_sum = A.sum()

    print("Транспонирование:\n", A_T)
    print("Матричное умнодение:\n", matmul)
    print("Поэлементное умножение:\n", elementwise)
    print("Сумма всех элементов тензора A:", total_sum)


def task_1_3():
    '''Индексация'''

    tensor = torch.arange(125).reshape(5, 5, 5)

    first_row = tensor[:, 0, :]
    last_column = tensor[:, :, -1]

    # центральная подматрица 2x2
    center = tensor[:, 2:4, 2:4]

    even_indices = tensor[::2, ::2, ::2]

    print("Первая колонка:\n", first_row)
    print("Последняя колонка:\n", last_column)
    print("Подматрица из центра 2x2:\n", center)
    print("Элементы с четными индексами:\n", even_indices)


def task_1_4():
    '''Работа с формами'''

    tensor = torch.arange(24)

    print(tensor.reshape(2, 12))
    print(tensor.reshape(3, 8))
    print(tensor.reshape(4, 6))
    print(tensor.reshape(2, 3, 4))
    print(tensor.reshape(2, 2, 2, 3))


task_1_1()
task_1_2()
task_1_3()
task_1_4()