from models import FullyConnectedModel


def build_depth_model(num_hidden_layers, base_width=256, use_bn=False, dropout_p=0.0):
    '''
    Строим модель заданной глубины.
    num_hidden_layers: количество скрытых слоев (0 = линейный классификатор)
    '''
    layers = []
    current_width = base_width

    for i in range(num_hidden_layers):
        layers.append({"type": "linear", "size": current_width})
        if use_bn:
            layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        if dropout_p > 0:
            layers.append({"type": "dropout", "rate": dropout_p})
        current_width = max(current_width // 2, 16)  # плавное сужение

    return FullyConnectedModel(input_size=784, num_classes=10, layers=layers)


def build_width_model(hidden_sizes, use_bn=False, dropout_p=0.0):
    '''Строит модель заданной ширины (глубина = len(hidden_sizes))'''
    layers = []
    for size in hidden_sizes:
        layers.append({"type": "linear", "size": size})
        if use_bn:
            layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        if dropout_p > 0:
            layers.append({"type": "dropout", "rate": dropout_p})

    return FullyConnectedModel(input_size=784, num_classes=10, layers=layers)


def build_regularized_model(hidden_sizes, use_bn=False, dropout_p=0.0):
    """Строит модель с заданными параметрами регуляризации."""
    return build_width_model(hidden_sizes, use_bn=use_bn, dropout_p=dropout_p)