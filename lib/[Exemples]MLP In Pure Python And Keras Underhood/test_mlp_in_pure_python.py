import random
from math import tanh
from typing import List


class MyMLP:
    def __init__(self, npl: List[int]):
        self.npl: List[int] = npl.copy()
        self.L: int = len(npl) - 1
        self.w: List[List[List[float]]] = []
        self.w.append([])
        for layer in range(1, self.L + 1):
            self.w.append([])
            for i in range(npl[layer - 1] + 1):
                self.w[layer].append([])
                for j in range(npl[layer] + 1):
                    self.w[layer][i].append(random.random() * 2.0 - 1.0)  # between [-1.0, 1.0)

        self.deltas: List[List[float]] = []
        self.deltas.append([])
        for layer in range(1, self.L + 1):
            self.deltas.append([])
            for j in range(npl[layer] + 1):
                self.deltas[layer].append(0.0)  # value doesn't matter

        self.x: List[List[float]] = []

        for layer in range(self.L + 1):
            self.x.append([])
            for j in range(npl[layer] + 1):
                if j == 0:
                    self.x[layer].append(1.0)  # Bias neuron output !
                else:
                    self.x[layer].append(0.0)  # value doesn't matter


def mlp_create(npl: List[int]):
    return MyMLP(npl)


def _mlp_predict_common(
        mlp: MyMLP,
        sample_inputs: List[float],
        classification_mode: bool):

    for j in range(1, mlp.npl[0] + 1):
        mlp.x[0][j] = sample_inputs[j - 1]

    for layer in range(1, mlp.L + 1):
        for j in range(1, mlp.npl[layer] + 1):
            result = 0.0
            for i in range(0, mlp.npl[layer - 1] + 1):
                result += mlp.w[layer][i][j] * mlp.x[layer - 1][i]
            if layer != mlp.L or classification_mode:
                result = tanh(result)
            mlp.x[layer][j] = result

    return mlp.x[mlp.L][1:]


def mlp_predict_classification(mlp: MyMLP,
                               sample_inputs: List[float]):
    return _mlp_predict_common(mlp, sample_inputs, True)


def mlp_predict_regression(mlp: MyMLP,
                           sample_inputs: List[float]):
    return _mlp_predict_common(mlp, sample_inputs, False)


def _mlp_train_common(mlp: MyMLP,
                      dataset_inputs: List[List[float]],
                      dataset_expected_outputs: List[List[float]],
                      iteration_count: int,
                      alpha: float,  # learning rate
                      classification_mode: bool
                      ):
    for it in range(iteration_count):
        k = random.randint(0, len(dataset_inputs) - 1)
        _mlp_predict_common(mlp, dataset_inputs[k], classification_mode)
        for j in range(1, mlp.npl[mlp.L] + 1):
            mlp.deltas[mlp.L][j] = mlp.x[mlp.L][j] - dataset_expected_outputs[k][j - 1]
            if classification_mode:
                mlp.deltas[mlp.L][j] *= 1 - mlp.x[mlp.L][j] * mlp.x[mlp.L][j]

        for layer in reversed(range(2, mlp.L + 1)):
            for i in range(1, mlp.npl[layer - 1] + 1):
                result = 0.0
                for j in range(1, mlp.npl[layer] + 1):
                    result += mlp.w[layer][i][j] * mlp.deltas[layer][j]
                result *= 1 - mlp.x[layer - 1][i] * mlp.x[layer - 1][i]
                mlp.deltas[layer - 1][i] = result

        for layer in range(1, mlp.L + 1):
            for i in range(0, mlp.npl[layer - 1] + 1):
                for j in range(1, mlp.npl[layer] + 1):
                    mlp.w[layer][i][j] -= alpha * mlp.x[layer - 1][i] * mlp.deltas[layer][j]


def mlp_train_classification(mlp: MyMLP,
                             dataset_inputs: List[List[float]],
                             dataset_expected_outputs: List[List[float]],
                             iteration_count: int,
                             alpha: float  # learning rate
                             ):
    _mlp_train_common(mlp, dataset_inputs, dataset_expected_outputs, iteration_count, alpha, True)


def mlp_train_regression(mlp: MyMLP,
                         dataset_inputs: List[List[float]],
                         dataset_expected_outputs: List[List[float]],
                         iteration_count: int,
                         alpha: float  # learning rate
                         ):
    _mlp_train_common(mlp, dataset_inputs, dataset_expected_outputs, iteration_count, alpha, False)


def mlp_dispose():
    pass


if __name__ == "__main__":
    x_train = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]

    y_train = [
        [-1],
        [1],
        [1],
        [-1],
    ]

    my_mlp = mlp_create([2, 2, 1])

    mlp_train_classification(my_mlp, x_train, y_train, 100000, 0.1)

    print(mlp_predict_classification(my_mlp, x_train[0]))
    print(mlp_predict_classification(my_mlp, x_train[1]))
    print(mlp_predict_classification(my_mlp, x_train[2]))
    print(mlp_predict_classification(my_mlp, x_train[3]))

    y_train = [
        [-3],
        [2],
        [8],
        [-5],
    ]

    my_mlp = mlp_create([2, 5, 1])

    mlp_train_regression(my_mlp, x_train, y_train, 100000, 0.1)

    print(mlp_predict_regression(my_mlp, x_train[0]))
    print(mlp_predict_regression(my_mlp, x_train[1]))
    print(mlp_predict_regression(my_mlp, x_train[2]))
    print(mlp_predict_regression(my_mlp, x_train[3]))

    y_train = [
        [1, -1, -1],
        [-1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ]

    my_mlp = mlp_create([2, 5, 3])

    mlp_train_classification(my_mlp, x_train, y_train, 100000, 0.1)

    print(mlp_predict_classification(my_mlp, x_train[0]))
    print(mlp_predict_classification(my_mlp, x_train[1]))
    print(mlp_predict_classification(my_mlp, x_train[2]))
    print(mlp_predict_classification(my_mlp, x_train[3]))

    y_train = [
        [1, -1, 3],
        [2, 2, 4],
        [8, -1.5, -1],
        [-5, -1, 2],
    ]

    my_mlp = mlp_create([2, 5, 3])

    mlp_train_regression(my_mlp, x_train, y_train, 100000, 0.1)

    print(mlp_predict_regression(my_mlp, x_train[0]))
    print(mlp_predict_regression(my_mlp, x_train[1]))
    print(mlp_predict_regression(my_mlp, x_train[2]))
    print(mlp_predict_regression(my_mlp, x_train[3]))

    exit(0)
