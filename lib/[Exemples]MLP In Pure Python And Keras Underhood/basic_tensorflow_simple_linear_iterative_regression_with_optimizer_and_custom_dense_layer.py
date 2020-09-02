import random
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.activations import tanh

import tensorflow as tf


class MyDense:
    def __init__(self, num_neurons: int, input_dim: int, activation=None):
        self.kernel = tf.Variable(tf.random.uniform((num_neurons, input_dim), -1.0, 1.0))
        self.bias = tf.Variable(tf.random.uniform((num_neurons,), -1.0, 1.0))
        self.activation = activation
        self.weights = [self.kernel, self.bias]

    def __call__(self, input_tensor):
        layer_output = tf.matmul(input_tensor, tf.transpose(self.kernel)) + self.bias
        if self.activation:
            layer_output = self.activation(layer_output)
        return layer_output


if __name__ == "__main__":
    opt = SGD(lr=0.001, momentum=0.9)

    x_train = tf.constant(((0.0, 0.0), (1.0, 0.0)))
    y_train = tf.constant(((1.0, -1, -1), (-1, 1, -1)))

    layer1 = MyDense(10, 2, activation=tanh)  # hidden layer, 10 neurons
    layer2 = MyDense(3, 10, activation=tanh)  # output layer, 3 neurons

    output = layer2(layer1(x_train))
    print("Result before training :")
    print(output)

    print("Training ... please wait")
    for it in range(10000):
        k = random.randint(0, len(x_train) - 1)
        with tf.GradientTape() as tape:
            output = layer2(layer1((x_train[k],)))
            loss = mean_squared_error((y_train[k],), (output,))

        grads = tape.gradient(loss, layer1.weights + layer2.weights)

        opt.apply_gradients(zip(grads, layer1.weights + layer2.weights))

        output = layer2(layer1(x_train))

    print("Result after training : ")
    print(output)
