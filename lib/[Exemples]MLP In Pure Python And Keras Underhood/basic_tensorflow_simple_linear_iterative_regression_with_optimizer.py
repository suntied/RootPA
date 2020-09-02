import random
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mean_squared_error

import tensorflow as tf

if __name__ == "__main__":
    opt = SGD(lr=0.1)

    x_train = tf.constant((0.0, 1.0))
    y_train = tf.constant((42.0, 69.0))

    a = tf.Variable(0.0)
    b = tf.Variable(0.0)

    output = a * x_train + b
    print(output)

    for it in range(10000):
        k = random.randint(0, len(x_train) - 1)
        with tf.GradientTape() as tape:
            output = a * x_train[k] + b
            loss = mean_squared_error((y_train[k],), (output,))

        grads = tape.gradient(loss, (a, b))
        print(grads)

        opt.apply_gradients(zip(grads, (a, b)))

        opt.lr = opt.lr * 0.99

        output = a * x_train + b
        print(output)
        print(opt.lr)


