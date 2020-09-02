import random

import tensorflow as tf

if __name__ == "__main__":
    x_train = tf.constant((0.0, 1.0))
    y_train = tf.constant((42.0, 69.0))

    a = tf.Variable(0.0)
    b = tf.Variable(0.0)

    output = a * x_train + b
    print(output)

    for it in range(100):
        k = random.randint(0, len(x_train) - 1)
        with tf.GradientTape() as tape:
            output = a * x_train[k] + b
            error = y_train[k] - output
            squared_error = error * error
            mean_squared_error = tf.reduce_mean(squared_error)
            loss = mean_squared_error

        grads = tape.gradient(loss, (a, b))
        print(grads)

        a.assign(a.numpy() - 0.1 * grads[0])
        b.assign(b.numpy() - 0.1 * grads[1])

        output = a * x_train + b
        print(output)


