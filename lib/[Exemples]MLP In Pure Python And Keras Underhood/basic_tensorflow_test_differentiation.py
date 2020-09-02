import tensorflow as tf

if __name__ == "__main__":
    x = tf.Variable(1.0)
    a = tf.constant(42.0)
    b = tf.constant(51.0)

    with tf.GradientTape() as tape:
        y = a * x * x + b

    grads = tape.gradient(y, x)

    print(grads)




