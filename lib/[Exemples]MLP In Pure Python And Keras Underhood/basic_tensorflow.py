import tensorflow as tf

if __name__ == "__main__":
    x1 = tf.constant(42.0)
    x2 = tf.constant(51.0)
    y = x1 + x2
    print(y)
