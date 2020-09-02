import tensorflow as tf

if __name__ == "__main__":

    tf.compat.v1.disable_eager_execution()

    x1 = tf.constant(42.0)
    x2 = tf.constant(51.0)
    y = x1 + x2

    with tf.python.Session() as sess:
        print(sess.run(y))
