import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('data_num', 100, """Flag of type integer""")
tf.app.flags.DEFINE_string('img_path', './img', """Flag of type string""")


def main(argv):
    print(FLAGS.data_num, FLAGS.img_path)


if __name__ == '__main__':
    tf.app.run()


