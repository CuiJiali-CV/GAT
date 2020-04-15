# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data:

from gat_net import gatNet
import tensorflow as tf
import os
import shutil


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('category', 'cora', 'DataSet category')

tf.flags.DEFINE_float('att_dropout', 0.6, 'Dropout rate (1 - keep probability).')
tf.flags.DEFINE_float('f_drop', 0.6, 'Dropout rate (1 - keep probability).')
tf.flags.DEFINE_list('hidden_dim', [8], 'hidden layers')
tf.flags.DEFINE_list('heads', [8, 1], 'hidden layers')
tf.flags.DEFINE_integer('Train_Epochs', 100000, 'Number of Epochs to train')
tf.flags.DEFINE_float('weight_decay', 0.0005, 'Weight for L2 loss on embedding matrix.')
tf.flags.DEFINE_float('lr', 0.005, 'learning rate')

tf.flags.DEFINE_string('history_dir', './output/history/', 'history')
tf.flags.DEFINE_string('checkpoint_dir', './output/checkpoint/', 'checkpoint')
tf.flags.DEFINE_string('logs_dir', './output/logs/', 'logs')

def main(_):

    model = gatNet(
        category=FLAGS.category,
        Train_Epochs=FLAGS.Train_Epochs,
        lr=FLAGS.lr,
        history_dir=FLAGS.history_dir,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logs_dir=FLAGS.logs_dir,
        weight_decay=FLAGS.weight_decay,
        att_dropout=FLAGS.att_dropout,
        f_drop=FLAGS.f_drop,
        heads=FLAGS.heads,
        hidden_dim=FLAGS.hidden_dim
    )

    continueTrain = False
    # continueTrain = True
    with tf.Session() as sess:
        if not continueTrain:
            if os.path.exists(FLAGS.checkpoint_dir):
                shutil.rmtree(FLAGS.checkpoint_dir[:-1])
            os.makedirs(FLAGS.checkpoint_dir)

        if os.path.exists(FLAGS.logs_dir):
            shutil.rmtree(FLAGS.logs_dir[:-1])
        os.makedirs(FLAGS.logs_dir)

        if not os.path.exists(FLAGS.history_dir):
            os.makedirs(FLAGS.history_dir)

        model.train(sess)


if __name__ == '__main__':
    tf.app.run()
