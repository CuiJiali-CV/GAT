from loadData import DataSet
import tensorflow as tf
import numpy as np
from utils import *
import time


class gatNet(object):

    def __init__(self, category='Mnist', hidden_dim=[8], weight_decay=5e-3, lr=0.01, att_dropout=0.5,
                 f_drop=0.6, history_dir='./', checkpoint_dir='./', logs_dir='./', Train_Epochs=200, heads=[8, 1],
                 batch_size=1):
        self.category = category
        self.hidden_dim = hidden_dim
        self.weight_decay = weight_decay
        self.att_dropout = att_dropout
        self.f_drop = f_drop
        self.heads = heads
        self.batch_size = batch_size

        self.lr = lr
        self.epoch = Train_Epochs
        self.history_dir = history_dir
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.data = DataSet(category=self.category)
        self.adj, self.features, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask = self.data.Load()

        self.kernel = tf.sparse_placeholder(tf.float32)
        self.input = tf.placeholder(tf.float32, shape=(self.batch_size, np.shape(self.features)[0], np.shape(self.features)[1]))
        self.labels = tf.placeholder(tf.float32, shape=(self.batch_size, None, self.y_train.shape[1]))
        self.labels_mask = tf.placeholder(tf.int32, shape=(self.batch_size, np.shape(self.features)[0]))
        self.num_cls = self.y_train.shape[1]
        self.num_nodes = np.shape(self.features)[0]

    def build(self):
        self.output = self.classifier(self.kernel, self.input, self.num_cls, self.num_nodes, self.heads,
                                      self.hidden_dim, self.f_drop, self.att_dropout)
        self.loss = self.loss_func(self.labels, self.output, self.labels_mask, self.weight_decay)
        self.accuracy = self.masked_accuracy(self.output, self.labels, self.labels_mask)

        self.output_test = self.classifier(self.kernel, self.input, self.num_cls, self.num_nodes, self.heads,
                                      self.hidden_dim, 0.0, 0.0)
        self.accuracy_test = self.masked_accuracy(self.output, self.labels, self.labels_mask)

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def classifier(self, kernel, input, num_cls, num_nodes, heads, hidden_dim, f_drop, att_dropout):
        attns = []
        for _ in range(heads[0]):
            attns.append(self.attn_head(input=input, adj=kernel, num_nodes=num_nodes, output_sz=hidden_dim[0],
                                        attn_drop=att_dropout, f_drop=f_drop))
        h_1 = tf.concat(attns, axis=-1)

        for i in range(1, len(hidden_dim)):
            attns = []
            for _ in range(heads[i]):
                attns.append(self.attn_head(input=h_1, adj=kernel, num_nodes=num_nodes, output_sz=hidden_dim[i],
                                            attn_drop=att_dropout, f_drop=f_drop))
            h_1 = tf.concat(attns, axis=-1)
        output = []
        for i in range(heads[-1]):
            output.append(
                self.attn_head(input=h_1, adj=kernel, num_nodes=num_nodes, output_sz=num_cls, attn_drop=att_dropout,
                               f_drop=f_drop, activate=lambda x: x))
        logits = tf.add_n(output) / heads[-1]

        return logits

    def attn_head(self, input, output_sz, num_nodes, adj, attn_drop, f_drop, activate=tf.nn.elu):

        with tf.name_scope('attn'):
            if f_drop != 0.0:
                input = tf.nn.dropout(input, 1.0 - f_drop)

            combined = tf.layers.conv1d(input, output_sz, 1, use_bias=False)

            f_1 = tf.layers.conv1d(combined, 1, 1)
            f_2 = tf.layers.conv1d(combined, 1, 1)

            f_1 = tf.reshape(f_1, (num_nodes, 1))
            f_2 = tf.reshape(f_2, (num_nodes, 1))

            f_1 = adj * f_1
            f_2 = adj * tf.transpose(f_2, [1, 0])

            output = tf.sparse_add(f_1, f_2)
            output = tf.SparseTensor(indices=output.indices,
                                     values=tf.nn.leaky_relu(output.values),
                                     dense_shape=output.dense_shape)

            coefs = tf.sparse_softmax(output)

            if attn_drop != 0.0:
                coefs = tf.SparseTensor(indices=coefs.indices,
                                        values=tf.nn.dropout(coefs.values, 1.0 - attn_drop),
                                        dense_shape=coefs.dense_shape)
            if f_drop != 0.0:
                combined = tf.nn.dropout(combined, 1.0 - f_drop)

            coefs = tf.sparse_reshape(coefs, [num_nodes, num_nodes])
            combined = tf.squeeze(combined)
            vals = tf.sparse_tensor_dense_matmul(coefs, combined)
            vals = tf.expand_dims(vals, axis=0)
            vals.set_shape([1, num_nodes, output_sz])
            ret = tf.contrib.layers.bias_add(vals)

        return activate(ret)

    def loss_func(self, labels, logits, mask, weight_decay):
        logits = tf.reshape(logits, [-1, tf.shape(labels)[2]])
        labels = tf.reshape(labels, [-1, tf.shape(labels)[2]])
        mask = tf.reshape(mask, [-1])

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        loss = tf.reduce_mean(loss)

        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * weight_decay

        return loss+lossL2

    def masked_accuracy(self, logits, labels, mask):
        logits = tf.reshape(logits, [-1, tf.shape(labels)[2]])
        labels = tf.reshape(labels, [-1, tf.shape(labels)[2]])
        mask = tf.reshape(mask, [-1])

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def train(self, sess):
        self.build()

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)

        writer = tf.summary.FileWriter(self.logs_dir, sess.graph)
        start = 0
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

        if latest_checkpoint:
            latest_checkpoint.split('-')
            start = int(latest_checkpoint.split('-')[-1])
            saver.restore(sess, latest_checkpoint)
            print('Loading checkpoint {}.'.format(latest_checkpoint))

        tf.get_default_graph().finalize()

        features = self.features[np.newaxis]
        y_train = self.y_train[np.newaxis]
        y_val = self.y_val[np.newaxis]
        y_test = self.y_test[np.newaxis]
        train_mask = self.train_mask[np.newaxis]
        val_mask = self.val_mask[np.newaxis]
        test_mask = self.test_mask[np.newaxis]

        for epoch in range(start + 1, self.epoch):
            t = time.time()

            feed_dict = dict()
            feed_dict.update({self.labels: y_train})
            feed_dict.update({self.labels_mask: train_mask})
            feed_dict.update({self.input: features})
            feed_dict.update({self.kernel: self.adj})

            # Training step
            _, train_loss, train_acc = sess.run([self.opt, self.loss, self.accuracy], feed_dict=feed_dict)

            feed_dict.update({self.labels: y_val})
            feed_dict.update({self.labels_mask: val_mask})

            v_loss, v_acc = sess.run([self.loss, self.accuracy_test], feed_dict=feed_dict)

            print("Epoch:", '%06d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
                  "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(v_loss),
                  "val_acc=", "{:.5f}".format(v_acc), "time=", "{:.5f}".format(time.time() - t))

        print("Optimization Finished!")

        feed_dict = dict()
        feed_dict.update({self.labels: y_test})
        feed_dict.update({self.labels_mask: test_mask})
        feed_dict.update({self.input: features})
        feed_dict.update({self.kernel: self.adj})

        test_cost, test_acc = sess.run([self.loss, self.accuracy_test], feed_dict=feed_dict)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost), "accuracy=", "{:.5f}".format(test_acc))
