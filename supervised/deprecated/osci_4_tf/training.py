import tensorflow as tf
import numpy as np
from make_data_new import polyomino_scenes, generate_small
from osci_tools import evolution, coherence_loss, matt_loss
from net import net


"""
Training pipeline in tensorflow version
"""


class TrainPipeline(object):
    def __init__(self,
                 n=1,
                 num=2,
                 lr=1e-5,
                 report=10,
                 batch_size=16,
                 img_side=2):

        # omino type
        self.n = n
        # number of objects
        self.num = num
        # learning rate
        self.lr = lr
        # report steps
        self.report = report
        # batch size
        self.batch = batch_size
        # image size
        self.img_side = img_side

        self.types = 2

        # training steps
        self.steps = 5000

        # iterate steps
        self.episodes = 50

        # start to build the graph

        tf.reset_default_graph()
        # initialize a network
        self.network = net()

        # training placeholders
        self.input = tf.placeholder(tf.float32, [None, self.img_side, self.img_side, 1])
        self.target = tf.placeholder(tf.float32, [None, self.types, self.img_side ** 2])

        # save map generated
        self.maps = []
        # one test image could be used
        self.test_img = np.reshape(np.concatenate([np.zeros((1, 2)), np.ones((1, 2))], axis=0), newshape=[1, 2, 2, 1])

    def opt(self):

        # main graph
        self.network.build(self.input)

        final_phase = evolution(self.episodes, self.batch, self.img_side ** 2, self.network.map)

        # I think 'coherence_loss' works better, but that might depend on params
        loss = matt_loss(final_phase, self.target)

        # prevent gradient from blowing up
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss[2])

        # To clip gradients, use codes below, uncomment and add an optimizer
        #gvs = optimizer.compute_gradients(loss[2])
        #capped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]
        #train_op = optimizer.apply_gradients(capped_gvs)

        # record the losses
        tf.summary.scalar('total loss', loss[2])
        tf.summary.scalar('sync loss', loss[0])
        tf.summary.scalar('desync loss', loss[1])

        # session & initialization
        sess = tf.Session()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train', sess.graph)
        sess.run(tf.global_variables_initializer())

        # self.network.save_npy(sess, './save0.npy')

        # iterations ,optimizations & report
        for step in range(self.steps):
            self.maps.append(sess.run(self.network.map, feed_dict={self.input: self.test_img}))

            # randomly create data for every step
            data = generate_small(self.batch, 2, 2)
            canvas = np.expand_dims(data[0], axis=-1)

            # list of dict -> list of arrays
            masks = list(map(self._mask_pro, data[1]))

            _, summary, l = sess.run([train_op, merged, loss],
                                     feed_dict={self.input: canvas, self.target: masks})

            if step % self.report == 0:
                # if want to log the loss value, open tensorboard and download as '.csv'
                print('STEPS:{}'.format(step) + '|LOSS:', l)
                train_writer.add_summary(summary, step)

        # This function save quicker than original tf.save
        self.network.save_npy(sess, './save.npy')
        np.save('./maps.npy', self.maps)

    def _mask_pro(self, group_dict):
        # create masks for training
        masks = []
        group_names = group_dict.keys()
        for group_name in group_names:
            group_idxes = group_dict[group_name]
            mask = [1. if i in group_idxes else 0. for i in range(self.img_side ** 2)]
            masks.append(mask)
        return masks


if __name__ == '__main__':
    print('TF training pipeline. Should provide a argument parser in the future')
    training = TrainPipeline(n=1, num=2, img_side=2, batch_size=16)
    training.opt()
