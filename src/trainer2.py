''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

import model2 as model


class Trainer(object):
    def __init__(self):
        self.batch_sizeK = 1024
        self.batch_sizeA = 32
        self.batch_sizeH = 1024
        self.batch_sizeL1 = 1024
        self.batch_sizeL2 = 1024
        self.dim = 64
        self._m1 = 0.5
        self._a1 = 5.
        self._a2 = 0.5
        self.multiG = None
        self.tf_parts = None
        self.save_path = 'this-model.ckpt'
        self.multiG_save_path = 'this-multiG.bin'
        self.L1 = False
        self.sess = None

    def build(self, multiG, dim=64, batch_sizeK=1024, batch_sizeA=32, batch_sizeH=1024, a1=5., a2=0.5, m1=0.5,
              save_path='this-model.ckpt', multiG_save_path='this-multiG.bin', L1=False):
        self.multiG = multiG
        self.dim = self.multiG.dim = self.multiG.KG1.dim = self.multiG.KG2.dim = dim
        # self.multiG.KG1.wv_dim = self.multiG.KG2.wv_dim = wv_dim
        self.batch_sizeK = self.multiG.batch_sizeK = batch_sizeK
        self.batch_sizeA = self.multiG.batch_sizeA = batch_sizeA
        self.batch_sizeH = self.multiG.batch_sizeH = batch_sizeH
        self.batch_sizeL1 = self.batch_sizeH * self.multiG.KG1.low_high
        self.batch_sizeL2 = self.batch_sizeH * self.multiG.KG2.low_high
        self.multiG_save_path = multiG_save_path
        self.save_path = save_path
        self.L1 = self.multiG.L1 = L1
        self.tf_parts = model.TFParts(num_rels1=self.multiG.KG1.num_rels(),
                                      num_ents1=self.multiG.KG1.num_ents(),
                                      num_rels2=self.multiG.KG2.num_rels(),
                                      num_ents2=self.multiG.KG2.num_ents(),
                                      dim=dim,
                                      batch_sizeK=self.batch_sizeK,
                                      batch_sizeA=self.batch_sizeA,
                                      batch_sizeH=self.batch_sizeH,
                                      batch_sizeL1=self.batch_sizeL1,
                                      batch_sizeL2=self.batch_sizeL2,
                                      L1=self.L1)
        self.tf_parts._m1 = m1
        c = tf.ConfigProto(inter_op_parallelism_threads=3, intra_op_parallelism_threads=3)
        c.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=c)
        sess.run(tf.initialize_all_variables())

    def gen_AD_batch_high(self, KG_index, forever=False, shuffle=True):
        KG = self.multiG.KG1
        if KG_index == 2:
            KG = self.multiG.KG2
        h = len(KG.high_freq)
        while True:
            high_f = np.asarray(KG.high_freq)
            if shuffle:
                np.random.shuffle(high_f)
            for i in range(0, h, self.batch_sizeH):
                batch = high_f[i: i + self.batch_sizeH]
                if batch.shape[0] < self.batch_sizeH:
                    batch = np.concatenate((batch, high_f[:self.batch_sizeH - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeH
                yield batch.astype(np.int64)
            if not forever:
                break

    def gen_AD_batch_low(self, KG_index, forever=False, shuffle=True):
        KG = self.multiG.KG1
        b = self.batch_sizeL1
        if KG_index == 2:
            KG = self.multiG.KG2
            b = self.batch_sizeL2
        l = len(KG.low_freq)
        while True:
            low_f = np.asarray(KG.low_freq)
            if shuffle:
                np.random.shuffle(low_f)
            for i in range(0, l, b):
                batch = low_f[i: i + b]
                if batch.shape[0] < b:
                    batch = np.concatenate((batch, low_f[:b - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == b

                yield batch.astype(np.int64)
            if not forever:
                break

    def gen_KM_batch(self, KG_index, forever=False, shuffle=True):
        KG = self.multiG.KG1
        if KG_index == 2:
            KG = self.multiG.KG2
        l = KG.triples.shape[0]
        while True:
            triples = KG.triples
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, self.batch_sizeK):
                batch = triples[i: i + self.batch_sizeK, :]
                if batch.shape[0] < self.batch_sizeK:
                    batch = np.concatenate((batch, self.multiG.triples[:self.batch_sizeK - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeK
                neg_batch = KG.corrupt_batch(batch)
                h_batch, r_batch, t_batch = batch[:, 0], batch[:, 1], batch[:, 2]
                neg_h_batch, neg_t_batch = neg_batch[:, 0], neg_batch[:, 2]
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), neg_h_batch.astype(
                    np.int64), neg_t_batch.astype(np.int64)
            if not forever:
                break

    def gen_AM_batch(self, forever=False, shuffle=True):
        multiG = self.multiG
        l = len(multiG.align)
        while True:
            align = multiG.align
            if shuffle:
                np.random.shuffle(align)
            for i in range(0, l, self.batch_sizeA):
                batch = align[i: i + self.batch_sizeA, :]
                if batch.shape[0] < self.batch_sizeA:
                    batch = np.concatenate((batch, align[:self.batch_sizeA - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeA
                n_batch = multiG.corrupt_align_batch(batch)
                e1_batch, e2_batch, e1_nbatch, e2_nbatch = batch[:, 0], batch[:, 1], n_batch[:, 0], n_batch[:, 1]
                yield e1_batch.astype(np.int64), e2_batch.astype(np.int64), e1_nbatch.astype(
                    np.int64), e2_nbatch.astype(np.int64)
            if not forever:
                break

    def gen_AM_batch_non_neg(self, forever=False, shuffle=True):
        multiG = self.multiG
        l = len(multiG.align)
        while True:
            align = multiG.align
            if shuffle:
                np.random.shuffle(align)
            for i in range(0, l, self.batch_sizeA):
                batch = align[i: i + self.batch_sizeA, :]
                if batch.shape[0] < self.batch_sizeA:
                    batch = np.concatenate((batch, align[:self.batch_sizeA - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeA
                e1_batch, e2_batch = batch[:, 0], batch[:, 1]
                yield e1_batch.astype(np.int64), e2_batch.astype(np.int64)
            if not forever:
                break

    def train1epoch_KM(self, sess, num_A_batch, num_B_batch, a2, lr, epoch):

        this_gen_A_batch = self.gen_KM_batch(KG_index=1, forever=True)
        this_gen_B_batch = self.gen_KM_batch(KG_index=2, forever=True)

        this_loss = []

        loss_A = loss_B = 0

        for batch_id in range(num_A_batch):
            # Optimize loss A
            A_h_index, A_r_index, A_t_index, A_hn_index, A_tn_index = next(this_gen_A_batch)
            _, loss_A = sess.run([self.tf_parts._train_op_A, self.tf_parts._A_loss],
                                 feed_dict={self.tf_parts._A_h_index: A_h_index,
                                            self.tf_parts._A_r_index: A_r_index,
                                            self.tf_parts._A_t_index: A_t_index,
                                            self.tf_parts._A_hn_index: A_hn_index,
                                            self.tf_parts._A_tn_index: A_tn_index,
                                            self.tf_parts._lr: lr})
            batch_loss = [loss_A]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)
            if ((batch_id + 1) % 500 == 0 or batch_id == num_A_batch - 1):
                print('\rprocess KG1: %d / %d. Epoch %d' % (batch_id + 1, num_A_batch + 1, epoch))

        for batch_id in range(num_B_batch):
            # Optimize loss B
            B_h_index, B_r_index, B_t_index, B_hn_index, B_tn_index = next(this_gen_B_batch)
            _, loss_B = sess.run([self.tf_parts._train_op_B, self.tf_parts._B_loss],
                                 feed_dict={self.tf_parts._B_h_index: B_h_index,
                                            self.tf_parts._B_r_index: B_r_index,
                                            self.tf_parts._B_t_index: B_t_index,
                                            self.tf_parts._B_hn_index: B_hn_index,
                                            self.tf_parts._B_tn_index: B_tn_index,
                                            self.tf_parts._lr: lr})

            # Observe total loss
            batch_loss = [loss_B]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)
            if ((batch_id + 1) % 500 == 0 or batch_id == num_B_batch - 1):
                print('\rprocess KG2: %d / %d. Epoch %d' % (batch_id + 1, num_B_batch + 1, epoch))
        this_total_loss = np.sum(this_loss)
        print("KM Loss of epoch", epoch, ":", this_total_loss)
        # self.logger.info('KM Loss of epoch: {}'.format(this_total_loss))
        print([l for l in this_loss])
        return this_total_loss

    def train1epoch_AM(self, sess, num_AM_batch, a1, a2, lr, epoch):

        # this_gen_AM_batch = self.gen_AM_batch(forever=True)
        this_gen_AM_batch = self.gen_AM_batch_non_neg(forever=True)

        this_loss = []

        loss_AM = 0

        for batch_id in range(num_AM_batch):
            # Optimize loss A
            e1_index, e2_index = next(this_gen_AM_batch)
            _, loss_AM, _, loss_BM, _, loss_trans = sess.run(
                [self.tf_parts._train_op_AM, self.tf_parts._AM_loss, self.tf_parts._train_op_BM, self.tf_parts._BM_loss,
                 self.tf_parts._train_op_trans, self.tf_parts._trans_loss],
                feed_dict={self.tf_parts._AM_index1: e1_index,
                           self.tf_parts._AM_index2: e2_index,
                           # self.tf_parts._AM_nindex1: e1_nindex,
                           # self.tf_parts._AM_nindex2: e2_nindex,
                           self.tf_parts._lr: lr * a1})

            batch_loss = [loss_AM + loss_BM + loss_trans]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)
            if ((batch_id + 1) % 50 == 0) or batch_id == num_AM_batch - 1:
                print('\rprocess: %d / %d. Epoch %d' % (batch_id + 1, num_AM_batch + 1, epoch))
        this_total_loss = np.sum(this_loss)
        print("AM Loss of epoch", epoch, ":", this_total_loss)
        # self.logger('AM Loss of epoch: {}'.format(this_total_loss))
        print([l for l in this_loss])
        return this_total_loss

    def train1epoch_ad(self, sess, num_A_batch, num_B_batch, epoch, lr):
        this_loss = []

        loss_A = loss_B = 0

        this_gen_A_batch_h = self.gen_AD_batch_high(KG_index=1, forever=True)
        this_gen_A_batch_l = self.gen_AD_batch_low(KG_index=1, forever=True)
        this_gen_B_batch_h = self.gen_AD_batch_high(KG_index=2, forever=True)
        this_gen_B_batch_l = self.gen_AD_batch_low(KG_index=2, forever=True)

        for batch_id in range(num_A_batch):
            # Optimize loss A
            A_high = next(this_gen_A_batch_h)
            A_low = next(this_gen_A_batch_l)
            _, loss_A = sess.run([self.tf_parts.train_disc1, self.tf_parts.disc_loss1],
                                 feed_dict={self.tf_parts.disc_input_h1: A_high,
                                            self.tf_parts.disc_input_l1: A_low,
                                            self.tf_parts.lr_ad: lr})
            batch_loss = [loss_A]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)
            if ((batch_id + 1) % 500 == 0 or batch_id == num_A_batch - 1):
                print('\rprocess KG1 adversarial: %d / %d. Epoch %d' % (batch_id + 1, num_A_batch + 1, epoch))

        for batch_id in range(num_B_batch):
            # Optimize loss B
            B_high = next(this_gen_B_batch_h)
            B_low = next(this_gen_B_batch_l)
            _, loss_B = sess.run([self.tf_parts.train_disc2, self.tf_parts.disc_loss2],
                                 feed_dict={self.tf_parts.disc_input_h2: B_high,
                                            self.tf_parts.disc_input_l2: B_low,
                                            self.tf_parts.lr_ad: lr})
            batch_loss = [loss_B]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)
            if ((batch_id + 1) % 500 == 0 or batch_id == num_A_batch - 1):
                print('\rprocess KG1: %d / %d. Epoch %d' % (batch_id + 1, num_A_batch + 1, epoch))
        this_total_loss = np.sum(this_loss)
        print("AD Loss of epoch", epoch, ":", this_total_loss)
        # self.logger("AD loss of epoch {}".format(this_total_loss))
        print([l for l in this_loss])
        return this_total_loss

    def train1epoch_adversarial(self, sess, epoch, ad_fold=1, lr=0.001):
        num_A_batch = int(len(self.multiG.KG1.high_freq) / self.batch_sizeH)

        num_B_batch = int(len(self.multiG.KG2.high_freq) / self.batch_sizeH)

        if epoch <= 1:
            print('num_KG1_batch =', num_A_batch)
            print('num_KG2_batch =', num_B_batch)
        loss_ad = 0
        for i in range(ad_fold):
            loss_ad = self.train1epoch_ad(sess, num_A_batch, num_B_batch, epoch, lr)

        return loss_ad

    def train1epoch_associative(self, sess, lr, a1, a2, epoch, AM_fold=1):

        num_A_batch = int(self.multiG.KG1.num_triples() / self.batch_sizeK)
        num_B_batch = int(self.multiG.KG2.num_triples() / self.batch_sizeK)
        num_AM_batch = int(self.multiG.num_align() / self.batch_sizeA)

        if epoch <= 1:
            print('num_KG1_batch =', num_A_batch)
            print('num_KG2_batch =', num_B_batch)
            print('num_AM_batch =', num_AM_batch)
        loss_KM = self.train1epoch_KM(sess, num_A_batch, num_B_batch, a2, lr, epoch)
        # keep only the last loss
        for i in range(AM_fold):
            loss_AM = self.train1epoch_AM(sess, num_AM_batch, a1, a2, lr, epoch)
        return (loss_KM, loss_AM)

    def train_SEA(self, epochs=20, save_every_epoch=10, lr=0.001, lr_ad=0.001, a1=0.1, a2=0.05, m1=0.5, AM_fold=1,
                      half_loss_per_epoch=-1):
        self.tf_parts._m1 = m1
        t0 = time.time()
        for epoch in range(epochs):
            if half_loss_per_epoch > 0 and (epoch + 1) % half_loss_per_epoch == 0:
                lr = lr * 0.90

            if half_loss_per_epoch > 0 and (epoch + 1) % (half_loss_per_epoch * 2) == 0:
                lr_ad = lr_ad * 0.7

            epoch_lossKM, epoch_lossAM = self.train1epoch_associative(self.sess, lr, a1, a2, epoch, AM_fold)
            epoch_lossAD = self.train1epoch_adversarial(self.sess, epoch, ad_fold=5, lr=lr_ad)
            print("Time use: %d" % (time.time() - t0))
            if np.isnan(epoch_lossKM) or np.isnan(epoch_lossAM) or np.isnan(epoch_lossAD):
            # if np.isnan(epoch_lossKM) or np.isnan(epoch_lossAM):
                print("Training collapsed.")
                return
            if (epoch + 1) % save_every_epoch == 0:
                this_save_path = self.tf_parts._saver.save(self.sess, self.save_path)
                self.multiG.save(self.multiG_save_path)
                print("SEA saved in file: %s. Multi-graph saved in file: %s" % (
                this_save_path, self.multiG_save_path))
                # self.logger.info('SEA saved in file:. Multi-graph saved in file:')
        this_save_path = self.tf_parts._saver.save(self.sess, self.save_path)
        print("SEA saved in file: %s" % this_save_path)
        print("Done")
        # self.logger.info('Done')


def load_tfparts(multiG, dim=64, batch_sizeK=1024, batch_sizeA=64,
                 save_path='this-model.ckpt', L1=False):
    tf_parts = model.TFParts(num_rels1=multiG.KG1.num_rels(),
                             num_ents1=multiG.KG1.num_ents(),
                             num_rels2=multiG.KG2.num_rels(),
                             num_ents2=multiG.KG2.num_ents(),
                             dim=dim,
                             batch_sizeK=batch_sizeK,
                             batch_sizeA=batch_sizeA,
                             L1=L1)
    # with tf.Session() as sess:
    sess = tf.Session()
    tf_parts._saver.restore(sess, save_path)