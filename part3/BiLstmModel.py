from time import time

import dynet as dy
import numpy as np
import utils
import pickle

UNK = utils.UNK
DEF_EMB_DIM = utils.DEF_EMB_DIM
DEF_LSTM_IN = utils.DEF_LSTM_IN
DEF_LSTM_OUT = utils.DEF_LSTM_OUT
DEF_LAYERS = utils.DEF_LAYERS


class BiLstmModel(object):
    def __init__(self, model, representor, w2i, l2i,
                 embed_dim=DEF_EMB_DIM, lstm_in_dim=DEF_LSTM_IN, lstm_out_dim=DEF_LSTM_OUT, layers=DEF_LAYERS):
        self.w2i = w2i
        self.l2i = l2i
        self.i2l = {i: l for l, i in l2i.iteritems()}

        vocab_size, out_dim = len(w2i), len(l2i)
        self.model = model
        self.representor = representor

        # bi-lstm-in
        builder = dy.VanillaLSTMBuilder
        self.lstm_in_f = builder(layers, embed_dim, lstm_in_dim, self.model)
        self.lstm_in_b = builder(layers, embed_dim, lstm_in_dim, self.model)

        # bi-lstm-out
        self.lstm_out_f = builder(layers, 2 * lstm_in_dim, lstm_out_dim, self.model)
        self.lstm_out_b = builder(layers, 2 * lstm_in_dim, lstm_out_dim, self.model)

        # linear layer
        self.pW = self.model.add_parameters((out_dim, 2 * lstm_out_dim))
        self.pb = self.model.add_parameters(out_dim)

        self.spec = [w2i, l2i, embed_dim, lstm_in_dim, lstm_out_dim, layers]

    def _bi_lstm(self, seq, builders):
        """ apply bi-lstm builders on the sequence """
        lstm_f = builders[0].initial_state().transduce(seq)
        lstm_b = builders[1].initial_state().transduce(seq[::-1])[::-1]
        return [dy.concatenate([f, b]) for f, b in zip(lstm_f, lstm_b)]

    def __call__(self, seq):
        """ seq = (w1, ... , wi, ... , wn), wi is a word/string """
        seq = [wi if wi in self.w2i else UNK for wi in seq]  # check for unknown words
        seq_as_xs = self.representor.represent(seq)

        # apply bi-lstm twice
        lstm_in = self._bi_lstm(seq_as_xs, [self.lstm_in_f, self.lstm_in_b])
        lstm_out = self._bi_lstm(lstm_in, [self.lstm_out_f, self.lstm_out_b])

        W, b = dy.parameter(self.pW), dy.parameter(self.pb)
        return [W * item + b for item in lstm_out]  # outputs

    def train_on(self, train, dev, to_save=False, model_name=None):
        dev_res_file = open('dev_results.txt', 'w')
        dev_res_file.write('accuracy\n')

        trainer = dy.AdamTrainer(self.model)

        for epoch in range(5):
            train_size = 0
            np.random.shuffle(train)
            total_loss = good = bad = 0.0
            t = time()

            for i, (words, tags) in enumerate(train):
                dy.renew_cg()
                train_size += len(words)

                outputs = self(words)
                errs = []
                for output, tag in zip(outputs, tags):
                    errs.append(dy.pickneglogsoftmax(output, self.l2i[tag]))  # include softmax

                    # update accuracy
                    pred_tag = self.i2l[np.argmax(output.npvalue())]
                    if pred_tag == tag:
                        good += 1
                    else:
                        bad += 1

                loss = dy.esum(errs)
                total_loss += loss.value()
                loss.backward()
                trainer.update()

                if i % 500 == 499:  # check dev accuracy every 500 sentences
                    dev_acc = self.check_on_dev(dev, i)
                    dev_res_file.write(str(dev_acc) + '\n')

            print epoch, 'loss:', (total_loss / train_size), \
                'acc:', (good / (good + bad)), \
                'time:', time() - t, '\n'
        dev_res_file.close()

        if to_save:
            self.save_model(model_name)

    def check_on_dev(self, dev, i):
        """ predict tags from dev and check the loss and accuracy on it """
        total_loss = good = bad = 0.0
        t = time()
        dev_size = 0

        for words, tags in dev:
            dev_size += len(words)
            outputs = self(words)

            errs = []
            for output, tag in zip(outputs, tags):
                errs.append(dy.pickneglogsoftmax(output, self.l2i[tag]))

                pred_tag = self.i2l[np.argmax(output.npvalue())]
                if pred_tag == tag:
                    good += 1
                else:
                    bad += 1
            loss = dy.esum(errs)
            total_loss += loss.value()
        print str(i + 1) + ': ' + 'loss:', (total_loss / dev_size), \
            'acc:', (good / (good + bad)), \
            'time:', time() - t
        return good / (good + bad)

    def save_model(self, name):
        obj = {'model': self.model, 'w2i': self.w2i, 'l2i': self.l2i}
        pickle.dump(obj, open('model_' + name, 'wb'))

    @staticmethod
    def load_model(filename):
        reader = pickle.load(open('model_' + filename, 'rb'))
        m = reader['model']
        w2i = reader['w2i']
        l2i = reader['l2i']
        return BiLstmModel(m, w2i, l2i)


def old():
    save = True

    t0 = time()
    print 'start'

    train_data_set = utils.make_data_set('../pos/train')
    dev_data_set = utils.make_data_set('../pos/dev')
    w_set, t_set = utils.extract_word_and_tag_sets_from(train_data_set)
    w_set.add(UNK)
    w_to_i = {w: i for i, w in enumerate(w_set)}
    l_to_i = {l: i for i, l in enumerate(t_set)}

    print 'time for loading and parsing the files:', time() - t0
    t0 = time()

    # pc = dy.ParameterCollection()
    # net = BiLstmModel(pc, w_to_i, l_to_i)
    # net.train_on(train_data_set, dev_data_set, to_save=save, model_name='pos_a')

    print 'time to train:', time() - t0


if __name__ == '__main__':
    d = {'a': 1, 'b': 2}
    pickle.dump(d, open('exp', 'wb'))

    d2 = pickle.load(open('exp', 'rb'))
    print d2
