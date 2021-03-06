import pickle
from time import time

import numpy as np

import utils
from part3.Representation import *

DEF_EMB_DIM = utils.DEF_EMB_DIM
DEF_LSTM_IN = utils.DEF_LSTM_IN
DEF_LSTM_OUT = utils.DEF_LSTM_OUT
DEF_LAYERS = utils.DEF_LAYERS


class BiLstmModel(object):
    def __init__(self, model, representor, l2i,
                 embed_dim=DEF_EMB_DIM, lstm_in_dim=DEF_LSTM_IN, lstm_out_dim=DEF_LSTM_OUT, layers=DEF_LAYERS):
        self.l2i = l2i
        self.i2l = {i: l for l, i in l2i.iteritems()}

        self.param_tuple = (embed_dim, lstm_in_dim, lstm_out_dim, layers)  # for saving and loading

        out_dim = len(l2i)
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

    def _bi_lstm(self, seq, builders):
        """ apply bi-lstm builders on the sequence """
        lstm_f = builders[0].initial_state().transduce(seq)
        lstm_b = builders[1].initial_state().transduce(seq[::-1])[::-1]
        return [dy.concatenate([f, b]) for f, b in zip(lstm_f, lstm_b)]

    def __call__(self, seq):
        """ seq = (w1, ... , wi, ... , wn), wi is a word/string """
        seq_as_xs = self.representor.represent(seq)

        # apply bi-lstm twice
        lstm_in = self._bi_lstm(seq_as_xs, [self.lstm_in_f, self.lstm_in_b])
        lstm_out = self._bi_lstm(lstm_in, [self.lstm_out_f, self.lstm_out_b])

        W, b = dy.parameter(self.pW), dy.parameter(self.pb)
        return [W * item + b for item in lstm_out]  # outputs

    def train_on(self, train, dev, to_save=False, model_name=None):
        """ if to_save set to True, user must specify the model_name """
        dev_res_file = open('dev_results.txt', 'w')
        dev_res_file.write('accuracy\n')
        best_dev_acc = 0.0

        trainer = dy.AdamTrainer(self.model)

        for epoch in range(5):
            train_size = 0
            np.random.shuffle(train)
            total_loss = good = bad = 0.0
            t = time()

            ignore_tag = utils.IGNORE_TAG  # O-tag of ner
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
                        if tag == ignore_tag:
                            continue
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
                    if to_save and dev_acc > best_dev_acc:
                        self.save_model(model_name)
                        best_dev_acc = dev_acc

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

        ignore_tag = utils.IGNORE_TAG  # O-tag of ner
        errs = []
        for words, tags in dev:
            dy.renew_cg()

            dev_size += len(words)
            outputs = self(words)

            for output, tag in zip(outputs, tags):
                errs.append(dy.pickneglogsoftmax(output, self.l2i[tag]))

                pred_tag = self.i2l[np.argmax(output.npvalue())]
                if pred_tag == tag:
                    if tag == ignore_tag:
                        continue
                    good += 1
                else:
                    bad += 1
            loss = dy.esum(errs)
            total_loss += loss.value()
            errs = []
        print str(i + 1) + ': ' + 'loss:', (total_loss / dev_size), \
            'acc:', (good / (good + bad)), \
            'time:', time() - t
        return good / (good + bad)

    def save_model(self, name):
        self.model.save(name)  # save the model file

        obj = {'repr': self.representor.spec, 'l2i': self.l2i, 'param_tuple': self.param_tuple}
        pickle.dump(obj, open(name + '.params', 'wb'))

    @staticmethod
    def load_model(filename, representation):
        # filename += '_' + representation
        reader = pickle.load(open(filename + '.params', 'rb'))
        repr_spec = reader['repr']
        l2i = reader['l2i']
        param_tuple = reader['param_tuple']
        m = dy.ParameterCollection()

        repr_spec[S_MODEL] = m
        representor = resolve_repr(representation, repr_spec)
        emb_dim, lstm_in_dim, lstm_out_dim, layers = param_tuple
        net = BiLstmModel(m, representor, l2i, emb_dim, lstm_in_dim, lstm_out_dim, layers)
        m.populate(filename)

        return net

    def test_on(self, test, output_file_path):
        t_test = time()
        output_file = open(output_file_path, 'w')

        for sentence in test:
            dy.renew_cg()

            outputs = self(sentence)
            predicted_tags = [self.i2l[np.argmax(output.npvalue())] for output in outputs]

            for w, t in zip(sentence, predicted_tags):
                output_file.write(w + ' ' + t + '\n')
            output_file.write('\n')  # end of sentence

        output_file.close()
        print 'time for blind test:', time() - t_test


if __name__ == '__main__':
    tup = (1, 2, 3)
    print tup + (4,)
