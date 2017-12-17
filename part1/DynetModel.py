from time import time
import dynet as dy
import numpy as np


class DynetModel(object):
    def __init__(self, w2i, l2i, emb_dim=40, rnn_dim=30, hid_dim=20):
        self.w2i = w2i
        self.l2i = l2i
        self.i2l = {i: l for l, i in l2i.iteritems()}
        vocab_size = len(w2i)
        out_dim = len(l2i)

        self.model = dy.Model()
        self.embed = self.model.add_lookup_parameters((vocab_size, emb_dim))
        self.lstm = dy.LSTMBuilder(1, emb_dim, rnn_dim, self.model)

        # linear1
        self.pW1 = self.model.add_parameters((hid_dim, rnn_dim))
        self.pb1 = self.model.add_parameters(hid_dim)

        # linear2
        self.pW2 = self.model.add_parameters((out_dim, hid_dim))
        self.pb2 = self.model.add_parameters(out_dim)

    def __call__(self, seq):
        dy.renew_cg()

        # feed each item of the sequence
        state = self.lstm.initial_state()
        for c in seq:
            embed = dy.lookup(self.embed, self.w2i[c])
            state = state.add_input(embed)

        w1, b1 = dy.parameter(self.pW1), dy.parameter(self.pb1)
        w2, b2 = dy.parameter(self.pW2), dy.parameter(self.pb2)

        out = state.output()
        out = w1 * out + b1  # linear 1
        out = dy.tanh(out)  # non-linear
        out = w2 * out + b2  # linear 2
        return dy.softmax(out)

    def train(self, train, test, iter_num=1, lr=0.001):
        train = make_data_set(train)
        test = make_data_set(test)
        train_size = len(train)

        trainer = dy.AdamTrainer(self.model, alpha=lr)

        for epoch in range(iter_num):
            np.random.shuffle(train)
            total_loss, good = 0.0, 0.0
            t = time()

            for seq, label in train:
                output = self(seq)  # probabilities
                loss = -dy.log(dy.pick(output, self.l2i[label]))
                total_loss += loss.value()

                pred_label = self.i2l[np.argmax(output.npvalue())]  # as string
                if pred_label == label:
                    good += 1

                loss.backward()  # compute grads
                trainer.update()  # update params

            print epoch, 'loss:', (total_loss / train_size), 'acc:', (good / train_size), 'time:', time() - t
            print self.test(test)

    def test(self, test):
        total_loss, good = 0.0, 0.0
        t = time()

        for seq, label in test:
            output = self(seq)
            loss = -dy.log(dy.pick(output, self.l2i[label]))
            total_loss += loss.value()

            pred_label = self.i2l[np.argmax(output.npvalue())]
            if pred_label == label:
                good += 1

        return '\tloss: ' + str(total_loss / len(test)) + ', acc: ' + str(good / len(test)) + ', time: ' + str(
            time() - t)


def make_data_set(data):
    """
    data - list of lines, each line is 'word label'.
    :return list of tuples, each tuple is (word, label) where each is a tensor.
    """
    for i, line in enumerate(data):
        seq, label = line.split()
        data[i] = (seq, label)
    return data
