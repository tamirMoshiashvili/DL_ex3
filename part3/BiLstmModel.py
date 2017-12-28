from time import time

import dynet as dy
import numpy as np

UNK = '_UNK_'


def make_data_set(filename):
    """
    :param filename: name of file to read from
    :return: list of sentences, each sentence in it is a tuple of (words, tags), each is a list of strings
    """
    sentences = []
    words, tags = [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.split()
            if not line:  # empty line, end of sentence
                sentences.append((words, tags))
                words, tags = [], []
            else:  # add the data to the current sentence
                word, tag = line
                words.append(word)
                tags.append(tag)
        if words:  # in case that the file doesn't end with empty line
            sentences.append((words, tags))
        f.close()
    return sentences


def extract_word_and_tag_sets_from(data_set):
    """
    :param data_set: list of sentences, each sentence in it is a tuple of (words, tags),
     each is a list of strings
    :return: two-sets, first for words and the other is for tags
    """
    word_set, tag_set = set(), set()
    for words, tags in data_set:
        for word, tag in zip(words, tags):
            word_set.add(word)
            tag_set.add(tag)
    return word_set, tag_set


class BiLstmModel(object):
    def __init__(self, w2i, l2i, embed_dim=64, lstm_in_dim=32, lstm_out_dim=32, layers=1):
        self.w2i = w2i
        self.l2i = l2i
        self.i2l = {i: l for l, i in l2i.iteritems()}

        vocab_size, out_dim = len(w2i), len(l2i)
        self.model = dy.Model()
        self.embed = self.model.add_lookup_parameters((vocab_size, embed_dim))

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
        seq = [wi if wi in self.w2i else UNK for wi in seq]  # check for unknown words
        seq_as_xs = [dy.lookup(self.embed, self.w2i[wi]) for wi in seq]  # current representation is embed

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


if __name__ == '__main__':
    save = True

    t0 = time()
    print 'start'

    train_data_set = make_data_set('../pos/train')
    dev_data_set = make_data_set('../pos/dev')
    w_set, t_set = extract_word_and_tag_sets_from(train_data_set)
    w_set.add(UNK)
    w_to_i = {w: i for i, w in enumerate(w_set)}
    l_to_i = {l: i for i, l in enumerate(t_set)}

    print 'time for loading and parsing the files:', time() - t0
    t0 = time()

    net = BiLstmModel(w_to_i, l_to_i)
    net.train_on(train_data_set, dev_data_set, to_save=save, model_name='model_pos_a')

    print 'time to train:', time() - t0
