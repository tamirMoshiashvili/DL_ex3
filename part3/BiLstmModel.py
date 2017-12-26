import dynet as dy


class BiLstmModel(object):
    def __init__(self, w2i, l2i, emb_dim, rnn_dim, layers):
        self.w2i = w2i
        self.l2i = l2i

        self.model = dy.Model()
        vocab_size = len(w2i)
        out_dim = len(l2i)
        self.embed = self.model.add_lookup_parameters((vocab_size, emb_dim))

        # bi-lstm
        self.lstm_f = dy.VanillaLSTMBuilder(layers, emb_dim, rnn_dim, self.model)
        self.lstm_b = dy.VanillaLSTMBuilder(layers, emb_dim, rnn_dim, self.model)

        # linear layer
        self.pW = self.model.add_parameters((out_dim, 2 * rnn_dim))
        self.pb = self.model.add_parameters(out_dim)

    def _bi_lstm(self, seq):
        seq_as_bs = []
        for i in range(len(seq)):
            state_f = self.lstm_f.initial_state()
            for x in seq[:i + 1]:
                state_f = state_f.add_input(x)

            state_b = self.lstm_b.initial_state()
            for x in reversed(seq[i:]):
                state_b = state_b.add_input(x)

            # create bi
            seq_as_bs.append(dy.concatenate([state_f.output(), state_b.output()]))
        return seq_as_bs

    def __call__(self, seq):
        """ seq = (w1, ... , wi, ... , wn ), wi is a word/string """
        seq_as_xs = [dy.lookup(self.embed, self.w2i[wi]) for wi in seq]  # current representation is embed
        seq_as_bs = self._bi_lstm(seq_as_xs)
        seq_as_b_tags = self._bi_lstm(seq_as_bs)

        dy.renew_cg()

        W, b = dy.parameter(self.pW), dy.parameter(self.pb)

        outputs = []
        for b_tag in seq_as_b_tags:
            out = W * b_tag + b
            outputs.append(dy.softmax(out))
        return outputs


if __name__ == '__main__':
    print 'hello'
