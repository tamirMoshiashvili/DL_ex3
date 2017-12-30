import dynet as dy

S_MODEL = 'model'
S_W2I = 'w2i'
S_EMB_DIM = 'emb_dim'
S_C2I = 'c2i'
S_LSTM_DIM = 'lstm_dim'
S_LAYERS = 'layers'
S_W_EMB_DIM = 'w_emb_dim'
S_C_EMB_DIM = 'c_emb_dim'
S_TOTAL_DIM = 'total_dim'


class BaseRepresentation(object):
    def represent(self, seq):
        pass


class WordRepresentation(BaseRepresentation):
    def __init__(self, w2i, model, emb_dim):
        self.w2i = w2i
        self.embed = model.add_lookup_parameters((len(w2i), emb_dim))

        from part3 import utils
        self.unk = utils.UNK

        self.spec = {S_W2I: w2i, S_EMB_DIM: emb_dim}

    def represent(self, seq):
        return [dy.lookup(self.embed, self.w2i[wi]) if wi in self.w2i
                else dy.lookup(self.embed, self.w2i[self.unk])
                for wi in seq]


class CharLevelRepresentation(BaseRepresentation):
    def __init__(self, c2i, model, emb_dim, lstm_dim, layers=1):
        self.c2i = c2i
        self.embed = model.add_lookup_parameters((len(c2i), emb_dim))
        self.builder = dy.VanillaLSTMBuilder(layers, emb_dim, lstm_dim, model)

        from part3 import utils
        self.c_unk = utils.CUNK

        self.spec = {S_C2I: c2i, S_EMB_DIM: emb_dim, S_LSTM_DIM: lstm_dim, S_LAYERS: layers}

    def represent(self, seq):
        output_vec = []
        s0 = self.builder.initial_state()
        for word in seq:
            word_as_char_vec = [dy.lookup(self.embed, self.c2i[ci]) if ci in self.c2i
                                else dy.lookup(self.embed, self.c2i[self.c_unk])
                                for ci in word]
            word_output = s0.transduce(word_as_char_vec)[-1]  # apply lstm and take last output
            output_vec.append(word_output)
        return output_vec


class SubWordRepresentation(WordRepresentation):
    def __init__(self, w2i, model, emb_dim):
        super(SubWordRepresentation, self).__init__(w2i, model, emb_dim)
        from part3 import utils
        self.pref_flag = utils.PREF_FLAG
        self.suff_flag = utils.SUFF_FLAG
        self.pref_unk = utils.PREF_UNK
        self.suff_unk = utils.SUFF_UNK

    def represent(self, seq):
        word_r = super(SubWordRepresentation, self).represent(seq)
        pref_r = [dy.lookup(self.embed, self.w2i[self.pref_flag + w[:3]]) if self.pref_flag + w[:3] in self.w2i
                  else dy.lookup(self.embed, self.w2i[self.pref_unk])
                  for w in seq]
        suff_r = [dy.lookup(self.embed, self.w2i[self.suff_flag + w[-3:]]) if self.suff_flag + w[-3:] in self.w2i
                  else dy.lookup(self.embed, self.w2i[self.suff_unk])
                  for w in seq]
        return [pref_r[i] + word_r[i] + suff_r[i] for i in range(len(word_r))]


class WordAndCharRepresentation(BaseRepresentation):
    def __init__(self, model, w2i, word_emb_dim, c2i, char_emb_dim, lstm_dim, total_dim, layers=1):
        self.w_repr = WordRepresentation(w2i, model, word_emb_dim)
        self.c_repr = CharLevelRepresentation(c2i, model, char_emb_dim, lstm_dim, layers)
        self.pW = model.add_parameters((total_dim, word_emb_dim + char_emb_dim))
        self.pb = model.add_parameters(total_dim)

        self.spec = {S_W2I: w2i, S_C2I: c2i, S_LSTM_DIM: lstm_dim, S_LAYERS: layers,
                     S_W_EMB_DIM: word_emb_dim, S_C_EMB_DIM: char_emb_dim, S_TOTAL_DIM: total_dim}

    def represent(self, seq):
        word_repr = self.w_repr.represent(seq)
        char_repr = self.c_repr.represent(seq)

        W, b = dy.parameter(self.pW), dy.parameter(self.pb)
        return [W * dy.concatenate([word_repr[i], char_repr[i]]) + b for i in range(len(seq))]


def resolve_repr(representation, args):
    """
    :param representation: one of a, b, c, d
    :param args: dict
    :return: representor
    """
    if representation == 'a':
        model, w2i, emb_dim = args[S_MODEL], args[S_W2I], args[S_EMB_DIM]
        return WordRepresentation(w2i, model, emb_dim)
    elif representation == 'b':
        model, c2i, emb_dim, lstm_dim, layers = args[S_MODEL], args[S_C2I], \
                                                args[S_EMB_DIM], args[S_LSTM_DIM], args[S_LAYERS]
        return CharLevelRepresentation(c2i, model, emb_dim, lstm_dim, layers)
    elif representation == 'c':
        model, w2i, emb_dim = args[S_MODEL], args[S_W2I], args[S_EMB_DIM]
        return SubWordRepresentation(w2i, model, emb_dim)
    elif representation == 'd':
        model, w2i, word_emb_dim, c2i = args[S_MODEL], args[S_W2I], args[S_W_EMB_DIM], args[S_C2I]
        char_emb_dim, lstm_dim, total_dim, layers = args[S_C_EMB_DIM], args[S_LSTM_DIM], args[S_TOTAL_DIM], args[
            S_LAYERS]
        return WordAndCharRepresentation(model, w2i, word_emb_dim, c2i, char_emb_dim, lstm_dim, total_dim, layers)
    else:
        return None
