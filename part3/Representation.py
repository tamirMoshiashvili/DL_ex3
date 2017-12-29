import dynet as dy


class BaseRepresentation(object):
    def represent(self, seq):
        pass


class WordRepresentation(BaseRepresentation):
    def __init__(self, w2i, model, emb_dim):
        self.w2i = w2i
        self.embed = model.add_lookup_parameters((len(w2i), emb_dim))

        from part3 import utils
        self.unk = utils.UNK

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
    def __init__(self):
        pass

    def represent(self, seq):
        pass


def resolve_repr(representation, args):
    """
    :param representation: one of a, b, c, d
    :param args: tuple
    :return: representor
    """
    if representation == 'a':
        model, w2i, emb_dim = args
        return WordRepresentation(w2i, model, emb_dim)
    elif representation == 'b':
        model, c2i, emb_dim, lstm_dim, layers = args
        return CharLevelRepresentation(c2i, model, emb_dim, lstm_dim, layers)
    else:
        return None
