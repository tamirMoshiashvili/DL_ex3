import dynet as dy


class BaseRepresentation(object):
    def represent(self, seq):
        pass


class WordRepresentation(BaseRepresentation):
    def __init__(self, w2i, model, emb_dim):
        self.w2i = w2i
        self.embed = model.add_lookup_parameters((len(w2i), emb_dim))

    def represent(self, seq):
        return [dy.lookup(self.embed, self.w2i[wi]) for wi in seq]


class CharLevelRepresentation(BaseRepresentation):
    def __init__(self, c2i, model, emb_dim, lstm_dim, layers=1):
        self.c2i = c2i
        self.embed = model.add_lookup_parameters((len(c2i), emb_dim))
        self.builder = dy.VanillaLSTMBuilder(layers, emb_dim, lstm_dim, model)

    def represent(self, seq):
        output_vec = []
        s0 = self.builder.initial_state()
        for word in seq:
            word_as_char_vec = [dy.lookup(self.embed, self.c2i[ci]) for ci in word]
            word_output = s0.transduce(word_as_char_vec)[-1]  # apply lstm and take last output
            output_vec.append(word_output)
        return output_vec


class SubWordRepresentation(BaseRepresentation):
    def __init__(self):
        pass

    def represent(self, seq):
        pass


class WordAndCharRepresentation(BaseRepresentation):
    def __init__(self):
        pass

    def represent(self, seq):
        pass


def resolve_repr(representation, args):
    """
    :param representation: one of a, b, c, d
    :param args: tuple of:
            0 model
            1 w2i
            2 embed dim
    :return:
    """
    if representation == 'a':
        model, w2i, emb_dim = args[0], args[1], args[2]
        return WordRepresentation(w2i, model, emb_dim)
