from part1.DynetModel import DynetModel


def read_file(filename):
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    return lines


def even_seq():
    train = read_file('data/train_even')
    dev = read_file('data/dev_even')

    vocab = ['a']
    labels = ['0', '1']
    w2i = {w: i for i, w in enumerate(vocab)}
    l2i = {l: i for i, l in enumerate(labels)}

    dynet_model = DynetModel(w2i, l2i, 128, 64, 32)
    dynet_model.train(train, dev, iter_num=10)


def palindrome_seq():
    train = read_file('data/train_pali')
    dev = read_file('data/dev_pali')

    vocab = ['a', 'b']
    labels = ['0', '1']
    w2i = {w: i for i, w in enumerate(vocab)}
    l2i = {l: i for i, l in enumerate(labels)}

    dynet_model = DynetModel(w2i, l2i, layers=16)
    dynet_model.train(train, dev, iter_num=10)


def pow_2_seq():
    train = read_file('data/train_pow')
    dev = read_file('data/dev_pow')

    vocab = ['a', 'b']
    labels = ['0', '1']
    w2i = {w: i for i, w in enumerate(vocab)}
    l2i = {l: i for i, l in enumerate(labels)}

    dynet_model = DynetModel(w2i, l2i)
    dynet_model.train(train, dev, iter_num=2)


def anbn_seq():
    train = read_file('data/train_anbn')
    dev = read_file('data/dev_anbn')

    vocab = ['a', 'b']
    labels = ['0', '1']
    w2i = {w: i for i, w in enumerate(vocab)}
    l2i = {l: i for i, l in enumerate(labels)}

    dynet_model = DynetModel(w2i, l2i, layers=8)
    dynet_model.train(train, dev, iter_num=10)


def div3_seq():
    train = read_file('data/train_3')
    dev = read_file('data/dev_3')

    vocab = [str(i) for i in range(10)]
    labels = ['0', '1']
    w2i = {w: i for i, w in enumerate(vocab)}
    l2i = {l: i for i, l in enumerate(labels)}

    dynet_model = DynetModel(w2i, l2i, 64, layers=16)
    dynet_model.train(train, dev, iter_num=20)


def credit_card_seq():
    train = read_file('data/train_credit_card')
    dev = read_file('data/dev_credit_card')

    vocab = [str(i) for i in range(10)] + ['-']
    labels = ['0', '1']
    w2i = {w: i for i, w in enumerate(vocab)}
    l2i = {l: i for i, l in enumerate(labels)}

    dynet_model = DynetModel(w2i, l2i, layers=4)
    dynet_model.train(train, dev, iter_num=20)


def div3_bin_seq():
    train = read_file('data/train_div3_bin')
    dev = read_file('data/dev_div3_bin')

    vocab = ['0', '1']
    labels = ['0', '1']
    w2i = {w: i for i, w in enumerate(vocab)}
    l2i = {l: i for i, l in enumerate(labels)}

    dynet_model = DynetModel(w2i, l2i, 64, layers=16)
    dynet_model.train(train, dev, iter_num=20)


if __name__ == '__main__':
    print 'start'

    mode = 'div3_bin'
    seqs = {'even': even_seq, 'pali': palindrome_seq, 'pow': pow_2_seq, 'anbn': anbn_seq, 'div3': div3_seq,
            'credit_card': credit_card_seq, 'div3_bin': div3_bin_seq}
    seqs[mode]()
