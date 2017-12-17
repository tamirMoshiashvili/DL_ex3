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

    dynet_model = DynetModel(w2i, l2i)
    dynet_model.train(train, dev, iter_num=10, lr=1e-5)


def palindrome_seq():
    train = read_file('data/train_pali')
    dev = read_file('data/dev_pali')

    vocab = ['a', 'b']
    labels = ['0', '1']
    w2i = {w: i for i, w in enumerate(vocab)}
    l2i = {l: i for i, l in enumerate(labels)}

    dynet_model = DynetModel(w2i, l2i)
    dynet_model.train(train, dev, iter_num=10)


if __name__ == '__main__':
    print 'start'

    mode = 'pali'
    seqs = {'even': even_seq, 'pali': palindrome_seq}
    seqs[mode]()
