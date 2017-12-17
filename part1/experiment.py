from part1.DynetModel import DynetModel


def read_file(filename):
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    return lines


if __name__ == '__main__':
    print 'start'

    train = read_file('train_set')
    test = read_file('test_set')

    vocab = map(str, range(1, 10)) + ['a', 'b', 'c', 'd']
    labels = ['pos', 'neg']
    w2i = {w: i for i, w in enumerate(vocab)}  # map letter to index
    l2i = {l: i for i, l in enumerate(labels)}  # map label to index

    net = DynetModel(w2i, l2i)
    net.train(train, test)
