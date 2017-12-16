VOCAB = map(str, range(1, 10)) + ['a', 'b', 'c', 'd']
LABELS = ['pos', 'neg']
W2I = {w: i for i, w in enumerate(VOCAB)}
I2W = {i: w for w, i in W2I.iteritems()}
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.iteritems()}


def read_data_file(filename):
    """
    filename - name of set-file, each line in it is in form of 'seq label'.
    :return list of tuples, each tuple is (seq, label).
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        f.close()
        for i, line in enumerate(lines):
            lines[i] = line.split()
        return lines


if __name__ == '__main__':
    print 'hello'
