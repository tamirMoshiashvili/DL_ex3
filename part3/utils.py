from time import time

UNK = '_UNK_'
CUNK = '_C' + UNK
PREF_FLAG = '_PREF_'
SUFF_FLAG = '_SUFF_'
PREF_UNK = PREF_FLAG + UNK
SUFF_UNK = SUFF_FLAG + UNK

DEF_W_EMB_DIM = 64
DEF_C_EMB_DIM = 64
DEF_TOTAL_DIM = 64
DEF_EMB_DIM = 64
DEF_LSTM_IN = 32
DEF_LSTM_OUT = 32
DEF_LAYERS = 1

IGNORE_TAG = 'O'


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


def create_c2i(train):
    t_c = time()
    chars = set(CUNK)
    for sentence, _ in train:
        for word in sentence:
            for c in word:
                chars.add(c)
    print 'time for c2i:', time() - t_c
    return {c: i for i, c in enumerate(chars)}


def add_pref_and_suff(data, w2i):
    """ update w2i to contain the prefixes and suffixes of the words in data-sentences """
    t_aff = time()

    pref_set, suff_set = set(), set()
    for sentence, _ in data:
        for word in sentence:
            pref_set.add(word[:3])
            suff_set.add(word[-3:])
    # prefix
    w2i[PREF_UNK] = len(w2i)
    for pref in pref_set:
        w2i[PREF_FLAG + pref] = len(w2i)
    # suffix
    w2i[SUFF_UNK] = len(w2i)
    for suff in suff_set:
        w2i[SUFF_FLAG + suff] = len(w2i)

    print 'time for affixes:', time() - t_aff


def test_data_set(filename):
    """
    :param filename: name of test file, each line is a single word, end of line is end of sentence.
    :return: list of sentences, each sentence is a list of words.
    """
    sentences = []
    with open(filename, 'r') as f:
        sentence = []
        for line in f:
            line = line.split()
            if line:
                sentence.append(line)
            else:   # empty line, end of sentence
                sentences.append(sentence)
                sentence = []
    return sentences
