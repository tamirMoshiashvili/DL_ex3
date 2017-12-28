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
