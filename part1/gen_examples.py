import random
from StringIO import StringIO

MAX_SEQ_SIZE = 25
SAMPLE_SIZE = 500
TRAIN_SIZE = 15000
TEST_SIZE = 15000


def generate_line(mode):
    """ mode - 'pos' for positive example, 'neg' for negative example """

    line = StringIO()
    digits = map(str, range(1, 10))

    if mode == 'pos':   # positive line
        letters = ['a', 'b', 'c', 'd']
    else:   # negative line
        letters = ['a', 'c', 'b', 'd']

    for letter in letters:
        # generate [1-9]+
        for _ in range(random.randint(1, MAX_SEQ_SIZE)):
            line.write(random.choice(digits))
        # generate letter+
        line.write(random.randint(1, MAX_SEQ_SIZE) * letter)
    # generate [1-9]+
    for _ in range(random.randint(1, MAX_SEQ_SIZE)):
        line.write(random.choice(digits))

    return line.getvalue()


def write_examples_to_file(filename, example_list):
    """ example_list - each item is string """
    f = open(filename, 'w')
    f.write('\n'.join(example_list))
    f.close()


def generate_sample_and_save(mode, filename):
    # generate samples
    example_list = []
    for _ in range(SAMPLE_SIZE):
        example_list.append(generate_line(mode))
    # save in file
    write_examples_to_file(filename, example_list)


def generate_set_and_save(filename, set_size):
    modes = ['pos', 'neg']
    lines = []
    for _ in range(set_size):
        lines.append(generate_line(random.choice(modes)))
    # save in file
    write_examples_to_file(filename, lines)


if __name__ == '__main__':
    # generate examples
    generate_sample_and_save('pos', 'pos_examples')
    generate_sample_and_save('neg', 'neg_examples')

    # generate sets
    generate_set_and_save('train_set', TRAIN_SIZE)
    generate_set_and_save('test_set', TEST_SIZE)
