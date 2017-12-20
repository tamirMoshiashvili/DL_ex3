import random
from time import time

from StringIO import StringIO

import numpy as np


def gen_even_seq(is_even, max_seq_size):
    seq = 'a' * (2 * random.randint(10, max_seq_size))
    if is_even == '1':
        return seq
    return seq + 'a'


def gen_palindrome_seq(is_palindrome, max_seq_size):
    alphabet = ['a', 'b']
    seq = []
    for _ in range(random.randint(1, max_seq_size)):
        seq.append(random.choice(alphabet))
    seq.append(random.choice([''] + alphabet))
    seq += seq[::-1]

    if is_palindrome == '1':
        return ''.join(seq)
    else:
        seq_size = len(seq)
        for _ in range(random.randint(2, seq_size)):
            i = random.randint(0, 100) % seq_size
            c = seq[i]
            if c == 'a':
                seq[i] = 'b'
            else:
                seq[i] = 'a'
        return ''.join(seq)


def gen_pow_2(is_pow_2, max_seq_size):
    seq = StringIO()
    ops = ['a', 'b']

    for _ in range(random.randint(1, 10)):
        c = random.choice(ops)
        size = np.power(2, random.randint(4, max_seq_size))
        seq.write(c * size)
        if is_pow_2 == '0':
            seq.write(c * random.randint(1, int(size / 2)))

    return seq.getvalue()


def gen_an_bn(is_true, max_seq_size):
    seq = StringIO()
    size = random.randint(int(max_seq_size / 2), max_seq_size)
    seq.write('a' * size)

    if is_true == '0':
        chose = random.choice([True, False])
        if chose:
            seq.write('a' * random.randint(1, size))
    seq.write('b' * size)

    if is_true == '0' and (not chose):
        seq.write('b' * random.randint(1, size))
    return seq.getvalue()


def gen_divide_by_3(is_true, max_seq_size):
    ops = [str(i) for i in range(10)]
    num = StringIO()

    num.write(random.choice(ops[1:]))
    for _ in range(random.randint(1, max_seq_size)):
        num.write(random.choice(ops))
    if is_true == '0':
        while int(num.getvalue()) % 3 == 0:
            num.write(random.choice(ops))
    else:
        while int(num.getvalue()) % 3 != 0:
            num.write(random.choice(ops))

    return num.getvalue()


def gen_credit_card_num(is_true, max_addition_size):
    ops = [str(i) for i in range(10)]
    seq = StringIO()

    for _ in range(3):
        for _2 in range(4):
            seq.write(random.choice(ops))
        if is_true == '0':
            for _3 in range(random.randint(0, max_addition_size)):
                seq.write(random.choice(ops))
        seq.write('-')
    for _2 in range(4):
        seq.write(random.choice(ops))
    if is_true == '0':
        for _3 in range(random.randint(1, max_addition_size)):
            seq.write(random.choice(ops))
    return seq.getvalue()


def gen_divide_3_bin(is_true, max_seq_size):
    ops = ['0', '1']
    num = StringIO()

    for _ in range(random.randint(1, max_seq_size)):
        num.write(random.choice(ops))
    if is_true == '0':
        while int(num.getvalue(), 2) % 3 == 0:
            num.write(random.choice(ops))
    else:
        while int(num.getvalue(), 2) % 3 != 0:
            num.write(random.choice(ops))

    return num.getvalue()


def gen_file(filename, gen_func, iter_num=1500, max_seq_size=15):
    examples = []
    ops = ['0', '1']
    for _ in range(iter_num):
        label = random.choice(ops)
        examples.append(gen_func(label, max_seq_size) + ' ' + str(label))

    f = open(filename, 'w')
    f.write('\n'.join(examples))
    f.close()


if __name__ == '__main__':
    print 'start'
    t = time()

    gen_file('data/train_div3_bin', gen_divide_3_bin, iter_num=3000, max_seq_size=15)
    gen_file('data/dev_div3_bin', gen_divide_3_bin, iter_num=1000, max_seq_size=15)

    print time() - t
