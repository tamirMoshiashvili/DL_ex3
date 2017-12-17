import random
from time import time


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

    gen_file('data/train_pali', gen_palindrome_seq, iter_num=500, max_seq_size=25)
    gen_file('data/dev_pali', gen_palindrome_seq, iter_num=500, max_seq_size=35)

    print time() - t
