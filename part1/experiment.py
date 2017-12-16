from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

VOCAB = map(str, range(1, 10)) + ['a', 'b', 'c', 'd']
LABELS = ['pos', 'neg']

n_letters = len(VOCAB)
n_labels = len(LABELS)
n_hidden = 20

w2i = {w: i for i, w in enumerate(VOCAB)}  # map letter to index
l2i = {l: i for i, l in enumerate(LABELS)}  # map label to index


def letter_to_tensor(letter):
    """ turn a letter into a <1 x n_letters> Tensor """
    tensor = torch.zeros(1, n_letters)
    tensor[0][w2i[letter]] = 1
    return tensor


def line_to_tensor(line):
    """
    Turn a line into a <line_length x 1 x n_letters>,
        or an array of one-hot letter vectors
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        tensor[i][0][w2i[letter]] = 1
    return tensor


def make_data_set(data):
    """
    data - list of lines, each line is 'word label'.
    :return list of tuples, each tuple is (word, label) where each is a tensor.
    """
    for i, line in enumerate(data):
        word, label = line.split()
        data[i] = (line_to_tensor(word), torch.LongTensor([l2i[label]]))
    return data


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        self.hidden_size = hidden_size
        self.lr = 0.005
        self.iter_number = 10

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, inputs, hidden):
        combined = torch.cat((inputs, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def train(self, train_data, test_data):
        train_data = make_data_set(train_data)
        test_data = make_data_set(test_data)
        data_size = len(train_data)

        criterion = nn.CrossEntropyLoss()  # include the softmax
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(self.iter_number):
            total_loss = 0.0
            good, bad = 0.0, 0.0
            curr_t = time()

            for inputs, label in train_data:
                inputs, label = Variable(inputs), Variable(label)
                hidden = self.init_hidden()

                optimizer.zero_grad()

                for i in range(inputs.size()[0]):
                    outputs, hidden = self(inputs[i], hidden)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.data[0]

                # extract the predicted label and update good and bad
                _, predicted = torch.max(outputs.data, 1)
                bad += (predicted != label.data).sum()
                good += (predicted == label.data).sum()

            print str(epoch) + ' - loss: ' + str(total_loss / data_size) + ', time: ' + str(
                time() - curr_t) + ', accuracy: ' + str(good / (good + bad))

            print '\ttest res:', self.predict_and_check_accuracy(test_data, criterion)

    def predict_and_check_accuracy(self, data_set, criterion):
        """
        :param data_set: list of windows, each is a list of 5 words.
        :param criterion: loss function.
        :return: string that represent performance, which is 'loss,accuracy'
        """
        good, bad = 0.0, 0.0
        total_loss = 0.0

        data_size = len(data_set)
        for inputs, label in data_set:
            # predict
            inputs, label = Variable(inputs), Variable(label)
            hidden = self.init_hidden()

            for i in range(inputs.size()[0]):
                outputs, hidden = self(inputs[i], hidden)
            total_loss += criterion(outputs, label).data[0]

            # extract the predicted label and update good and bad
            predicted = category_from_output(outputs)  # todo delete
            _, predicted = torch.max(outputs.data, 1)
            bad += (predicted != label.data).sum()
            good += (predicted == label.data).sum()

        # loss, acc
        return 'loss: ' + str(total_loss / data_size) + ', acc: ' + str(good / (good + bad))


def category_from_output(output):
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return LABELS[category_i], category_i


def read_data_file(filename):
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    return lines


if __name__ == '__main__':
    print 'start'

    t = time()

    train = read_data_file('train_set')
    test = read_data_file('test_set')

    rnn = Net(n_letters, n_hidden, n_labels)
    rnn.train(train, test)

    print 'time:', time() - t
