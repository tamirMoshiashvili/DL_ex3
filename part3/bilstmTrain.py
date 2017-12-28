import dynet as dy

"""
 NOTE - args in command line need to be the following:
 repr trainFile modelFile [options]
 - repr - one of a, b, c, d
 - trainFile path to the train file
 - modelFile path to the model file to be created
 - [options]:
 -  mode flag:
 -      '-pos' for POS tagging
 -      '-ner' for NER tagging
 -      everything else will fail and set mode to 'unknown'
 -  save flag:
 -      add '-save' to save the model after training
 -  dev flag:
 -      add '-dev-path param' where param is the path to the dev file
"""
from time import time

import sys
import utils
from part3.BiLstmModel import BiLstmModel

POS_FLAG = '-pos'
NER_FLAG = '-ner'
SAVE_FLAG = '-save'
DEV_FLAG = '-dev-path'


def resolve_mode(arr):
    if POS_FLAG in arr:
        return 'pos'
    elif NER_FLAG in arr:
        return 'ner'
    return 'unknown'


if __name__ == '__main__':
    t0 = time()
    print 'start'

    args = sys.argv[1:]
    representation = sys.argv[0]
    train_file_path = sys.argv[1]
    model_file_path = sys.argv[2]

    mode = resolve_mode(args)
    save_model = SAVE_FLAG in args
    dev_file_path = args[args.index(DEV_FLAG) + 1]

    train_data_set = utils.make_data_set(train_file_path)
    dev_data_set = utils.make_data_set(dev_file_path)
    w_set, t_set = utils.extract_word_and_tag_sets_from(train_data_set)
    w_set.add(utils.UNK)
    w_to_i = {w: i for i, w in enumerate(w_set)}
    l_to_i = {l: i for i, l in enumerate(t_set)}

    print 'time for loading and parsing the files:', time() - t0
    t0 = time()

    pc = dy.ParameterCollection()
    net = BiLstmModel(pc, w_to_i, l_to_i)
    net.train_on(train_data_set, dev_data_set, to_save=save_model, model_name=mode + '_' + representation)

    print 'time to train:', time() - t0
