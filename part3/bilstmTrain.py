import dynet as dy

"""
 NOTE - args in command line need to be the following:
 repr trainFile modelFile [options]
 - repr - one of a, b, c, d
 - trainFile path to the train file
 - modelFile path to the model file to be created
 - [options]:
 -  save flag:
 -      add '-save' to save the model after training
 -  dev flag:
 -      add '-dev-path param' where param is the path to the dev file

 example: a ../pos/train pos_a -pos -save -dev-path ../pos/dev
"""
from time import time

import sys
import utils
from Representation import *
from part3.BiLstmModel import BiLstmModel

SAVE_FLAG = '-save'
DEV_FLAG = '-dev-path'

if __name__ == '__main__':
    """
        args:       repr train_file        model_file -dev-path dev_path        [-save]
        example:    a    ../data/pos/train model_pos  -dev-path ../data/pos/dev -save
    """

    t0 = time()
    print 'start'

    args = sys.argv[1:]
    representation = args[0]
    train_file_path = args[1]
    model_file_path = args[2]

    save_model = SAVE_FLAG in args
    dev_file_path = args[args.index(DEV_FLAG) + 1]  # must include dev-file-path

    train_data_set = utils.make_data_set(train_file_path)
    dev_data_set = utils.make_data_set(dev_file_path)
    w_set, t_set = utils.extract_word_and_tag_sets_from(train_data_set)
    w_set.add(utils.UNK)
    w2i = {w: i for i, w in enumerate(w_set)}
    l2i = {l: i for i, l in enumerate(t_set)}

    print 'time for loading and parsing the files:', time() - t0
    t0 = time()

    pc = dy.ParameterCollection()

    if representation == 'a':
        args = {S_MODEL: pc, S_W2I: w2i, S_EMB_DIM: utils.DEF_EMB_DIM}
    elif representation == 'b':
        del w2i
        c2i = utils.create_c2i(train_data_set)
        args = {S_MODEL: pc, S_C2I: w2i, S_EMB_DIM: utils.DEF_EMB_DIM,
                S_LSTM_DIM: 2 * utils.DEF_LSTM_IN, S_LAYERS: utils.DEF_LAYERS}
    elif representation == 'c':  # TODO change args to dict
        utils.add_pref_and_suff(train_data_set, w2i)
        args = (pc, w2i, utils.DEF_EMB_DIM)
    representor = resolve_repr(representation, args)

    net = BiLstmModel(pc, representor, l2i)
    net.train_on(train_data_set, dev_data_set,
                 to_save=save_model, model_name=model_file_path + '_' + representation)

    print 'time to train:', time() - t0
