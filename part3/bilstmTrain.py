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
 a ../pos/train pos_a -pos -save -dev-path ../pos/dev
"""
from time import time

import sys
import utils
from part3.BiLstmModel import BiLstmModel
import Representation

SAVE_FLAG = '-save'
DEV_FLAG = '-dev-path'

if __name__ == '__main__':
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
    args = (pc, w2i, utils.DEF_EMB_DIM)
    representor = Representation.resolve_repr(representation, args)
    net = BiLstmModel(pc, representor, w2i, l2i)
    net.train_on(train_data_set, dev_data_set,
                 to_save=save_model, model_name=model_file_path + '_' + representation)

    print 'time to train:', time() - t0
