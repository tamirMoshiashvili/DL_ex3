"""
 NOTE - args in command line need to be the following:
 repr modelFile inputFile [options]
 - repr - one of a, b, c, d
 - modelFile path to the model file to be created
 - inputFile path to the input file, to predict on
 - [options]: TODO

 example: a data_ner/repr_a/model_ner dummy
"""

from time import time

import sys

from part3.BiLstmModel import BiLstmModel

if __name__ == '__main__':
    t0 = time()
    print 'start'

    args = sys.argv[1:]
    representation = args[0]
    model_file_path = args[1]
    input_file_path = args[2]

    net = BiLstmModel.load_model(model_file_path, representation)

    print 'time to run all:', time() - t0
