"""
 NOTE - args in command line need to be the following:
 repr modelFile inputFile [options]
 - repr - one of a, b, c, d
 - modelFile path to the model file to be created
 - inputFile path to the input file, to predict on
 - [options]:
 -  output_file_path path of output-file to be created

 example: a data_ner/repr_a/model_ner ../data/ner/test test4.ner
"""

from time import time

import sys

from part3 import utils
from part3.BiLstmModel import BiLstmModel

if __name__ == '__main__':
    t0 = time()
    print 'start'

    args = sys.argv[1:]
    representation = args[0]
    model_file_path = args[1]
    input_file_path = args[2]
    output_file_path = args[3]

    net = BiLstmModel.load_model(model_file_path, representation)
    test = utils.test_data_set(input_file_path)

    net.test_on(test, output_file_path)

    print 'time to run all:', time() - t0
