#!/usr/bin/python
from __future__ import print_function
from itertools import product
import os

nDims = [12]
nHidden = [72]
nPredictors = [3]
nEpochs = [200]
batch_sz = [20]
lex_relative_rate = [3, 1]
kernel_relative_rate = [3, 1]
wt_decay = [1]
lex_wt_decay = [1]
penalize_l = [2, 1]
kernel = ['gauss', 'laplace']
train_fraction = [.75]

def init_learn_rate(bs):
    return .01 * bs


def init_kernel_sz(nd, kern):
    if kern is 'gauss':
        return nd
    elif kern is 'laplace':
        return nd**.5


get_dict_str = '' \
               '{{' \
               '\n"nDims" : {},' \
               '\n"nHidden" : {},' \
               '\n"nPredictors" : {},' \
               '\n"nEpochs" : {},' \
               '\n"batch_sz" : {},' \
               '\n"lex_relative_rate" : {},' \
               '\n"kernel_relative_rate" : {},' \
               '\n"wt_decay" : {},' \
               '\n"lex_wt_decay" : {},' \
               '\n"penalize_l" : {},' \
               '\n"kernel" : "{}",' \
               '\n"train_fraction" : {},' \
               '\n"init_learn_rate" : {},' \
               '\n"init_kernel_sz" : {},' \
               '\n}}'.format

if not os.path.exists('./params'):
    os.makedirs('./params')
filename = './params/{}.prm'
ifile = 0

for params in product(nDims, nHidden, nPredictors, nEpochs, batch_sz,
                      lex_relative_rate, kernel_relative_rate, wt_decay,
                      lex_wt_decay, penalize_l, kernel, train_fraction):
    d, h, p, e, b, lrr, krr, wd, lwd, l, k, tf = params
    ilr = init_learn_rate(b)
    iks = init_kernel_sz(d, k)
    dict_str = get_dict_str(*(params + (ilr, iks)))
    with open(filename.format(ifile), 'w') as pfile:
        pfile.write(dict_str)
    ifile += 1