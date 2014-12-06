#!/usr/bin/python
from __future__ import print_function
import sys
import ast
import os
import pickle
from operator import mul

import numpy as np
import theano
import theano.tensor as T
from theano import config, shared
from theano.tensor.nnet import conv

if len(sys.argv) < 2:
    print('Usage: {} <corpus.pkl> [params.prm] [report_satus]'
          '\n\t params.prm can be "default"'
          '\n\t report_status reports the progress of testing and training '
          '\n\t\t within an epoch. False by default'.format(sys.argv[0]))
    sys.exit()

#############################     Parameters    ##############################

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

parameters = {
    'nDims': 12,
    'nPredictors': 3,
    'nHidden': 72,
    'nEpochs': 300,
    'batch_sz': 20,
    'lex_relative_rate': 2,
    'kernel_relative_rate': 2,
    'wt_decay': 1,
    'lex_wt_decay': 1,
    'penalize_l': 2,
    'kernel': 'laplace',
    'train_fraction': .75,
}
parameters['init_learn_rate'] = .01 * parameters['batch_sz']
if parameters['kernel'] is 'laplace':
	parameters['init_kernel_sz'] = parameters['nDims'] ** .5
else:
	parameters['init_kernel_sz'] = parameters['nDims']

prm_file_name = 'default.prm'
if (len(sys.argv) > 2) and (sys.argv[2].endswith('.prm')):
    prm_file_name = sys.argv[2]
    with open(prm_file_name, 'rb') as param_file:
        parameters.update(ast.literal_eval(param_file.read()))

P = Bunch(parameters)

report_progress = (len(sys.argv) > 3)

#############################  Open file & Info  #############################

inpt_file_name = sys.argv[1]
with open(inpt_file_name, 'rb') as f:
    corpus = pickle.load(f)
corpus = np.array(corpus, dtype='int32')

nCorpus = len(corpus)
nWords = corpus.max() + 1

lexicon = np.random.normal(size=(nWords, P.nDims))
for word in lexicon:
    word /= np.linalg.norm(word)

print('\nUnique Aksharas: {}\nCorpus Size: {}'.format(nWords, nCorpus))
for k, v in sorted(parameters.items(), key=lambda x: x[0]):
    print('{}: {}'.format(k, v))

# #############################  Helper Functions ##############################


def relative_rate(prm):
    if prm is lexicon:
        return P.lex_relative_rate
    if prm is kernel_sz:
        return P.kernel_relative_rate
    return 1


def share(data, dtype=config.floatX):
    return shared(np.asarray(data, dtype), borrow=True)


def borrow(sharedvar, boro=True):
    return sharedvar.get_value(borrow=boro)


def print_lex_info(lex):
    for word in lex:
        neighbour = np.mean([np.linalg.norm(word - word2) for word2 in lex])
        print('({:7.4} {:7.4} {:7.4} {:7.4}), '.format(
            word.max(), word.min(), np.linalg.norm(word), neighbour), end='\n')


def get_wts(size_w, size_b, wname='W', bname='b'):
    w_values = np.asarray(
        np.random.uniform(low=-1, high=1, size=size_w), dtype=config.floatX)
    w = shared(w_values, name=wname, borrow=True)
    b_values = np.asarray(
        np.random.uniform(low=-1, high=1, size=size_b), dtype=config.floatX)
    b = shared(b_values, name=bname, borrow=True)
    return w, b


def get_size(shared_var):
    return reduce(mul, shared_var.get_value().shape)


def penalty(param):
    if P.penalize_l == 2:
        return (param ** 2).sum()
    elif P.penalize_l == 1:
        return abs(param).sum()

##############################  The Neural Net ################################
lexicon = share(lexicon)
corpus = share(corpus, 'int32')

######################## Input Layer
start_idx = T.iscalar()
xdim = P.nPredictors + P.batch_sz - 1
input_indexen = corpus[start_idx:start_idx + xdim]  # xdim
output_indexen = corpus[start_idx + P.nPredictors:
                        start_idx + P.batch_sz + P.nPredictors]     # batch_sz
predictors = lexicon[input_indexen].dimshuffle('x', 'x', 0, 1)
                                                            # 1x1x dim x nDims

######################### Hidden Layer
sizeW1 = (P.nHidden, 1, P.nPredictors, P.nDims)
image_shape = (1, 1, xdim, P.nDims)
W1, b1 = get_wts(sizeW1, P.nHidden, 'W1', 'b1')
hidden_conv = conv.conv2d(predictors, W1, image_shape, sizeW1)
                                                    # 1 x nHidden x batch_sz x 1
hidden_conv = hidden_conv.dimshuffle(2, 1)          # batch_sz x nHidden
hidden = T.tanh(hidden_conv + b1)

### Dropout
srs = T.shared_randomstreams.RandomStreams()
mask = srs.binomial(n=1, p=.5, size=(P.batch_sz, P.nHidden))
dropped_hidden = hidden * T.cast(mask, theano.config.floatX)


######################### Output Layer
W2, b2 = get_wts((P.nHidden, P.nDims,), P.nDims, 'W2', 'b2')
got_out_vecs_tr = T.dot(dropped_hidden, W2) + b2  # batch_sz x nDims
got_out_vecs_te = T.dot(hidden / 2, W2) + b2

got_out_vecs_tr = got_out_vecs_tr.dimshuffle(0, 'x', 1)  # batch_sz x 1 x nDims
got_out_vecs_te = got_out_vecs_te.dimshuffle(0, 'x', 1)

lex1 = lexicon.dimshuffle('x', 0, 1)  # 1 x nWords x nDims
lex2 = lexicon.dimshuffle(0, 'x', 1)  # nWords x 1 x nDims

# batch_sz x nWords x nDims (--sum-->>) batch_sz x nWords = dists.shape
if P.kernel in ('gauss', 'cauchy',):
    dists = T.sum((got_out_vecs_tr - lex1) ** 2, axis=2)
    dists_te = T.sum((got_out_vecs_te - lex1) ** 2, axis=2)
elif P.kernel == 'laplace':
    dists = T.sum(abs(got_out_vecs_tr - lex1), axis=2)
    dists_te = T.sum(abs(got_out_vecs_te - lex1), axis=2)

kernel_sz = shared(np.cast[theano.config.floatX](P.init_kernel_sz))
dists /= kernel_sz
dists_te /= kernel_sz
if P.kernel in ('gauss', 'laplace'):
    probs = T.nnet.softmax(-dists)                  #  batch_sz x nWords
    probs_te = T.nnet.softmax(-dists_te)
elif P.kernel == 'cauchy':
    raw_probs = 1 / (1 + dists)
    probs = raw_probs / T.sum(raw_probs, axis=1).dimshuffle(0, 'x')
    raw_probs_te = 1 / (1 + dists_te)
    probs_te = raw_probs_te / T.sum(raw_probs_te, axis=1).dimshuffle(0, 'x')

######################### Cost, Gradient, Updates & the Like
cur_learn_rate = shared(np.cast[theano.config.floatX](0.0))

right_probs = probs[T.arange(P.batch_sz), output_indexen]  # batch_sz
right_probs_te = probs[T.arange(P.batch_sz), output_indexen]  # batch_sz
logprob = T.log(right_probs)  # batch_sz

prediction_cost = -T.mean(logprob)

wt_decayed_params = (W1, W2,)   # b1, b1
n_wts = sum([get_size(p) for p in wt_decayed_params])
wt_cost = T.sum([penalty(p) for p in wt_decayed_params]) / n_wts

lex_wt_cost = penalty(lexicon) / get_size(lexicon)
cost = prediction_cost \
       + P.wt_decay * wt_cost \
       + P.lex_wt_decay * lex_wt_cost

updates = []
for prm in (W1, b1, W2, b2, lexicon, kernel_sz):
    if prm is kernel_sz:
        prm_updt = shared(np.cast[theano.config.floatX](0.0))
    else:
        prm_updt = shared(borrow(prm) * 0., broadcastable=prm.broadcastable)

    updates.append((prm_updt, .99 * prm_updt + .01 * T.grad(cost, prm)))

    updated_prm = prm - cur_learn_rate * prm_updt * relative_rate(prm)
    updates.append((prm, updated_prm))


######################### Compile Functions
print('\nCompiling ...')
trainer = theano.function([start_idx], cost, updates=updates)
tester = theano.function([start_idx], right_probs_te.mean())
wt_costs_ = theano.function([], [wt_cost, lex_wt_cost])

def wt_costs():
    return map(float, wt_costs_())

############################# Actual Training #############################
print('Initial wt_cost {:6.4f}, lexcost {:6.4f} kernel_sz '
      ''.format(*wt_costs()), round(borrow(kernel_sz), 4))
stats = ['epoch,   cost,tr_prob,ts_prob, wtcost,lexcost,kern_sz']
print('\n\n', stats[0])

TRAIN = int(nCorpus * P.train_fraction)
best_ts_prob = 0
best_pkl_file = None
pkl_file_name = os.path.splitext(inpt_file_name)[0] + '_{:04.1f}_' +\
                os.path.splitext(os.path.basename(prm_file_name))[0] + '.pkl'

for epoch in range(P.nEpochs):
	# Set current learning rate
    cur_learn_rate.set_value(P.init_learn_rate / (1 + epoch // 5))
    
    # Train the data
    #
    tr_prob, ts_prob, cost = 0.0, 0.0, 0.0
    for i in range(0, TRAIN - P.nPredictors - P.batch_sz, P.batch_sz):
        if report_progress : print('TR{:6d}'.format(i), end='')
        cost += trainer(i)
        if report_progress : print('\b\b\b\b\b\b\b\b', end='')

    # Find test & train probabilities
    #
    for i in range(0, TRAIN - P.nPredictors - P.batch_sz, P.batch_sz):
        if report_progress : print('TS{:6d}'.format(i), end='')
        tr_prob += tester(i)
        if report_progress : print('\b\b\b\b\b\b\b\b', end='')

    for i in range(TRAIN, nCorpus - P.nPredictors - P.batch_sz, P.batch_sz):
        if report_progress : print('TS{:6d}'.format(i), end='')
        ts_prob += tester(i)
        if report_progress : print('\b\b\b\b\b\b\b\b', end='')

    tr_prob /= (TRAIN - P.nPredictors - P.batch_sz) // P.batch_sz
    ts_prob /= (nCorpus - P.nPredictors - P.batch_sz - TRAIN) // P.batch_sz
    cost /= (TRAIN - P.nPredictors - P.batch_sz) // P.batch_sz
    wt_c, lex_wt_c = wt_costs()

    # Print some stats
    #
    if ts_prob > best_ts_prob:
        tmp_best_pkl_file = pkl_file_name.format(100 * ts_prob)

    stat = ('{:5}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {}' 
            ''.format(epoch, cost, tr_prob, ts_prob, wt_c, lex_wt_c,
                     float(borrow(kernel_sz)), tmp_best_pkl_file))
    print(stat)
    stats.append(stat)

    # If performance is better, save pkl file
    #
    if ts_prob > best_ts_prob:
        if best_pkl_file:
            os.remove(best_pkl_file)
        best_ts_prob = ts_prob
        best_pkl_file = pkl_file_name.format(100 * best_ts_prob)
        wts_lex = {'W1':borrow(W1), 'W2': borrow(W2), 
        		   'b1':borrow(b1), 'b2': borrow(b2), 
        		   'lexicon': borrow(lexicon), 
        		   'params': P.__dict__, 'stats' : stats}
        with open(best_pkl_file, "wb") as f:
            pickle.dump(wts_lex, f, -1)

    print('{:5}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {}'
          ''.format(epoch, cost, tr_prob, ts_prob, wt_c, lex_wt_c,
                    float(borrow(kernel_sz)), best_pkl_file))
