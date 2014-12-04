#!/usr/bin/python
from __future__ import print_function
import sys
import pickle
from operator import mul

import numpy as np
import theano
import theano.tensor as T
from theano import config, shared
from theano.tensor.nnet   import conv

#############################     Parameters    ##############################

nDims = 9
nPredictors = 3
nHidden = 100
nEpochs = 300
batch_sz = 20
init_learn_rate = .01 * batch_sz
lex_relative_rate = 20
kernel_relative_rate = 10
wt_decay = 1
lex_wt_decay = 1
penalize_l = 2
kernel = 'gauss'

init_kernel_sz = nDims
# Use nDims for gauss

#############################  Open file & Info  #############################

with open(sys.argv[1], 'rb') as f:
    corpus = pickle.load(f)
corpus = np.array(corpus, dtype='int32')

nCorpus = len(corpus)
nWords = corpus.max() + 1

lexicon = np.random.normal(size=(nWords, nDims))
for word in lexicon:
    word /= np.linalg.norm(word)

print('\nUnique Aksharas: {}'
      '\nCorpus Size: {}'
      '\nPredictors: {}'
      '\nHidden: {}'
      '\nDimensions: {}'
      '\nWeight Decay: {} (lex: {})'
      '\nEpochs: {}'
      '\nBatch Size: {}'
      '\nLearning Rate: {} (lex: *{})'.format(
        nWords, nCorpus, nPredictors, nHidden, nDims, 
        wt_decay, lex_wt_decay, nEpochs, batch_sz, init_learn_rate, lex_relative_rate))

##############################  Helper Functions ##############################

def share(data, dtype=config.floatX):
    return shared(np.asarray(data, dtype), borrow=True)

def borrow(sharedvar, borrow=True):
    return sharedvar.get_value(borrow=borrow)

def print_lex_info(lex):
    for word in lex:
        neighbour = np.mean([np.linalg.norm(word - word2) for word2 in lex])
        print('({:7.4} {:7.4} {:7.4} {:7.4}), '.format(
            word.max(), word.min(), np.linalg.norm(word), neighbour), end='\n')

def get_wts(sizeW, sizeB, wname='W', bname='b'):
    w_values = np.asarray(
        np.random.uniform(low=-1, high=1, size=sizeW), dtype=config.floatX)
    w = shared(w_values, name=wname, borrow=True)
    b_values = np.asarray(
        np.random.uniform(low=-1, high=1, size=sizeB), dtype=config.floatX)
    b = shared(b_values, name=bname, borrow=True)
    return w, b

def get_size(shared_var):
    return reduce(mul, shared_var.get_value().shape)

def penalty(param):
    if penalize_l == 2:
        return (param**2).sum()
    elif penalize_l == 1:
        return abs(param).sum()

##############################  The Neural Net ################################

lexicon = share(lexicon)
corpus = share(corpus, 'int32')

######################## Input Layer

start_idx = T.iscalar()
xdim = nPredictors + batch_sz - 1
input_indexen = corpus[start_idx:start_idx+xdim]          # xdim
output_indexen = corpus[start_idx+nPredictors:start_idx+batch_sz+nPredictors] 
                                                                # batch_sz
predictors = lexicon[input_indexen].dimshuffle('x','x', 0, 1) # 1x1x dim x nDims


######################### Hidden Layer

sizeW1 = (nHidden, 1, nPredictors, nDims)
image_shape  = (1, 1, xdim, nDims)
W1, b1 = get_wts(sizeW1, nHidden, 'W1', 'b1')
hidden_conv = conv.conv2d(predictors, W1, image_shape, sizeW1)  
                                                    # 1 x nHidden x batch_sz x 1
hidden_conv = hidden_conv.dimshuffle(2, 1)          # batch_sz x nHidden
hidden = T.tanh(hidden_conv + b1)


######################### Output Layer

W2, b2 = get_wts((nHidden, nDims,), nDims, 'W2', 'b2')
got_out_vecs  = T.dot(hidden, W2) + b2                 # batch_sz x nDims
got_out_vecs = got_out_vecs.dimshuffle(0, 'x', 1)      # batch_sz x 1 x nDims

lex1 = lexicon.dimshuffle('x', 0,  1)                  # 1 x nWords x nDims
lex2 = lexicon.dimshuffle(0, 'x', 1)                   # nWords x 1 x nDims

# batch_sz x nWords x nDims (--sum-->>) batch_sz x nWords = dists.shape
if kernel in ('gauss', 'cauchy',):
    dists = T.sum((got_out_vecs - lex1)**2, axis=2)
elif kernel == 'laplace':
    dists = T.sum(abs(got_out_vecs - lex1), axis=2)

kernel_sz = theano.shared(np.cast[theano.config.floatX](init_kernel_sz))
dists /= kernel_sz

    # probs.shape =  batch_sz x nWords
if kernel in ('gauss', 'laplace'):
    probs = T.nnet.softmax(-dists)
elif kernel == 'cauchy':
    raw_probs = 1/(1 + dists)
    probs = raw_probs / T.sum(raw_probs, axis=1).dimshuffle(0, 'x')

######################### Cost, Gradient, Updates & the Like
cur_learn_rate = theano.shared(np.cast[theano.config.floatX](0.0))

right_probs = probs[T.arange(batch_sz), output_indexen] # batch_sz
logprob = T.log(right_probs)                            # batch_sz

prediction_cost = -T.mean(logprob)

params = (W1,W2,) #b1, b2,)
n_wts = sum([get_size(p) for p in params])
wt_cost = T.sum([penalty(p) for p in params]) / n_wts

lex_wt_cost = penalty(lexicon) / get_size(lexicon)
cost = prediction_cost \
        + wt_decay * wt_cost \
        + lex_wt_decay * lex_wt_cost #- T.log(kernel_sz) * 1e-3

updates = []

def relative_rate(prm):
    if prm is lexicon:
        return lex_relative_rate
    if prm is kernel_sz:
        return kernel_relative_rate
    return 1

for prm in (W1, b1, W2, b2, lexicon, kernel_sz):
    if prm is kernel_sz:
        prm_updt = theano.shared(np.cast[theano.config.floatX](0.0))
    else:
        prm_updt = theano.shared(borrow(prm)*0., broadcastable=prm.broadcastable)

    updates.append((prm_updt, .99*prm_updt + .01*T.grad(cost, prm)))

    updated_prm = prm - cur_learn_rate * prm_updt * relative_rate(prm)
    updates.append((prm, updated_prm))

volume = T.mean((lex1 - lex2) ** 2)


######################### Compile Functions
print('\nCompiling ...')
trainer = theano.function([start_idx], cost, updates=updates)
tester = theano.function([start_idx], right_probs.mean())
volumer = theano.function([], [volume, wt_cost, lex_wt_cost])


############################# Actual Training #############################
print('Initial volume {:6.4f}, wt_cost {:6.4f}, lexcost {:6.4f} kernel_sz'
      ''.format(*map(float, volumer())), borrow(kernel_sz))
print('\n\nepoch,   cost,tr_prob,ts_prob, volume, wtcost,lexcost,kern_sz')

TRAIN = int(nCorpus * .75)
for epoch in range(nEpochs):
    cur_learn_rate.set_value(init_learn_rate /(1 + epoch//5))
    tr_prob, ts_prob, cost = 0.0, 0.0, 0.0
    for i in range(0, TRAIN - nPredictors - batch_sz, batch_sz):
        print('TR{:6d}'.format(i), end='')
        cost += trainer(i)
        print('\b\b\b\b\b\b\b\b', end='')

    for i in range(0, TRAIN - nPredictors - batch_sz, batch_sz):
        print('TS{:6d}'.format(i), end='')
        tr_prob += tester(i)
        print('\b\b\b\b\b\b\b\b', end='')

    for i in range(TRAIN, nCorpus - nPredictors - batch_sz, batch_sz):
        print('TS{:6d}'.format(i), end='')
        ts_prob += tester(i)
        print('\b\b\b\b\b\b\b\b', end='')

    tr_prob /= (TRAIN - nPredictors - batch_sz)//batch_sz
    ts_prob /= (nCorpus - nPredictors - batch_sz - TRAIN)//batch_sz
    cost /= (TRAIN - nPredictors - batch_sz)//batch_sz
    vol, wt_c, lex_wt_c = map(float, volumer())
    print('{:5}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, '
          '{:6.4f}'.format(epoch, cost, tr_prob, ts_prob, vol, wt_c, lex_wt_c,
                           float(borrow(kernel_sz))))
