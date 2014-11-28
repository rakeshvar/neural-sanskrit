#!/usr/bin/python
from __future__ import print_function
import sys
import numpy as np
import theano
import theano.tensor as T
from theano import config, shared
import pickle
from   theano.tensor.nnet   import conv

#############################     Parameters    ##############################

nDims = 10
nPredictors = 3
nHidden = 50
nEpochs = 20
batch_sz = 20
rate = .01 * batch_sz
lex_update_speed = 2
wt_decay = .0
lex_decay_speed = 10

rootndims = float(nDims) ** .5
sqrndims = nDims ** 2

#############################  Open file & Info  #############################

with open(sys.argv[1], 'rb') as f:
    corpus = pickle.load(f)
corpus = np.array(corpus, dtype='int32')

nCorpus = len(corpus)
nWords = corpus.max() + 1

lexicon = np.random.normal(size=(nWords, nDims))
for word in lexicon:
    word /= np.linalg.norm(word) / rootndims

print('\nUnique Aksharas: {}'
      '\nCorpus Size: {}'
      '\nPredictors: {}'
      '\nHidden: {}'
      '\nDimensions: {}'
      '\nWeight Decay: {} (lex: *{})'
      '\nEpochs: {}'
      '\nBatch Size: {}'
      '\nLearning Rate: {} (lex: *{})'.format(
        nWords, nCorpus, nPredictors, nHidden, nDims, 
        wt_decay, lex_decay_speed, nEpochs, batch_sz, rate, lex_update_speed))

##############################  Helper Functions ##############################

def share(data, dtype=config.floatX):
    return shared(np.asarray(data, dtype), borrow=True)

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

dists   = T.sum((got_out_vecs - lex1)**2, axis=2)       
                  # batch_sz x nWords x nDims (--sum-->>) batch_sz x nWords
probs   = T.nnet.softmax(dists/nDims)                   # batch_sz x nWords


######################### Cost, Gradient, Updates & the Like

right_probs = probs[T.arange(batch_sz), output_indexen] # batch_sz
logprob = T.log(right_probs)                            # batch_sz

prediction_cost = -T.mean(logprob)
weight_cost = T.sum([(param ** 2).mean() 
                        for param in (W1, b1, W2, b2, lexicon)])
cost = prediction_cost + wt_decay * weight_cost

updates = []

for param in (W1, b1, W2, b2):
    update = param - rate * T.grad(cost, param)
    updates.append((param, update))

lex_update = lexicon - lex_update_speed * rate * T.grad(cost, lexicon)
updates.append((lexicon, lex_update))

volume = T.mean((lex1 - lex2) ** 2)


######################### Compile Functions
print('\nCompiling ...')
trainer = theano.function([start_idx], cost, updates=updates)
tester = theano.function([start_idx], right_probs.mean())
volumer = theano.function([], [volume, weight_cost])


############################# Actual Training #############################
print('\n\nepoch,   cost,tr_prob,ts_prob, volume, weight_cost ')

TRAIN = int(nCorpus * .75)
for epoch in range(nEpochs):
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
    ar, wt_c = volumer()
    print('{:5}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}'.format(
                    epoch, cost, tr_prob, ts_prob, float(ar), float(wt_c)))


############################### Do Some PCA ################################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=4)
pcs = pca.fit_transform(lexicon.get_value())

def plot_scatter_label(x, y):
    plt.scatter(x, y)

    for label, i, j in zip(range(nWords), x, y):
        plt.annotate(
            str(label),
            xy=(i, j),
            #xytext = (-20, 20),
            #textcoords = 'offset points', ha = 'right', va = 'bottom',
            #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
        )

plot_scatter_label(pcs[:, 0], pcs[:, 1])
plt.show()
plot_scatter_label(pcs[:, 1], pcs[:, 2])
plt.show()
plot_scatter_label(pcs[:, 2], pcs[:, 3])
plt.show()
