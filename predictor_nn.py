#!/usr/bin/python
from __future__ import print_function
import sys
import numpy as np
import theano
import theano.tensor as T
from theano import config, shared
import pickle

config.optimizer = 'fast_compile'
nDims = 7
nPredictors = 3
nHidden = 100
rate = .01
nEpochs = 200
wt_decay = .0
area_decay = .0
var = nDims

rootndims = float(nDims) ** .5
sqrndims = nDims ** 2

# nWords = 30
# nCorpus = 1000
# nWordPerClass = 10
# corpus = np.zeros(nCorpus, dtype='int32')
# corpus[0] = 0#np.random.random_integers(nWords)
# corpus[1] = np.random.random_integers(nWords-1)
# nClasses = nWords//nWordPerClass
# for i in range(2, nCorpus):
#     newClass = ((corpus[0] + corpus[1])//2)//nWordPerClass
#     newClass += int(np.random.normal(scale=1))
#     gen = np.random.random_integers(0, nWordPerClass-1) + (newClass * nWordPerClass)
#     corpus[i] = gen % nWords
# print (corpus)

with open('blah.pkl', 'rb') as f:
    corpus = pickle.load(f)
corpus = np.array(corpus, dtype='int32')
nCorpus = len(corpus)
nWords = corpus.max()+1
print(corpus.min(), nWords, nCorpus)

lexicon = np.random.normal(size=(nWords, nDims))
for word in lexicon:
    word /= np.linalg.norm(word) / rootndims


def share(data, dtype=config.floatX):
    return shared(np.asarray(data, dtype), borrow=True)


lexicon = share(lexicon)


def print_lex():
    for word in lexicon.get_value():
        neighbour = np.mean([np.linalg.norm(word - word2) for word2 in lexicon.get_value()])
        print('({:7.4} {:7.4} {:7.4} {:7.4}), '.format(
            word.max(), word.min(), np.linalg.norm(word), neighbour), end='\n')

#print_lex()

inpt = T.ivector()
output = T.iscalar()
predictors = lexicon[inpt].flatten()


def get_wts(n_in, n_out, wname='W', bname='b'):
    w_values = np.asarray(
        np.random.uniform(low=-1, high=1, size=(n_in, n_out)), dtype=config.floatX)
    w = shared(w_values, name=wname, borrow=True)
    b_values = np.asarray(
        np.random.uniform(low=-1, high=1, size=n_out), dtype=config.floatX)
    b = shared(b_values, name=bname, borrow=True)
    return w, b


W1, b1 = get_wts(nPredictors * nDims, nHidden, 'W1', 'b1')
hidden = T.tanh(T.dot(predictors, W1) + b1)

W2, b2 = get_wts(nHidden, nDims, 'W2', 'b2')
output_vec = (T.dot(hidden, W2) + b2)

mse = T.sum((output_vec - lexicon[output]) ** 2)/var
others = T.sum((lexicon-output_vec)**2, axis=1)/var
prob = T.exp(-mse)/T.sum(T.exp(-others))

lex1 = lexicon.dimshuffle('x', 0, 1)
lex2 = lexicon.dimshuffle(0, 'x', 1)
area = T.mean((lex1 - lex2) ** 2)

wt_cost = 0
for param in (W1, b1, W2, b2):
    wt_cost += (param ** 2).sum()

cost = -T.log(prob) #+ wt_decay * wt_cost #- area_decay * area

updates = []
for param in (W1, b1, W2, b2):
    update = param - rate * T.grad(cost, param)
    updates.append((param, update))

lex_update = lexicon - rate * T.grad(cost, lexicon)
lex_norms = T.sqrt(T.sum(T.sqr(lexicon), axis=1, keepdims=True))
lex_update = rootndims * lex_update / lex_norms
updates.append((lexicon, lex_update))

print('Compiling them eggs...')
trainer = theano.function([inpt, output], cost, updates=updates)
tester = theano.function([inpt, output], prob)
areaer = theano.function([], [area, wt_cost])

print('\n\nepoch, cost , prob , area , wt_cost ')

for epoch in range(nEpochs):
    prob, cost = 0.0, 0.0
    for i in range(nCorpus - nPredictors):
        print('TR{:6d}'.format(i), end='')
        cost += trainer(corpus[i:i + nPredictors], corpus[i + nPredictors])
        print('\b\b\b\b\b\b\b\b', end='')

    for i in range(nCorpus - nPredictors):
        print('TS{:6d}'.format(i), end='')
        prob += tester(corpus[i:i + nPredictors], corpus[i + nPredictors])
        print('\b\b\b\b\b\b\b\b', end='')

    prob /= nCorpus - nPredictors
    cost /= nCorpus - nPredictors
    ar, wt_c = areaer()
    print('{:5}, {:6.4f}, {:6.4f}, {:6.4f}, {:6.4f}'.format(epoch, cost, prob, float(ar), float(wt_c)))

print_lex()

########################### Do Some PCA ##########################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=2)
pcs = pca.fit_transform(lexicon.get_value())
print('PC shape:', pcs.shape)
plt.scatter(pcs[:,0], pcs[:,1], c=np.arange(nWords)//nWordPerClass, s=100)

for label, x, y in zip(range(nWords), pcs[:, 0], pcs[:, 1]):
    plt.annotate(
        str(label),
        xy = (x, y),
        #xytext = (-20, 20),
        #textcoords = 'offset points', ha = 'right', va = 'bottom',
        #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
    )


plt.show()

