from __future__ import print_function
import numpy as np
import sys
import os
import ast
import pickle

# ############################### The Net ##############################


def softmax(w):
    maxx = np.amax(w)
    e = np.exp(w - maxx)
    dist = e / np.sum(e)
    return dist


class Net():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.nHidden, one, self.nPredictors, self.nDims = self.W1.shape
        self.nWords = self.lexicon.shape[0]
        if not hasattr(self, 'kernel_sz'):
            self.kernel_sz = 36.0

    def run(self, indexen):
        assert len(indexen) == self.nPredictors

        inpt = self.lexicon[indexen]
        hidden = np.tanh(np.tensordot(self.W1, inpt, axes=([2, 3], [0, 1])).T +
                         self.b1)
        out = np.dot(hidden/2, self.W2) + self.b2   # /2 for dropout

        lex1 = self.lexicon.reshape(1, self.nWords, self.nDims)

        if self.params['kernel'] in ('gauss', 'cauchy',):
            dists = np.sum((out - lex1) ** 2, axis=2)
        elif self.params['kernel'] == 'laplace':
            dists = np.sum(abs(out - lex1), axis=2)


        dists /= self.kernel_sz
        if self.params['kernel'] in ('gauss', 'laplace'):
            probs = softmax(-dists)  #  batch_sz x nWords

        elif self.params['kernel'] == 'cauchy':
            raw_probs = 1 / (1 + dists)
            probs = raw_probs / np.sum(raw_probs, axis=1)

        return probs.flatten()

    def __str__(self):
        ret = 'Params: '
        for p in self.params:
            ret += '\n\t"{}": {}'.format(p, self.params[p])
        ret += '\n{}: {}'.format('W1', self.W1.shape)
        ret += '\n{}: {}'.format('b1', self.b1.shape)
        ret += '\n{}: {}'.format('W2', self.W2.shape)
        ret += '\n{}: {}'.format('b2', self.b2.shape)
        ret += '\n{}: {}'.format('lex', self.lexicon.shape)
        ret += '\n{}: {}'.format('kernel_sz', self.kernel_sz)
        return ret


def sample_from_distbn(probabilities):
    bins = np.add.accumulate(probabilities)
    return np.digitize(np.random.random_sample(1), bins)

################################ Parse Arguments ##############################

if len(sys.argv) < 3:
    print('Usage\n'
          '{} <nnet.pkl> <akshara list.list>'
          ''.format(sys.argv[0]))
    sys.exit()

nnet_file = sys.argv[1]
aksh_file = sys.argv[2]

with open(nnet_file, 'rb') as nf:
    nnet = Net(**pickle.load(nf))

with open(aksh_file, 'r') as af:
    aksharas = ast.literal_eval(af.read())

############################# Generate Text ##################################

print(nnet)

corpus = [5, 1443, 1197]
for i in range(1000):
    distbn = nnet.run(corpus[i: i + 3])
    nxt = sample_from_distbn(distbn)
    corpus.append(nxt[0])

sans_corpus = ''.join([aksharas[c][0] for c in corpus])
sans_corpus = sans_corpus.replace('_', ' ').replace('$', '\n')
print(sans_corpus)
print(corpus)
