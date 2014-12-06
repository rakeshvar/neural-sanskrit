#!/usr/bin/python
from __future__ import print_function
import importlib, ast
import sys
import pickle
from collections import Counter, defaultdict

if len(sys.argv) < 2:
    print('Usage: {} <corpus.pkl> [train_fraction=.5] [n=3]'
          '\n\t Uses corpus.pkl and corpus.list as generated by corpus_builder'
          ''.format(sys.argv[0]))
    sys.exit()

try:
    train_fraction = float(sys.argv[2])
except:
    train_fraction = .5

try:
    n = int(sys.argv[3])
except:
    n = 3

ngrams = defaultdict(Counter)
tup_cnt = Counter()

with open(sys.argv[1][:-4] + '.list') as f:
    aksharas = ast.literal_eval(f.read())

print('Loading...')
with open(sys.argv[1], 'rb') as f:
    corpus = pickle.load(f)

nCorpus = len(corpus)
nWords = max(corpus) + 1
nTrain = int(nCorpus * train_fraction)

print('Unique Aksharas: {}'
      '\nCorpus Size: {}'
      '\nPredictors: {}'
      '\nTrain Fraction: {}'.format(nWords, nCorpus, n, train_fraction))


print('Training...')
for i in range(nTrain - n):
    t = tuple(corpus[i:i + n])
    ngrams[t][corpus[i + n]] += 1
    tup_cnt[t] += 1


def printd(dd, uu):
    for tt, cc in sorted(uu.items(), key=lambda x: x[1]):
        if cc < 10: continue
        print(''.join([aksharas[l] for l in tt]), cc)
        for kk, vv in sorted(dd[tt].items(), key=lambda x:x[1])[-10:]:
            print('\t{} {}'.format(aksharas[kk], vv))


#printd(ngrams, tup_cnt)


def get_prob(tup, follower, laplace):
    if not tup in tup_cnt:  # Unseen tuple
        return 1. / nWords
    else:
        return float(ngrams[tup][follower] + laplace) /\
               (tup_cnt[tup] + laplace * nWords)


def find_avg_probability(begin, end, laplace):
    sumprob = 0
    for ii in range(begin, end - n):
        tt = tuple(corpus[ii:ii + n])
        sumprob += get_prob(tt, corpus[ii + n], laplace)
    return sumprob / (end - n - begin)


print('Testing...')
print('Smoothing, train probability, test probability')
for l in (0, 1):
    lap = float(l)
    tr_prob = find_avg_probability(0, nTrain, lap)
    ts_prob = find_avg_probability(nTrain, nCorpus, lap)
    print('{:8.5} {:8.5} {:8.5}'.format(lap, tr_prob, ts_prob))