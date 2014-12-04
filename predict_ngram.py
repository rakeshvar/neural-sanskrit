from __future__ import print_function
import importlib, ast
import sys
import pickle
from collections import Counter, defaultdict

n = 3
d = defaultdict(Counter)
u = Counter()

with open(sys.argv[1][:-4] + '.list') as f:
    aksharas = ast.literal_eval(f.read())
#aksharas = importlib.import_module(sys.argv[1][:-4]).codes

print('Loading...')
with open(sys.argv[1], 'rb') as f:
    corpus = pickle.load(f)

nCorpus = len(corpus)
nWords = max(corpus) + 1

print('Unique Aksharas: {}'
      '\nCorpus Size: {}'
      '\nPredictors: {}'.format(nWords, nCorpus, n))

nTrain = int(nCorpus * .75)

print('Training...')
for i in range(nTrain - n):
    t = tuple(corpus[i:i + n])
    d[t][corpus[i + n]] += 1
    u[t] += 1


def printd(dd, uu):
    for tt, cc in sorted(uu.items(), key=lambda x: x[1]):
        if uu[tt] < 10: continue
        print(''.join([aksharas[l] for l in tt]), cc)
        for kk, vv in sorted(dd[tt].items(), key=lambda x:x[1])[-10:]:
            print('\t{} {}'.format(aksharas[kk], vv))


printd(d, u)


def get_prob(tup, follower, laplace):
    if not tup in u:  # Unseen tuple
        return 1. / nCorpus
    else:
        return float(d[tup][follower] + laplace) / (u[tup] + laplace * nWords)


def find_avg_probability(begin, end, laplace):
    sumprob = 0
    for ii in range(begin, end - n):
        tt = tuple(corpus[ii:ii + n])
        sumprob += get_prob(tt, corpus[ii + n], laplace)
    return sumprob / (nTrain - n)


print('Testing...')
print('Smoothing, train probability, test probability')
for l in range(0, 10+1):
    lap = l/10.
    tr_prob = find_avg_probability(0, nTrain, lap)
    ts_prob = find_avg_probability(nTrain, nCorpus, lap)
    print('{:8.5} {:8.5} {:8.5}'.format(lap, tr_prob, ts_prob))
