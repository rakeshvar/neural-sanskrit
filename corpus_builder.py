#! /usr/bin/python3
# -*- coding: utf-8 -*-

import re
import sys
from collections import Counter

aksh_pattern = re.compile(r"""([ऀ-औॠॡ!(),\-.0-9=?'"०-९।॥])| # Vowels
                             (([क-ह]्)*[क-ह][ा-ौ])|     # compounds
                             (([क-ह]्)*[क-ह](?![ा-्]))|  # compounds in 'a'
                             (([क-ह]्)+(?=\s))""", re.X)  # pollu

aksh2_pattern = re.compile(r"""([अ-औॠॡ][ँ-ः]?)| # Vowels
                             (([क-ह]्)*[क-ह][ा-ौ][ँ-ः]?)|     # compounds
                             (([क-ह]्)*[क-ह][ँ-ः])|  # compounds in 'a'
                             (([क-ह]्)*[क-ह](?![ा-्]))|  # compounds in 'a'
                             (([क-ह]्)+$)""", re.X)  # pollu
counts = Counter()
txtfile = open(sys.argv[1])
for line in txtfile:
    print(line)
    for word in line.split():
        print(word, end='  :  ')
        aks 2aksh_match in aksh_pattern.finditer(word):
            counts[aksh_match.group()] += 1
            print('{{{}}}'.format(aksh_match.group()), end='')
        counts['_'] += 1
        print('{_}')
    counts['$'] += 1
    print('{$}')

hashcodes = {}
i, corpus_sz = 0, 0
for k, v in sorted(counts.items(), key=lambda x:x[0]):
    hashcodes[k] = i
    print(i, k, v)
    i += 1
    corpus_sz += v

txtfile.seek(0)
corpus = []
for line in txtfile:
    for word in line.split():
        aks 2aksh_match in aksh_pattern.finditer(line):
            corpus.append(hashcodes[aksh_match.group()])
        corpus.append(hashcodes['_'])
    corpus.append(hashcodes['$'])

txtfile.close()

#print(hashcodes)

print(corpus)

import pickle, os
outfile = os.path.basename(sys.argv[1])[:-4] + '.pkl'
with open(outfile, 'wb') as f:
    pickle.dump(corpus, f, 2)

