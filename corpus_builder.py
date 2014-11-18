#! /usr/bin/python3
# -*- coding: utf-8 -*-

import re
import sys
from collections import Counter

aksh_pattern = re.compile(r"""([ऀ-औॠॡ!(),\-.0-9=?'"०-९।॥])|	# Vowels
							 (([क-ह]्)*[क-ह][ा-ौ])|     # compounds
							 (([क-ह]्)*[क-ह](?![ा-्]))|  # compounds in 'a'
							 (([क-ह]्)+(?=\s))""", re.X)  # pollu
counts = Counter()
dump = open(sys.argv[1])
for line in dump:
    for aksh_match in aksh_pattern.finditer(line):
        counts[aksh_match.group()] += 1

hashcodes = {}
i, corpus_sz = 0, 0
for k, v in sorted(counts.items(), key=lambda x:x[0]):
    hashcodes[k] = i
    print(i, k, v)
    i += 1
    corpus_sz += v

print(hashcodes)
#import  numpy as np
dump.seek(0)
corpus = []
for line in dump:
    for aksh_match in aksh_pattern.finditer(line):
        corpus.append(hashcodes[aksh_match.group()])

print(corpus)
dump.close()

import numpy as np
import pickle
with open('blah.pkl', 'wb') as f:
    pickle.dump(corpus, f, 2)
