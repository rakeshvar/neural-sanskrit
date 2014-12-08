#! /usr/bin/python3
# -*- coding: utf-8 -*-

import re
import sys
from collections import Counter

if len(sys.argv) < 2:
    print('Usage:\n{} <path_to_file.txt> [log_output=0]'.format(sys.argv[0]))
    sys.exit()

try:
    log_enabled = bool(sys.argv[2])
except IndexError:
    log_enabled = False

if log_enabled:
    def log(*args, **kwargs):
        print(*args, **kwargs)
else:
    def log(*args, **kwargs):
        pass

akshara_pattern = re.compile(r"""([अ-औॠॡ][ँ-ः]?)| # Vowels
                             (([क-ह]्)*[क-ह][ा-ौ][ँ-ः]?)|     # compounds
                             (([क-ह]्)*[क-ह][ँ-ः])|  # compounds in 'a'
                             (([क-ह]्)*[क-ह](?![ा-्]))|  # compounds in 'a'
                             (([क-ह]्)+$)""", re.X)  # pollu
counts = Counter()
txtfile = open(sys.argv[1])

#
# First parse the input file to find unique aksharas
#
for line in txtfile:
    log(line)
    for word in line.split():
        log(word, end='  :  ')
        for aksh_match in akshara_pattern.finditer(word):
            counts[aksh_match.group()] += 1
            log('{{{}}}'.format(aksh_match.group()), end='')
        counts['_'] += 1
        log('{_}')
    counts['$'] += 1
    log('{$}')

#
# Encode the above aksharas as integers
#
akaara_codes = {}
i_code = 0
for akshara, count in sorted(counts.items(), key=lambda x:x[0]):
    akaara_codes[akshara] = i_code
    log(i_code, akshara, count)
    i_code += 1

#
# Now build the actual corpus as a sequence of integers
#
txtfile.seek(0)
corpus = [akaara_codes['$']]
for line in txtfile:
    for word in line.split():
        # Add each akshara from each word
        for aksh_match in akshara_pattern.finditer(word):
            corpus.append(akaara_codes[aksh_match.group()])
            log('{{{}}}'.format(aksh_match.group()), end='')

        # Add a space if the previous character is not a space or newline
        if not corpus[-1] in (akaara_codes['$'], akaara_codes['_']):
            corpus.append(akaara_codes['_'])
            log('{_}')

    # Add newline if previous character was not newline
    # If previous character is a space replace it with newline
    if corpus[-1] != akaara_codes['$']:
        if corpus[-1] == akaara_codes['_']:
            corpus[-1] = akaara_codes['$']
            log('{<$}')
        else:
            corpus.append(akaara_codes['$'])
            log('{$}')

txtfile.close()
log(akaara_codes)
log(corpus)

#
# Now save the corpus and the akaara_codes
#
import pickle, os
outfile = os.path.basename(sys.argv[1])[:-4] + '.pkl'
with open(outfile, 'wb') as f:
    pickle.dump(corpus, f, 2)

hashfile = os.path.basename(sys.argv[1])[:-4] + '.list'
with open(hashfile, 'w') as f:
    f.write('# -*- coding: utf-8 -*-\n'
            '# The aksharas, their codes and counts stored as a list\n'
            '[\n')
    for akshara, code in sorted(akaara_codes.items(), key=lambda x:x[1]):
        f.write("(u'{}', {}), \n".format(akshara, counts[akshara]))
    f.write(']')
