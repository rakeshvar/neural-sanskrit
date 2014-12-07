############################ VISUALIZATION ################################
from __future__ import print_function
import sys
import os
import pickle
from operator import mul
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import datasets
import tsne
import ast 

####################### READ IN DATA FROM FILE##########################

# Read in the lexicon
inpt_file_name = sys.argv[1]
with open(inpt_file_name, 'rb') as f:
    corpus = pickle.load(f)

lexicon = corpus.get('lexicon')

nWords, nDims = lexicon.shape
print(lexicon.shape)

corpus_name = os.path.splitext(inpt_file_name)[0]
print(corpus_name)

####################### PCA ##########################

pca = PCA(n_components=5)
pcs = pca.fit_transform(lexicon)

with open(corpus_name + '_pca.pkl', "wb") as f:
    pickle.dump(pcs, f, -1)


####################### TSNE ##########################

tsnes = tsne.tsne(lexicon.astype('float64'), no_dims = 2, 
    initial_dims = lexicon.shape[1], perplexity = 30.0)

with open(corpus_name + '_tsne.pkl', "wb") as f:
    pickle.dump(tsnes, f, -1)
