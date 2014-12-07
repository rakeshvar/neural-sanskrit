############################ VISUALIZATION ################################
from __future__ import print_function
import sys
import ast
import os
import pickle
from operator import mul
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.cluster import KMeans
from sklearn import datasets
import pylab
import tsne
import ast 

####################### PARAMS ##########################

REDUCTION_TYPE=sys.argv[4]
DIMENSIONS = 2
CLUSTERS = 3

####################### READ IN DATA FROM FILE##########################

# Read in the lexicon
inpt_file_name = sys.argv[1]
with open(inpt_file_name, 'rb') as f:
    corpus = pickle.load(f)
lexicon = corpus.get('lexicon')

with open(sys.argv[3],'rb') as f1: 
    aksharas_freq = ast.literal_eval(f1.read())

# Read in the aksharas
with open(sys.argv[2],'rb') as f1: 
    aksharas = pickle.load(f1)
nWords=len(aksharas)

####################### DIMENSIONALITY REDUCTION ##########################

if(REDUCTION_TYPE=="PCA"):
    pca = PCA(n_components=DIMENSIONS)
    pcs = pca.fit_transform(lexicon)
    print('PC shape:', pcs.shape)
else: 
    print(lexicon.shape)
    pcs = tsne.tsne(lexicon.astype('float64'), no_dims = 2, initial_dims = 12, perplexity = 30.0)
    print(type(pcs))
    print('tSNE PC shape:', pcs.shape)

####################### CLUSTERING ALGORITHM (K MEANS) ##########################

estimator = KMeans(n_clusters=CLUSTERS)
estimator.fit(pcs)
labels = estimator.labels_
if DIMENSIONS>=3:
    np.random.seed(5)
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()

    ax.scatter(pcs[:,0], pcs[:,1], pcs[:,2], c=labels.astype(np.float), s=100)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('1st PC')
    ax.set_ylabel('2nd PC')
    ax.set_zlabel('3rd PC')

    for i, txt in enumerate([aksharas[j] for j in range(0,len(aksharas),50)]):
        x2, y2, _ = proj3d.proj_transform(pcs[i,0],pcs[i,1],pcs[i,2], ax.get_proj())    
        label = pylab.annotate(txt,
            xycoords='data', 
            xy = (x2, y2), xytext = (60, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
else:
    x=pcs[:,0]
    y=pcs[:,1]
    plt.scatter(x, y, c=[1, 1, 1], s=0)#c=labels.astype(np.float), s=1000)
    cols = [
    (0, 0, 0),
    (.1, .1, 1),
    (1, .1, .1),
    (.1, 1, .1),
    (1, .5, .5),
    (1, .1, 1),
    (.1, .5, 0),
    (.5, 0, .1),
    (.0, .1, .5),
    (.51, .51, 0),
    ]
    for label, i, j in zip(range(nWords), x, y):
        if aksharas_freq[label][1]//1000 < 10:
            pass #continue
        plt.annotate(
            aksharas[label], size = min(25, 2 + 1.5 * aksharas_freq[label][1]/1000),
            xy=(i, j), color = cols[-1+len(aksharas[label])],
            xytext = (0, 0),
            textcoords = 'offset points', # ha = 'right', va = 'bottom',
            #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
        )

plt.show()
