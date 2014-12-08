# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys
import pickle
import ast

import numpy as np
import pylab
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# ###################### PARSE PARAMS #########################

if len(sys.argv) < 3:
    print('\nUsage: python {} <components.pkl> <corpus.list file> [threshold=0]'
          ''.format(sys.argv[0]))
    sys.exit()

pca_file_name = sys.argv[1]
list_file_name = sys.argv[2]

try:
    threshold = int(sys.argv[3])
except IndexError:
    threshold = 0

# ###################### READ IN DIM REDUCTION DATA FROM FILE ##################

print('Reading the representaion information..')
with open(pca_file_name, 'rb') as pca_file:
    components = pickle.load(pca_file)

print('Reading the akshara counts list...')
with open(list_file_name, 'rb') as list_file:
    aksh_counts = ast.literal_eval(list_file.read())

colours = [(0, 0, 0),
           (.1, .1, 1),
           (1, .1, .1),
           (.1, 1, .1),
           (1, .5, .5),
           (1, .1, 1),
           (.1, .5, 0),
           (.5, 0, .1),
           (.0, .1, .5),
           (.51, .51, 0),
           (.1, .5, .1),
           (.5, .1, .1),
           (.1, .1, .5),
           (.5, .5, .1),
           (.1, .5, .5), ]

if sys.platform.startswith('linux'):
    matplotlib.rc('font', **{'sans-serif': 'gargi', 'family': 'sans-serif'})

nAksharas = len(aksh_counts)

# ###################### PLOT #########################################


def plot3(x, y, z, cols):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()

    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()

    ax.scatter(x, y, z, c=cols.astype(np.float), s=100)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('1st Component')
    ax.set_ylabel('2nd Component')
    ax.set_zlabel('3rd Component')

    for i in range(0, nAksharas, nAksharas//100):
        akshara, count = aksh_counts[i]
        if count < threshold:
            continue
        x2, y2, _ = proj3d.proj_transform(x[i], y[i], z[i], ax.get_proj())
        pylab.annotate(
            akshara,
            xycoords='data',
            xy=(x2, y2),
            xytext=(60, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.show()


def plot2(x, y, col_fn, size_fn):
    plt.scatter(x, y, s=0)  # Dummy call plots blanks

    for i in range(nAksharas):
        akshara, count = aksh_counts[i]
        if count < threshold:
            continue

        plt.annotate(
            akshara,
            size=size_fn(i),
            color=col_fn(i),
            xy=(x[i], y[i]),
            xytext=(0, 0),
            textcoords='offset points',
        )

    plt.show()


def vowel_colour(akshara_idx):
    akshara = aksh_counts[akshara_idx][0]
    vowel_endings = [u'ा', u'ि', u'ी', u'ु', u'ू', u'ृ', u'े', u'ै', u'ो', u'ौ',
                     u'ं', u'ः', u'्']

    if (akshara[-1]) in vowel_endings:
        idx = vowel_endings.index(akshara[-1])
    elif akshara in u'अआइईउऊऋएऐओऔ':
        idx = len(vowel_endings)
    else:
        idx = len(vowel_endings) + 1

    return colours[idx]


def predef_colour(classes):
    def _predef_colour(akshara_idx):
        return colours[classes[akshara_idx]]

    return _predef_colour


def count_size(akshara_idx):
    count = aksh_counts[akshara_idx][1]
    newlines = aksh_counts[0][1]
    return min(25, 2 + 23 * (float(count) / newlines) ** .2)


def fixed_size(size):
    def _fixed_size(askhara_idx):
        return size

    return _fixed_size

##################################### Main Code ###############################
print('Plotting 2D with vowel colour & count size')
plot2(components[:, 0], components[:, 1], vowel_colour, count_size)
print('Plotting 2D with vowel colour & fixed size')
plot2(components[:, 0], components[:, 1], vowel_colour, fixed_size(15))

## Calculate KMeans Clustering
print('Finding K-Means...')
estimator = KMeans(n_clusters=15)
estimator.fit(components)
kmeans_labels = estimator.labels_

print('Plotting 2D with kmeans colour & count size')
plot2(components[:, 0], components[:, 1], predef_colour(kmeans_labels),
      count_size)
print('Plotting 2D with kmeans colour & fixed size')
plot2(components[:, 0], components[:, 1], predef_colour(kmeans_labels),
      fixed_size(15))

## Calculate Agglomerative Clustering
print('Finding Agglomorative Clusters...')
estimator = AgglomerativeClustering(n_clusters=15)
estimator.fit(components)
agglo_labels = estimator.labels_

print('Plotting 2D with agglomorative colour & count size')
plot2(components[:, 0], components[:, 1], predef_colour(agglo_labels),
      count_size)
print('Plotting 2D with agglomorative colour & fixed size')
plot2(components[:, 0], components[:, 1], predef_colour(agglo_labels),
      fixed_size(15))

# 3D plot
print('Plotting 3D with kmeans colours')
plot3(components[:, 0], components[:, 1], components[:, 2], kmeans_labels)
print('Plotting 3D with agglomorative colours')
plot3(components[:, 0], components[:, 1], components[:, 2], agglo_labels)
