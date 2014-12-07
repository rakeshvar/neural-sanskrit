# -*- coding: utf-8 -*-
# Usage
# python visual_output.py <DIMENSIONS> <pca pickle> <tsne pickle> <akshara pickle file> <.list file>

import sys
import pickle
import pylab
import ast
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

if(len(sys.argv)<7):
	print('\nUsage: python {} <DIMENSIONS> <COUNT_THRESHOLD> <pca pickle> <tsne pickle> '
		  '<akshara pickle file> <.list file>'.format(sys.argv[0]))
	sys.exit()

DIMENSIONS = int(sys.argv[1])
COUNT_THRESHOLD = int(sys.argv[2])

####################### READ IN DIM REDUCTION DATA FROM FILE #########################

# Read in PCA information
with open(sys.argv[3],'rb') as f: 
    pcs_pca = pickle.load(f)

# Read in tSNE information
with open(sys.argv[4],'rb') as f: 
    pcs_tsne = pickle.load(f)

# Read in the aksharas
with open(sys.argv[5],'rb') as f: 
    aksharas = pickle.load(f)

# Read in akshara frequencies
with open(sys.argv[6],'rb') as f: 
    aksharas_freq = ast.literal_eval(f.read())

nWords=len(aksharas)
x=[]
y=[]
z=[]
if DIMENSIONS>=3:
	# plot PCA 
    x.append(pcs_pca[:,0])
    y.append(pcs_pca[:,1])
    z.append(pcs_pca[:,2])

    # plot tSNE
    x.append(pcs_tsne[:,0])
    y.append(pcs_tsne[:,1])
    z.append(pcs_tsne[:,2])

    for ind in [0,1]:
	    np.random.seed(5)
	    fig = plt.figure(1, figsize=(4, 3))
	    plt.clf()
	    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	    plt.cla()

	    ax.scatter(x[ind],y[ind], z[ind], c=labels.astype(np.float), s=100)

	    ax.w_xaxis.set_ticklabels([])
	    ax.w_yaxis.set_ticklabels([])
	    ax.w_zaxis.set_ticklabels([])
	    ax.set_xlabel('1st PC')
	    ax.set_ylabel('2nd PC')
	    ax.set_zlabel('3rd PC')

	    for i, txt in enumerate([aksharas[j] for j in range(0,len(aksharas),50)]):
	        x2, y2, _ = proj3d.proj_transform(x[ind][i],y[ind][i],z[ind][i], ax.get_proj())    
	        label = pylab.annotate(txt,
	            xycoords='data', 
	            xy = (x2, y2), xytext = (60, 20),
	            textcoords = 'offset points', ha = 'right', va = 'bottom',
	            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
	            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
	    plt.show()
else:
    # plot PCA 
    x.append(pcs_pca[:,0])
    y.append(pcs_pca[:,1])

    # plot tSNE
    x.append(pcs_tsne[:,0])
    y.append(pcs_tsne[:,1])

    # # Calculate KMeans Clustering
    # estimator = KMeans(n_clusters=15)
    # estimator.fit(pcs_tsne)
    # labels = estimator.labels_

    # Calculate Agglomerative Clustering
    estimator = AgglomerativeClustering(n_clusters=15)
    estimator.fit(pcs_tsne)
    labels = estimator.labels_

    for ind in [0,1]:
	    plt.scatter(x[ind], y[ind], c=[1, 1, 1], s=0)#c=labels.astype(np.float), s=1000)
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
	    (.1, .5, .1),  
	    (.5, .1, .1),
	    (.1, .1, .5), 
	    (.5, .5, .1), 
	    (.1, .5, .5),   
	    ]
	    for label, i, j in zip(range(nWords), x[ind], y[ind]):
	        if aksharas_freq[label][1]<COUNT_THRESHOLD:
	            continue
	        if ind==0: # PCA
		        unicode_endings = [u'ा', u'ि', u'ी', u'ु', u'ू', u'ृ', u'े', u'ै', u'ो', u'ौ', u'ं', u'ः', u'्']
		        if (aksharas[label][-1]) in unicode_endings:
		        	color_ind = unicode_endings.index(aksharas[label][-1])
		        elif aksharas[label] in u'अआइईउऊऋएऐओऔ':
		        	color_ind = 13
		        else: 
		        	color_ind = 14
		        plt.annotate(
		            aksharas[label], size = min(25, 2 + 1.5 * aksharas_freq[label][1]/500),
		            # xy=(i, j), color = cols[-1+len(aksharas[label])],
		            xy=(i, j), color = cols[color_ind],
		            xytext = (0, 0),
		            textcoords = 'offset points', # ha = 'right', va = 'bottom',
		            #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
		            #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
		        )
	        else: # tSNE - use k-means to cluster
		    	plt.annotate(
		            # aksharas[label], size = min(25, 2 + 1.5 * aksharas_freq[label][1]/500),
		            aksharas[label], size = 15,
		            # xy=(i, j), color = cols[-1+len(aksharas[label])],
		            xy=(i, j), color = cols[labels[label]],
		            xytext = (0, 0),
		            textcoords = 'offset points', # ha = 'right', va = 'bottom',
		            #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
		            #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
		        )
	    plt.show()
