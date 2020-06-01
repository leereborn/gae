import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle as pkl
from input_data import parse_index_file
import collections
import matplotlib.cm as cm
import scipy.sparse as sp
import sys
from synthetic_data_generator import barabasi_albert_graph
import networkx as nx

def pca_decomp(arr):
    pca = PCA(n_components=2)
    x2d = pca.fit_transform(arr)
    return x2d

def tSNE(arr):
    embbed = TSNE(n_components=2).fit_transform(arr)
    return embbed

def get_labels():
    names = ['tx', 'ty', 'allx', 'ally']
    objects = []
    for i in range(len(names)):
        with open("data/ind.cora.{}".format(names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    tx, ty, allx, ally = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.cora.test.index")
    test_idx_range = np.sort(test_idx_reorder)
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels,axis=1)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.todense()
    return labels, features

def plot_encoder_output(arr,labels):
    plt.style.use('seaborn-whitegrid')
    plt.scatter(arr[:,0],arr[:,1],s=9,c=labels,cmap='viridis')
    plt.show()

if __name__ == "__main__":
    #labels, features = get_labels()
    #np.set_printoptions(threshold=sys.maxsize)
    #print(features[0])
    #input features plot
    '''
    features = PCA(n_components=50).fit_transform(features)
    print(features.shape)
    plot_encoder_output(features,labels)
    '''
    # encoder output plot
    #filepath = 'encoder_output_parametric.npy'
    '''
    filepath = 'encoderout_nonparametric.npy'
    arr = np.load(filepath,allow_pickle=True)
    print(arr.shape)
    arr_tSNE = tSNE(arr)
    plot_encoder_output(arr_tSNE,labels)
    '''
    G, attrs, labels = barabasi_albert_graph(1000,2,5,50,25,0.0)
    attrs=attrs.toarray()
    #arr_tSNE = tSNE(attrs)
    arr_pca = pca_decomp(attrs)
    #plot_encoder_output(arr_pca,labels)
    pos = {}
    for node in G.nodes:
        pos[node] = arr_pca[node]
    nx.draw(G,pos)
    plt.show()