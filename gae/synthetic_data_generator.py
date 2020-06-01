import networkx as nx
from networkx.generators.classic import empty_graph
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

#count1 = 0
#count2 = 0
def choose_targets(seq, m, label_dic, source_label):
    """ 
    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    """
    #global count1
    #global count2
    targets = set()
    if len(label_dic[source_label]) > m:
    #    count1 += 1
        selected = []
        for i in label_dic[source_label]:
            for j in seq:
                if i == j:
                    selected.append(j)
        while len(targets) < m:
            x = rd.choice(selected)
            targets.add(x)
    else:
    #    count2 += 1
        targets.update(label_dic[source_label])
        while len(targets) < m:
            x = rd.choice(seq)
            targets.add(x)
    return targets

def barabasi_albert_graph(n, m, num_labels, vocab_size, num_words, attr_noise):
    """Returns a random graph according to the Barabási–Albert preferential
    attachment model.

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n``.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """
    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))
    # init labels dic
    label_dic = {}
    attributes = []
    for i in range(num_labels):
        label_dic[i]=[]
    # Add m initial nodes (m0 in barabasi-speak)
    G = empty_graph(m) #Returns the empty graph with m nodes and zero edges
    labels = []
    for i in range(m):
        l = rd.randint(0,num_labels-1)
        label_dic[l].append(i)
        attributes.append(get_attributes(l,num_labels,vocab_size,num_words,attr_noise))
        labels.append(l)
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [] # accurence in repeated_nodes represents node degree.
    # Start adding the other n-m nodes. The first node is m.
    source = m
    l = rd.randint(0,num_labels-1)
    while source < n:
        labels.append(l)
        label_dic[l].append(source)
        attributes.append(get_attributes(l,num_labels,vocab_size,num_words,attr_noise))
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        l = rd.randint(0,num_labels-1)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = choose_targets(repeated_nodes, m, label_dic,l)
        source += 1

    attributes = np.array(attributes)
    attributes = sp.coo_matrix(attributes)
    print(attributes.shape)
    return G, attributes, labels

def get_attributes(label, num_labels, vocab_size, num_words, attr_noise):
    words=np.zeros(vocab_size,dtype=np.float32)
    p=(1+label)/(1+num_labels)
    n=vocab_size
    for i in range(num_words):
        r = rd.random()
        if r <= attr_noise:
            words[rd.randint(0,vocab_size-1)]=1.
        else:
            words[np.random.binomial(n-1,p,size=1)]=1. #if n then index out of bound exeception
    return words

def get_synthetic_data(num_nodes, m, num_labels, vocab_size, num_words, attr_noise):
    G, attrs,_ = barabasi_albert_graph(num_nodes, m, num_labels, vocab_size, num_words, attr_noise)
    sparse_adj = nx.adjacency_matrix(G)
    return sparse_adj, attrs

if __name__ == "__main__":
    '''
    link prediction tasks no need node labels!
    '''
    G, attrs,_ = barabasi_albert_graph(100,2,5,50,25,0.0)
    #print(count1,count2)
    #sparse_adj = nx.adjacency_matrix(G)
    #print(G.nodes())
    #G = nx.barabasi_albert_graph(100,5)
    #print(G.degree)
    nx.draw(G)
    #nx.draw_spring(G)
    #nx.draw_circular(G)
    #nx.draw_spectral(G)
    plt.show()
    #print(get_attributes(1, 6, 10, 5, 0.2))