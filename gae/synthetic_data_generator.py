import networkx as nx
from networkx.generators.classic import empty_graph
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import collections

#count1 = 0
#count2 = 0
def choose_targets(label_dic_set, m, label_dic, source_label):
    """ 
    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    """
    #global count1
    #global count2
    targets = set()
    if len(label_dic_set[source_label]) > m:
        #print(1)
        while len(targets) < m:
            x = rd.choice(label_dic[source_label])
            targets.add(x)
    else:
        #print(2)
        targets.update(label_dic_set[source_label])
        seq = []
        for k, v in label_dic.items():
            seq.extend(v)
        while len(targets) < m:
            x = rd.choice(seq)
            targets.add(x)
    return targets

def choose_targets_random(existing_nodes, m, label_dic, source_label):
    #global count1
    #global count2
    targets = set()
    if len(label_dic[source_label]) > m:
        while len(targets) < m:
            x = rd.choice(label_dic[source_label])
            targets.add(x)
    else:
        targets.update(label_dic[source_label])
        while len(targets) < m:
            x = rd.choice(existing_nodes)
            targets.add(x)
    return targets

def random_graph(n, m, num_labels, vocab_size, num_words, attr_noise):
    if m < 1 or m >= n:
        raise nx.NetworkXError("Random network must have m >= 1"
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
    existing_nodes = [] # accurence in repeated_nodes represents node degree.
    existing_nodes.extend(targets)
    # Start adding the other n-m nodes. The first node is m.
    source = m
    l = rd.randint(0,num_labels-1)
    while source < n:
        labels.append(l)
        label_dic[l].append(source)
        attributes.append(get_attributes(l,num_labels,vocab_size,num_words,attr_noise))
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        
        existing_nodes.append(source)
        l = rd.randint(0,num_labels-1)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = choose_targets_random(existing_nodes, m, label_dic,l)
        source += 1

    attributes = np.array(attributes)
    attributes = sp.coo_matrix(attributes)
    print(attributes.shape)
    return G, attributes, labels

def caveman_small_world(p, community_num, community_size, vocab_size, num_words, attr_noise):
    graph = nx.connected_caveman_graph(community_num, community_size)
    attributes = []
    count = 0
    count2 = 0
    p_remove = 0.0

    for (u, v) in graph.edges():
        if rd.random() < p_remove:
            graph.remove_edge(u, v)
            count2 += 1

    for (u, v) in graph.edges():
        if rd.random() < p:  # rewire the edge
            x = rd.choice(list(graph.nodes))
            if graph.has_edge(u, x):
                continue
            graph.remove_edge(u, v)
            graph.add_edge(u, x)
            count += 1
    print('rewire:', count)
    print('removed:',count2)

    #import pdb;pdb.set_trace()
    for u in list(graph.nodes):
        label = u//community_size
        attributes.append(get_attributes(label,community_num,vocab_size,num_words,attr_noise))
    attributes = np.array(attributes)
    attributes = sp.coo_matrix(attributes)
    return graph, attributes

def pure_random_graph(num_nodes, num_edges, num_labels, vocab_size, num_words, attr_noise):
    G = nx.dense_gnm_random_graph(num_nodes, num_edges)
    attributes = []
    for u in list(G.nodes):
        label = rd.randint(0,num_labels-1)
        attributes.append(get_attributes(label,num_labels,vocab_size,num_words,attr_noise))
    attributes = np.array(attributes)
    attributes = sp.coo_matrix(attributes)
    return G, attributes

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
    """
    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))
    # init labels dic
    label_dic = {}
    label_dic_set = {}
    attributes = []
    for i in range(num_labels):
        label_dic[i]=[]
        label_dic_set[i] = []
    # Add m initial nodes (m0 in barabasi-speak)
    G = empty_graph(m) #Returns the empty graph with m nodes and zero edges
    labels = []
    for i in range(m):
        l = rd.randint(0,num_labels-1)
        label_dic_set[l].append(i)
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
        #print('node:',source)
        labels.append(l)
        attributes.append(get_attributes(l,num_labels,vocab_size,num_words,attr_noise))
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        
        # Add one node to the list for each new edge just created.
        for i in targets:
            label_dic[labels[i]].append(i)
        # And the new node "source" has m edges to add to the list.
        label_dic[l].extend([source] * m)
        label_dic_set[l].append(source)
        l = rd.randint(0,num_labels-1)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = choose_targets(label_dic_set, m, label_dic,l)
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

def get_synthetic_data(p=0.01, attrNoise = 0.2):
    G, attrs,_ = barabasi_albert_graph(3000,10,10,50,25,attrNoise)
    #G, attrs,_ = random_graph(3000,10,5,50,25,0.2)
    #G, attrs = caveman_small_world(p=p, community_num=10, community_size=300, vocab_size=50, num_words=25, attr_noise=attrNoise)
    #G, attrs = pure_random_graph(num_nodes=3000, num_edges=448500, num_labels=10, vocab_size=50, num_words=25, attr_noise=0.2)
    print(G.number_of_nodes(), G.number_of_edges())
    sparse_adj = nx.adjacency_matrix(G)
    return sparse_adj, attrs

def draw_degree_distro(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    #print(deg,cnt)
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.8, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()

if __name__ == "__main__":
    '''
    link prediction tasks no need node labels!
    '''
    G, attrs,_ = barabasi_albert_graph(500,10,10,50,25,0.0)
    #G, attrs = pure_random_graph(num_nodes=200, num_edges=1000, num_labels=5, vocab_size=50, num_words=25, attr_noise=0.2)
    #G, attrs = caveman_small_world(p=0.01, community_num=10, community_size=20, vocab_size=50, num_words=25, attr_noise=0.2)
    #print(count1,count2)
    #sparse_adj = nx.adjacency_matrix(G)
    #print(G.nodes())
    #G = nx.barabasi_albert_graph(100,5)
    #print(G.degree)
    #nx.draw_networkx(G,with_labels = False, node_size=100)
    draw_degree_distro(G)
    plt.show()
    #print(get_attributes(1, 6, 10, 5, 0.2))