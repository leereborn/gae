import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

def trim_data(p):
    with open(p,'r') as f:
        s = f.read()
    with open(p,'w') as f: 
        s = s.replace(']','')
        s = s.replace(' [','')
        s = s.replace('[','')
        f.write(s)

def count(nparr):
    print(nparr.shape)
    print(np.count_nonzero(nparr))
    l=[]
    l.append(((0 < nparr) & (nparr < 0.1)).sum())
    for i in range(1,9):
        l.append(((i/10 <= nparr) & (nparr < i/10+0.1)).sum())
    l.append(((0.9 <= nparr) & (nparr <= 1.0)).sum())
    print(sum(l))
    print(l)
    return l

def plot(y_attn, y_norm):
    plt.style.use('ggplot')
    x = ['(0.0,0.1)', '[0.1,0.2)', '[0.2,0.3)', '[0.3,0.4)', '[0.4,0.5)', '[0.5,0.6)', '[0.6,0.7)', '[0.7,0.8)', '[0.8,0.9)', '[0.9,1]']
    ind = np.arange(len(x))
    width = 0.35
    plt.bar(ind,y_attn,width,label='Attention', color='blue')
    plt.bar(ind+width,y_norm,width, label = 'Normalization',color='red')
    plt.xlabel('Edge weight ranges')
    plt.ylabel('Frequency')
    plt.xticks(ind+width/2,x)
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    file_path = './out.txt'
    adj_norm_path = './adj_norm_sparse.npz'
    #trim_data(p=file_path)
    attn_w = np.loadtxt(file_path)
    sparse_adj_norm = sp.load_npz(adj_norm_path)
    dense_adj_norm = sparse_adj_norm.toarray()
    idxs = np.where(dense_adj_norm==1.0)
    coords = list(zip(idxs[0],idxs[1]))
    print(coords)
    print(len(coords)) # There are 104 isolated nodes
    #freqs_norm = count(dense_adj_norm)
    #freqs_attn = count(attn_w)
    #plot(freqs_attn,freqs_norm)