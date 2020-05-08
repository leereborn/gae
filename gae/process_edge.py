import numpy as np
import matplotlib.pyplot as plt

def load_data(p):
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
    for i in range(1,10):
        l.append(((i/10 <= nparr) & (nparr < i/10+0.1)).sum())
    print(sum(l))
    return l

def plot(y):
    plt.style.use('ggplot')
    x = ['(0.0,0.1)', '[0.1,0.2)', '[0.2,0.3)', '[0.3,0.4)', '[0.4,0.5)', '[0.5,0.6)', '[0.6,0.7)', '[0.7,0.8)', '[0.8,0.9)', '[0.9,1)']
    x_pos = [i for i,_ in enumerate(x)]
    print(x_pos)
    plt.bar(x,y,color='blue')
    plt.xlabel('Attention range')
    plt.ylabel('Frequency')
    plt.xticks(x_pos,x)
    plt.show()

if __name__ == "__main__":
    file_path = './out.txt'
    #load_data(p=file_path)
    nparr = np.loadtxt(file_path)
    freqs = count(nparr)
    plot(freqs)