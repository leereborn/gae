import matplotlib.pyplot as plt
import pickle
def line_chart(x,y1, y2, label1='GAE', label2 = 'AGAE'):
    plt.plot(x, y1, x, y2)
    plt.xlabel('p')
    plt.ylabel('accuracy')
    plt.legend((label1,label2))
    plt.show()

if __name__ == "__main__":
    with open('GAE_attr.pkl', 'rb') as f:
        acc_gae = pickle.load(f)
    with open('AGAE_attr.pkl', 'rb') as f:
        acc_agae = pickle.load(f)
    x= [i/10 for i in range(11)]
    print(acc_gae)
    print(acc_agae)
    line_chart(x,acc_gae,acc_agae)