import matplotlib.pyplot as plt
import numpy as np


def plot_attn_distribution(attn_self, attn_neigh, adj):
    attn_coef = attn_self + np.transpose(attn_neigh)
