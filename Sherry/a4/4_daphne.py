from daphne import daphne
from bbvi import bbvi, calculate_mean
import time
from graph_based_sampling import sample_from_joint
from plot import plot_histogram_bbvi, plot_trace, plot_heatmap
import torch
import numpy as np

i = 4
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])
_, _, topological_sort = sample_from_joint(graph)



num_samples = 2000
L = 10
t = time.time()
rs, Ws, bbvi = bbvi(num_samples, L, graph, topological_sort)
print("\nBlack Box Variational Inference for 4.daphne took %f seconds" % (time.time() - t))

W_0s = []
b_0s = []
W_1s = []
b_1s = []
for i in range(len(rs)):
    W_0s.append(rs[i][0].flatten())
    b_0s.append(rs[i][1].flatten())
    W_1s.append(rs[i][2])
    b_1s.append(rs[i][3].flatten())
W_0s = torch.stack(W_0s).numpy()
b_0s = torch.stack(b_0s).numpy()
W_1s = torch.stack(W_1s).numpy()
b_1s = torch.stack(b_1s).numpy()

def calculate_vector_mean(rs):
    means = []
    for i in range(len(rs[1])):
        mean = np.mean(rs[:, i])
        means.append(mean)
    return means

def calculate_vector_variance(rs):
    variances = []
    for i in range(len(rs[1])):
        variance = np.var(rs[:, i])
        variances.append(variance)
    return variances


def calculate_matrix_mean(rs):
    means = np.zeros((len(rs[0]), len(rs[0])))

    for i in range(len(rs[0])):
        for j in range(len(rs[0])):
            means[i][j] = np.mean(rs[:, i, j])

    return means

def calculate_matrix_variance(rs):
    variances = np.zeros((len(rs[0]), len(rs[0])))

    for i in range(len(rs[0])):
        for j in range(len(rs[0])):
            variances[i][j] = np.var(rs[:, i, j])

    return variances



plot_trace(torch.stack(bbvi).numpy(), 4, "BBVI loss", "4_daphne_ELBO")

plot_heatmap(np.array(calculate_vector_mean(W_0s))[:, np.newaxis], "Heapmap for W0 mean", "4_daphne_w0_mean_heatmap")
plot_heatmap(np.array(calculate_vector_variance(W_0s))[:, np.newaxis], "Heatmap for W0 variance", "4_daphne_w0_variance_heatmap")

plot_heatmap(np.array(calculate_vector_mean(b_0s))[:, np.newaxis], "Heatmap for b0 mean", "4_daphne_b0_mean_heatmap")
plot_heatmap(np.array(calculate_vector_variance(b_0s))[:, np.newaxis], "Heatmap for b0 variance", "4_daphne_b0_variance_heatmap")

plot_heatmap(calculate_matrix_mean(W_1s), "Heapmap for W1 mean", "4_daphne_w1_mean_heatmap")
plot_heatmap(calculate_matrix_variance(W_1s), "Heatmap for W1 variance", "4_daphne_w1_variance_heatmap")

plot_heatmap(np.array(calculate_vector_mean(b_1s))[:, np.newaxis], "Heatmap for b1 mean", "4_daphne_b1_mean_heatmap")
plot_heatmap(np.array(calculate_vector_variance(b_1s))[:, np.newaxis], "Heatmap for b1 variance", "4_daphne_b1_variance_heatmap")
