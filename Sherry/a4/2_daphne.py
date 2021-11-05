from daphne import daphne
from bbvi import bbvi, calculate_mean, calculate_variance
import time
from graph_based_sampling import sample_from_joint
from plot import plot_histogram_bbvi, plot_trace
import torch


i = 2
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])
_, _, topological_sort = sample_from_joint(graph)

num_samples = 8000
L = 10
t = time.time()
rs, Ws, bbvi, sigma = bbvi(num_samples, L, graph, topological_sort)
print("\nBlack Box Variational Inference for 2.daphne took %f seconds" % (time.time() - t))

rs = torch.stack(rs)
mean = calculate_mean(rs, Ws, len(rs))
variance = calculate_variance(rs, Ws, mean)
print("\nPosterior means for slope and bias for 2.daphne are {}, {}".format(mean[0], mean[1]))

plot_histogram_bbvi(rs[:, 0], (Ws / torch.sum(Ws) * len(rs)), "Histogram for slope in 2.daphne", "slope", "2_daphne_slope_histogram")
plot_histogram_bbvi(rs[:, 1], (Ws / torch.sum(Ws) * len(rs)), "Histogram for bias in 2.daphne", "bias", "2_daphne_bias_histogram")
plot_trace(torch.stack(bbvi).numpy(), 2, "BBVI loss", "2_daphne_ELBO")