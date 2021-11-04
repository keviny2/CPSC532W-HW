from daphne import daphne
from bbvi import bbvi, calculate_mean, calculate_variance
import time
from graph_based_sampling import sample_from_joint
from plot import plot_histogram_bbvi, plot_trace
import torch


i = 1
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])
_, _, topological_sort = sample_from_joint(graph)

num_samples = 4000
L = 10
t = time.time()
rs, Ws,bbvi = bbvi(num_samples, L, graph, topological_sort)
print("\nBlack Box Variational Inference for 1.daphne took %f seconds" % (time.time() - t))

rs = torch.stack(rs)
mean = calculate_mean(rs, Ws, len(rs))
variance = calculate_variance(rs, Ws, mean)
print("\nPosterior expected value of mu is {}".format(mean))
print("\nPosterior variance of mu is {}".format(variance))

plot_histogram_bbvi(rs, Ws, "Histogram for mu in 1.daphne", "mu", "1_daphne_histogram")
plot_trace(torch.stack(bbvi).numpy(), 1, "BBVI loss", "1_daphne_ELBO")



