from daphne import daphne
from bbvi import bbvi, calculate_mean
import time
from graph_based_sampling import sample_from_joint
import torch
from plot import plot_trace


i = 3
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])
_, _, topological_sort = sample_from_joint(graph)



num_samples = 2000
L = 100
t = time.time()
rs, Ws, bbvi = bbvi(num_samples, L, graph, topological_sort)
print("\nBlack Box Variational Inference for 3.daphne took %f seconds" % (time.time() - t))

rs = torch.stack(rs)
mean = calculate_mean(rs, Ws, len(rs))

plot_trace(torch.stack(bbvi).numpy(), 3, "BBVI loss", "3_daphne_ELBO")
