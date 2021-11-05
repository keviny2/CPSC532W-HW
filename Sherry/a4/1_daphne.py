from daphne import daphne
from bbvi import bbvi, calculate_mean, calculate_variance
import time
from graph_based_sampling import sample_from_joint
from plot import plot_histogram_bbvi, plot_trace, plot_histogram, plot_trace_params
import torch
import math
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import os

i = 1
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])
_, _, topological_sort = sample_from_joint(graph)

num_samples = 10000
L = 10
t = time.time()
rs, Ws, bbvi, sigma = bbvi(num_samples, L, graph, topological_sort)
print("\nBlack Box Variational Inference for 1.daphne took %f seconds" % (time.time() - t))

locs = []
scales = []
for i in range(len(sigma)):
    locs.append(sigma[i][0])
    scales.append(sigma[i][1])

locs = torch.stack(locs)
scales = torch.stack(scales)


rs = torch.stack(rs)
mean = calculate_mean(rs, Ws, len(rs))
variance = calculate_variance(rs, Ws, mean)
print("\nPosterior expected value of mu is {}".format(mean))
print("\nParameter loc of variational distribution of mu is {}".format(locs[-1].numpy()))
print("\nParameter scale of variational distribution of mu is {}".format(scales[-1].numpy()))

plot_histogram_bbvi(rs, (Ws / torch.sum(Ws) * len(rs)), "Histogram for mu in 1.daphne", "mu", "1_daphne_histogram")
plot_trace(torch.stack(bbvi).numpy(), 1, "BBVI loss", "1_daphne_ELBO")
plot_trace_params(locs.numpy(),
                  "Trace plot for parameter loc of mu in 1.daphne\n converging to {}".format(locs.numpy()[-1]),
                  "loc", "1_daphne_loc_trace")
plot_trace_params(scales.numpy(),
                  "Trace plot for parameter scale for mu in 1.daphne\n converging to {}".format(scales.numpy()[-1]),
                  "scale", "1_daphne_scale_trace")


loc = sigma[-1][0].detach().numpy()
scale = sigma[-1][1].detach().numpy()

def normal_pdf(x, mu=loc, sigma=scale):
    sqrt_two_pi = math.sqrt(math.pi * 2)
    return math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma)

df = pd.DataFrame({'x1': numpy.arange(-10, 10, 0.1), 'y1': map(normal_pdf, numpy.arange(-10, 10, 0.1))})

fig, ax = plt.subplots(figsize = (8, 6))
ax.plot('x1', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=5, color='skyblue', linewidth=1)
ax.set_title("Variational distribution for mu")
ax.set(ylabel = "Probability", xlabel = "mu")
fname = os.path.join("figs", "1_daphne_pdf")
plt.savefig(fname)
print("\nFigure saved as '%s'" % fname)

