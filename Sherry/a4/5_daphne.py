from daphne import daphne
from bbvi import bbvi, calculate_mean, calculate_variance
import time
from graph_based_sampling import sample_from_joint
from plot import plot_histogram_bbvi, plot_trace, plot_trace_params
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

i = 5
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])
_, _, topological_sort = sample_from_joint(graph)

num_samples = 10000
L = 10
t = time.time()
rs, Ws, bbvi, sigma = bbvi(num_samples, L, graph, topological_sort)
print("\nBlack Box Variational Inference for 5.daphne took %f seconds" % (time.time() - t))

alpha = []
beta = []
for i in range(len(sigma)):
    alpha.append(sigma[i][0])
    beta.append(sigma[i][1])

alpha = torch.stack(alpha)
beta = torch.stack(beta)

rs = torch.stack(rs)
mean = calculate_mean(rs, Ws, len(rs))
print("\n posterior expected value of s is {}".format(mean))

print("\n Parameter alpha of variational distribution of s is {}".format(alpha[-1].numpy()))
print("\n Parameter beta of variational distribution of s is {}".format(beta[-1].numpy()))


plot_histogram_bbvi(rs, (Ws / torch.sum(Ws) * len(rs)), "Histogram for s in 5.daphne", "s", "5_daphne_s_histogram")
plot_trace(torch.stack(bbvi).numpy(), 5, "BBVI loss", "5_daphne_ELBO")
plot_trace_params(alpha.numpy(),
                  "Trace plot for parameter loc of mu in 5.daphne\n converging to {}".format(alpha.numpy()[-1]),
                  "alpha", "5_daphne_alpha_trace")
plot_trace_params(beta.numpy(),
                  "Trace plot for parameter scale for mu in 5.daphne\n converging to {}".format(beta.numpy()[-1]),
                  "beta", "5_daphne_beta_trace")


concentrate = sigma[-1][0].numpy()
rate = sigma[-1][1].numpy()

x = np.linspace (0, 30, 50)
y1 = stats.gamma.pdf(x, a = concentrate, loc = rate)
fig, ax = plt.subplots(figsize = (8, 6))#a is alpha, loc is beta???
ax.plot(x, y1, "y-")
ax.set_title("Variational distribution for s")
ax.set(ylabel = "Probability", xlabel = "mu")
fname = os.path.join("figs", "5_daphne_pdf")
plt.savefig(fname)
print("\nFigure saved as '%s'" % fname)