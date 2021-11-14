from smc import SMC
from daphne import daphne
import torch
from plot import plot_histogram, plot_trace

i = 1
exp = daphne(['desugar-hoppl-cps', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a6/programs/{}.daphne'.format(i)])
n_particles = [1, 10, 100, 1000, 10000, 100000]
samples = []

for i in range(len(n_particles)):
    print(n_particles[i])
    logZ, particles = SMC(n_particles[i], exp)

    print('logZ: ', logZ)

    values = torch.stack(particles)
    samples.append(values)
    plot_histogram(samples[i].numpy(), "Histogram for mu when n-particles = {} in 1.daphne".format(n_particles[i]), "mu", "1_daphne_{}".format(n_particles[i]))

mean_samples = []
for i in range(len(n_particles)):
    mean_samples.append(torch.mean(samples[i], dtype = float))

mean_samples = torch.stack(mean_samples)

plot_trace(mean_samples.numpy(), 1, "mu", "1_daphne_trace_plot")