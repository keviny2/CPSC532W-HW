from smc import SMC
from daphne import daphne
import torch
import numpy as np
from plot import plot_histogram_evidence, plot_histogram_variance_evidence, plot_trace

i = 2
exp = daphne(['desugar-hoppl-cps', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a6/programs/{}.daphne'.format(i)])
n_particles = [1, 10, 100, 1000, 10000, 100000]
samples = []
mean_samples = []
logZs = []

for i in range(len(n_particles)):
    print(n_particles[i])
    logZ, particles = SMC(n_particles[i], exp)

    print('logZ: ', logZ)
    logZs.append(logZ)

    values = torch.stack(particles)
    samples.append(values)
    mean = torch.mean(samples[i], dtype = float)
    mean_samples.append(mean)

    if i == 0:
        plot_histogram_evidence(samples[i].numpy(), mean, torch.exp(logZ),
                                "Histogram for mu when n-particles = {} in 2.daphne".format(n_particles[i]),
                       "mu", "2_daphne_{}".format(n_particles[i]))
    else:
        variance = np.var(samples[i].numpy())
        plot_histogram_variance_evidence(samples[i].numpy(), mean, variance, torch.exp(logZ),
                                         "Histogram for mu when n-particles = {} in 2.daphne".format(n_particles[i]),
                                "mu", "2_daphne_{}".format(n_particles[i]))



mean_samples = torch.stack(mean_samples)
logZs = torch.stack(logZs)

plot_trace(mean_samples.numpy(), "Posterior expectation for 2.daphne", "mu", "2_daphne_trace_plot")
Zs = torch.exp(torch.FloatTensor(logZs))
plot_trace(Zs.numpy(), "Marginal probability/evidence estimate for 2.daphne", "probability", "2_daphne_evidence")