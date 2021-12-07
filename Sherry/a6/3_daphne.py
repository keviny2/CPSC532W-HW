from smc import SMC
from daphne import daphne
import torch
import numpy as np
from plot import plot_trace, plot_histogram_evidence, plot_histogram_variance_evidence

i = 3
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
    mean = np.mean(samples[i].numpy(), axis = 0)
    mean_samples.append(torch.FloatTensor(mean))

    if i == 0:
        for j in range(len(samples[i].numpy()[0])):
            plot_histogram_evidence(samples[i].numpy()[:, j], mean[j], torch.exp(logZ),
                           "Histogram for time step {} when n-particles = {} in 3.daphne".format(j + 1, n_particles[i]),
                           "time step", "3_daphne_{}_{}".format(n_particles[i], j + 1))

    else:
        variance = np.var(samples[i].numpy(), axis = 0)
        for j in range(len(samples[i].numpy()[0])):
            plot_histogram_variance_evidence(samples[i].numpy()[:, j], mean[j], variance[j], torch.exp(logZ),
                                    "Histogram for time step {} when n-particles = {} in 3.daphne".format(j + 1, n_particles[i]),
                                    "time step", "3_daphne_{}_{}".format(n_particles[i], j + 1))

mean_samples = torch.stack(mean_samples)
logZs = torch.stack(logZs)

for i in range(len(mean_samples[0])):
    plot_trace(mean_samples[:, i].numpy(), "Posterior expectation for time step {} 3.daphne".format(i + 1),
               "time step", "3_daphne_{}_trace_plot".format(i + 1))
Zs = torch.exp(torch.FloatTensor(logZs))
plot_trace(Zs.numpy(), "Marginal probability/evidence estimate for 3.daphne", "probability", "3_daphne_evidence")