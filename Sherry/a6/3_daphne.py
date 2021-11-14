from smc import SMC
from daphne import daphne
import torch

i = 3
exp = daphne(['desugar-hoppl-cps', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a6/programs/{}.daphne'.format(i)])
n_particles = [1, 10, 100, 1000, 10000, 100000]
samples = []

for i in range(len(n_particles)):
    print(n_particles[i])
    logZ, particles = SMC(n_particles[i], exp)

    print('logZ: ', logZ)

    values = torch.stack(particles)
    samples.append(values)

samples