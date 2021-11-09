from evaluator import evaluate
from daphne import daphne
import matplotlib.pyplot as plt
import torch
import os
import time
import sys
import threading
import numpy as np


num_samples = 10000

def plot_histogram(sample, title, xlabel, file_name):
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.hist(sample)
    ax.set_title(title)
    ax.set(ylabel = "Frequency", xlabel = xlabel)
    fname = os.path.join("figs", file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

def sample(samples1, num_samples):
    i = 1
    exp = daphne(['desugar-hoppl', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a5/programs/{}.daphne'.format(i)])
    t = time.time()
    for i in range(num_samples):
        samples1.append(evaluate(exp))
        print(i)

    print("\n running 1.daphne took %f seconds" %(time.time() - t))
    samples1 = torch.stack(samples1).numpy()
    mean = np.mean(samples1)
    variance = np.var(samples1)
    print("\n Prior expectation of mu in 1.daphne is {}".format(mean))
    print("\n Prior variance of mu in 1.daphne is {}".format(variance))

    plot_histogram(samples1, "Histogram for mu in 1.daphne", "mu", "1_daphne")


sys.setrecursionlimit(100000)
threading.stack_size(64*1024*64)
thread = threading.Thread(target=sample([], num_samples))

# i = 1
# exp = daphne(['desugar-hoppl', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a5/programs/{}.daphne'.format(i)])
#
# samples1 = []
# t = time.time()
# while i < num_samples:
#
#     samples1.append(evaluate(exp))
#     i += 1
#
# print("\n running 1.daphne took %f seconds" %(time.time() - t))

# samples1 = torch.stack(samples1).numpy()
# plot_histogram(samples1, "Histogram for mu in 1.daphne", "mu", "1_daphne")

i = 2
exp = daphne(['desugar-hoppl', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a5/programs/{}.daphne'.format(i)])

samples2 = []

t = time.time()
while i < num_samples:
    samples2.append(evaluate(exp))
    i += 1
print("\n running 2.daphne took %f seconds" %(time.time() - t))

samples2 = torch.stack(samples2).numpy()
mean = np.mean(samples2)
variance = np.var(samples2)
print("\n Prior expectation of mu in 1.daphne is {}".format(mean))
print("\n Prior variance of mu in 1.daphne is {}".format(variance))
plot_histogram(samples2, "Histogram for mu in 2.daphne", "mu", "2_daphne")


i = 3
exp = daphne(['desugar-hoppl', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a5/programs/{}.daphne'.format(i)])

samples3 = []

t = time.time()
while i < num_samples:
    samples3.append(evaluate(exp))
    i += 1
print("\n running 3.daphne took %f seconds" %(time.time() - t))

samples3 = torch.stack(samples3)
for i in range(len(samples3[1])):
    print(i)
    plot_histogram(samples3[:, i].numpy(), "Histogram for time step {} in 3.daphne".format(i + 1), "time step", "4_daphne_{}".format(i + 1))

