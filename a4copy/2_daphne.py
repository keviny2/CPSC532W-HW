from daphne import daphne
import torch
from bbvi import bbvi, calculate_mean, calculate_variance
from plot import plot_trace
from bbvi import calculate_mean
import time


i = 2
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])

num_samples = 2000
L = 10
t = time.time()
rs, Ws = bbvi(num_samples, L, ast)
print("\nBlack Box Variational Inference for 2.daphne took %f seconds" % (time.time() - t))

mean = calculate_mean(rs, Ws, len(rs))
print("Posterior means for slope and bias for 2.daphne are {}, {}".format(mean[0], mean[1]))