from daphne import daphne
import torch
from bbvi import bbvi
from plot import plot_trace


i = 1
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])

rs, Ws = bbvi(10000, 10, ast)
mean = 0
for i in range(len(rs)):
    mean += rs[i] * Ws[i]

mean = mean / torch.sum(Ws)
