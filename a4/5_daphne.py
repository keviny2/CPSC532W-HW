from daphne import daphne
from bbvi import bbvi, calculate_mean
import time
from graph_based_sampling import sample_from_joint
from plot import plot_histogram_bbvi, plot_trace, plot_heatmap
import torch

i = 5
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])
_, _, topological_sort = sample_from_joint(graph)

