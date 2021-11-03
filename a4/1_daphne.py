from daphne import daphne
from bbvi import bbvi, calculate_mean
import time
from graph_based_sampling import sample_from_joint


i = 1
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])
_, _, topological_sort = sample_from_joint(graph)

num_samples = 10000
L = 10
t = time.time()
rs, Ws = bbvi(num_samples, L, graph, topological_sort)
print("\nBlack Box Variational Inference for 1.daphne took %f seconds" % (time.time() - t))

mean = calculate_mean(rs, Ws, len(rs))
print(mean)


