from daphne import daphne
from bbvi import bbvi, calculate_mean, calculate_variance
import time


i = 3
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])

num_samples = 10000
L = 10
t = time.time()
rs, Ws = bbvi(num_samples, L, ast)
print("\nBlack Box Variational Inference for 3.daphne took %f seconds" % (time.time() - t))

mean = calculate_mean(rs, Ws, len(rs))