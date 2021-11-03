from daphne import daphne
from bbvi import bbvi, calculate_mean, calculate_variance
import time


i = 1
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])

num_samples = 1000
L = 10
t = time.time()
rs, Ws, parameters = bbvi(num_samples, L, ast)
print("\nBlack Box Variational Inference for 1.daphne took %f seconds" % (time.time() - t))

mean = calculate_mean(rs, Ws, len(rs))
variance = calculate_variance(rs, Ws, mean)
print(mean)

mean
