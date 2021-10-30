from daphne import daphne
import torch
from evaluation_based_sampling import evaluate_program


i = 1
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])
evaluate_program(ast)