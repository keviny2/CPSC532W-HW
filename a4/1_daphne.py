from daphne import daphne
import torch
from evaluation_based_sampling import evaluate_program
from bbvi import bbvi


i = 1
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a4/programs/{}.daphne'.format(i)])

results = bbvi(150, 5, ast)
