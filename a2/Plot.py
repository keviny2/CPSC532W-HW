import numpy as np
from matplotlib import pyplot as plt
from evaluation_based_sampling import evaluate_program
from daphne import daphne
import os

# array = np.ones(1000)
# ast1 = daphne(['desugar', '-i',
#               '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/1.daphne'])
# ast1 = [['let', ['mu', ['sample', ['normal', 1, ['sqrt', 5]]]], ['let', ['sigma', ['sqrt', 2]], ['let', ['lik', ['normal', 'mu', 'sigma']], ['let', ['dontcare0', ['observe', 'lik', 8]], ['let', ['dontcare1', ['observe', 'lik', 9]], 'mu']]]]]]
#
# for i in range(1000):
#     array[i] = evaluate_program(ast1)
#
# plt.figure()
# plt.hist(array)
# fname = os.path.join("figs", "evaluation_1")
# plt.savefig(fname)
# print("\nFigure saved as '%s'" % fname)

#ast2 = daphne(['desugar', '-i',
#               '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/2.daphne'])

# ast2 = [['defn', 'observe-data', ['_', 'data', 'slope', 'bias'], ['let', ['xn', ['first', 'data']], ['let', ['yn', ['second', 'data']], ['let', ['zn', ['+', ['*', 'slope', 'xn'], 'bias']], ['let', ['dontcare0', ['observe', ['normal', 'zn', 1.0], 'yn']], ['rest', ['rest', 'data']]]]]]], ['let', ['slope', ['sample', ['normal', 0.0, 10.0]]], ['let', ['bias', ['sample', ['normal', 0.0, 10.0]]], ['let', ['data', ['vector', 1.0, 2.1, 2.0, 3.9, 3.0, 5.3, 4.0, 7.7, 5.0, 10.2, 6.0, 12.9]], ['let', ['dontcare1', ['let', ['a2', 'slope'], ['let', ['a3', 'bias'], ['let', ['acc4', ['observe-data', 0, 'data', 'a2', 'a3']], ['let', ['acc5', ['observe-data', 1, 'acc4', 'a2', 'a3']], ['let', ['acc6', ['observe-data', 2, 'acc5', 'a2', 'a3']], ['let', ['acc7', ['observe-data', 3, 'acc6', 'a2', 'a3']], ['let', ['acc8', ['observe-data', 4, 'acc7', 'a2', 'a3']], ['let', ['acc9', ['observe-data', 5, 'acc8', 'a2', 'a3']], 'acc9']]]]]]]]], ['vector', 'slope', 'bias']]]]]]
#
# array = np.ones((1000, 2))
# for i in range(1000):
#     array[i] = evaluate_program(ast2)
#
# plt.figure()
# plt.hist(array[:, 0])
# fname = os.path.join("figs", "evaluation_2_1")
# plt.savefig(fname)
# print("\nFigure saved as '%s'" % fname)
#
# plt.figure()
# plt.hist(array[:, 1])
# fname = os.path.join("figs", "evaluation_2_2")
# plt.savefig(fname)
# print("\nFigure saved as '%s'" % fname)

#ast3 = daphne(['desugar', '-i',
#               '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/3.daphne'])

# ast3 = [['defn', 'hmm-step', ['t', 'states', 'data', 'trans-dists', 'likes'], ['let', ['z', ['sample', ['get', 'trans-dists', ['last', 'states']]]], ['let', ['dontcare0', ['observe', ['get', 'likes', 'z'], ['get', 'data', 't']]], ['append', 'states', 'z']]]], ['let', ['data', ['vector', 0.9, 0.8, 0.7, 0.0, -0.025, -5.0, -2.0, -0.1, 0.0, 0.13, 0.45, 6, 0.2, 0.3, -1, -1]], ['let', ['trans-dists', ['vector', ['discrete', ['vector', 0.1, 0.5, 0.4]], ['discrete', ['vector', 0.2, 0.2, 0.6]], ['discrete', ['vector', 0.15, 0.15, 0.7]]]], ['let', ['likes', ['vector', ['normal', -1.0, 1.0], ['normal', 1.0, 1.0], ['normal', 0.0, 1.0]]], ['let', ['states', ['vector', ['sample', ['discrete', ['vector', 0.33, 0.33, 0.34]]]]], ['let', ['a1', 'data'], ['let', ['a2', 'trans-dists'], ['let', ['a3', 'likes'], ['let', ['acc4', ['hmm-step', 0, 'states', 'a1', 'a2', 'a3']], ['let', ['acc5', ['hmm-step', 1, 'acc4', 'a1', 'a2', 'a3']], ['let', ['acc6', ['hmm-step', 2, 'acc5', 'a1', 'a2', 'a3']], ['let', ['acc7', ['hmm-step', 3, 'acc6', 'a1', 'a2', 'a3']], ['let', ['acc8', ['hmm-step', 4, 'acc7', 'a1', 'a2', 'a3']], ['let', ['acc9', ['hmm-step', 5, 'acc8', 'a1', 'a2', 'a3']], ['let', ['acc10', ['hmm-step', 6, 'acc9', 'a1', 'a2', 'a3']], ['let', ['acc11', ['hmm-step', 7, 'acc10', 'a1', 'a2', 'a3']], ['let', ['acc12', ['hmm-step', 8, 'acc11', 'a1', 'a2', 'a3']], ['let', ['acc13', ['hmm-step', 9, 'acc12', 'a1', 'a2', 'a3']], ['let', ['acc14', ['hmm-step', 10, 'acc13', 'a1', 'a2', 'a3']], ['let', ['acc15', ['hmm-step', 11, 'acc14', 'a1', 'a2', 'a3']], ['let', ['acc16', ['hmm-step', 12, 'acc15', 'a1', 'a2', 'a3']], ['let', ['acc17', ['hmm-step', 13, 'acc16', 'a1', 'a2', 'a3']], ['let', ['acc18', ['hmm-step', 14, 'acc17', 'a1', 'a2', 'a3']], ['let', ['acc19', ['hmm-step', 15, 'acc18', 'a1', 'a2', 'a3']], 'acc19']]]]]]]]]]]]]]]]]]]]]]]]
# array = np.ones((1000, 17))
# for i in range(1000):
#     array[i] = evaluate_program(ast3)
#
# for i in range(1, 18):
#     plt.figure()
#     plt.hist(array[:, i - 1])
#     fname = os.path.join("figs", "evaluation_3_%s" %i)
#     plt.savefig(fname)
#     print("\nFigure saved as '%s'" % fname)


# ast4 = [['let', ['weight-prior', ['normal', 0, 1]], ['let', ['W_0', ['vector', ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']]]], ['let', ['W_1', ['vector', ['vector', ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior']]]], ['let', ['W_2', ['vector', ['vector', ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior'], ['sample', 'weight-prior']]]], ['let', ['b_0', ['vector', ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']]]], ['let', ['b_1', ['vector', ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']], ['vector', ['sample', 'weight-prior']]]], ['let', ['b_2', ['vector', ['vector', ['sample', 'weight-prior']]]], ['let', ['x', ['mat-transpose', ['vector', ['vector', 1], ['vector', 2], ['vector', 3], ['vector', 4], ['vector', 5]]]], ['let', ['y', ['vector', ['vector', 1], ['vector', 4], ['vector', 9], ['vector', 16], ['vector', 25]]], ['let', ['h_0', ['mat-tanh', ['mat-add', ['mat-mul', 'W_0', 'x'], ['mat-repmat', 'b_0', 1, 5]]]], ['let', ['h_1', ['mat-tanh', ['mat-add', ['mat-mul', 'W_1', 'h_0'], ['mat-repmat', 'b_1', 1, 5]]]], ['let', ['mu', ['mat-transpose', ['mat-tanh', ['mat-add', ['mat-mul', 'W_2', 'h_1'], ['mat-repmat', 'b_2', 1, 5]]]]], ['let', ['dontcare0', ['vector', ['let', ['y_r', ['get', 'y', 0]], ['let', ['mu_r', ['get', 'mu', 0]], ['vector', ['let', ['y_rc', ['get', 'y_r', 0]], ['let', ['mu_rc', ['get', 'mu_r', 0]], ['observe', ['normal', 'mu_rc', 1], 'y_rc']]]]]], ['let', ['y_r', ['get', 'y', 1]], ['let', ['mu_r', ['get', 'mu', 1]], ['vector', ['let', ['y_rc', ['get', 'y_r', 0]], ['let', ['mu_rc', ['get', 'mu_r', 0]], ['observe', ['normal', 'mu_rc', 1], 'y_rc']]]]]], ['let', ['y_r', ['get', 'y', 2]], ['let', ['mu_r', ['get', 'mu', 2]], ['vector', ['let', ['y_rc', ['get', 'y_r', 0]], ['let', ['mu_rc', ['get', 'mu_r', 0]], ['observe', ['normal', 'mu_rc', 1], 'y_rc']]]]]], ['let', ['y_r', ['get', 'y', 3]], ['let', ['mu_r', ['get', 'mu', 3]], ['vector', ['let', ['y_rc', ['get', 'y_r', 0]], ['let', ['mu_rc', ['get', 'mu_r', 0]], ['observe', ['normal', 'mu_rc', 1], 'y_rc']]]]]], ['let', ['y_r', ['get', 'y', 4]], ['let', ['mu_r', ['get', 'mu', 4]], ['vector', ['let', ['y_rc', ['get', 'y_r', 0]], ['let', ['mu_rc', ['get', 'mu_r', 0]], ['observe', ['normal', 'mu_rc', 1], 'y_rc']]]]]]]], ['vector', 'W_0', 'b_0', 'W_1', 'b_1']]]]]]]]]]]]]]]
