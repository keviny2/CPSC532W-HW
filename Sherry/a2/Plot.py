import numpy as np
from matplotlib import pyplot as plt
from evaluation_based_sampling import evaluate_program
from graph_based_sampling import sample_from_joint
from daphne import daphne
import os

ast_set = []
graph_set = []
for i in range(1, 5):
    ast = daphne(['desugar', '-i',
                    '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/{}.daphne'.format(i)])
    graph = daphne(['graph', '-i',
                    '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/{}.daphne'.format(i)])
    ast_set.append(ast)
    graph_set.append(graph)

ast1 = ast_set[0]
evaluation_1 = np.zeros(1000)
for i in range(1000):
    evaluation_1[i] = evaluate_program(ast1)

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(evaluation_1)
# mean = round(evaluation_1.mean(), 3)
# ax.set_title("Histogram for mean in 1.daphne from Evaluation-based Sampling\n Marginal expectation is %f" %mean)
# ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
# fname = os.path.join("figs", "evaluation_1")
# plt.savefig(fname)
# print("\nFigure saved as '%s'" % fname)
#
# ast2 = ast_set[1]
# evaluation_2 = np.zeros((1000, 2))
# for i in range(1000):
#     evaluation_2[i] = evaluate_program(ast2)
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(evaluation_2[:, 0])
# mean = round((evaluation_2[:, 0]).mean(), 3)
# ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
# ax.set_title("Histogram for slope in 2.daphne from Evaluation-based Sampling\n Marginal expectation is %f" %mean)
# fname = os.path.join("figs", "evaluation_2_1")
# plt.savefig(fname)
# print("\nFigure saved as '%s'" % fname)
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist((evaluation_2[:, 1]).mean())
# mean = round((evaluation_2[:, 1]).mean(), 3)
# ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
# ax.set_title("Histogram for bias in 2.daphne from Evaluation-based Sampling\n Marginal expectation is %f" %mean)
# fname = os.path.join("figs", "evaluation_2_2")
# plt.savefig(fname)
# print("\nFigure saved as '%s'" % fname)

ast3 = ast_set[2]
evaluation_3 = np.ones((1000, 17))
for i in range(1000):
    evaluation_3[i] = evaluate_program(ast3)

trans = np.zeros((3, 3))

for i in range(len(evaluation_3)):
    for j in range(1, len(evaluation_3[1])):
        trans[int(evaluation_3[i][j-1])][int(evaluation_3[i][j])] += 1

stationary = np.sum(trans, axis = 0)/np.sum(trans)

for i in range(1, 18):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(evaluation_3[:, i - 1])
    ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
    ax.set_title("Histogram for time step {} in 3.daphne from Evaluation-based Sampling".format(i))
    fname = os.path.join("figs", "evaluation_3_%s" %i)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

# ast4 = ast_set[3]
# evaluation_4 = np.ones((1000, 4))
# for i in range(1000):
#     temp = evaluate_program(ast4)
#     for j in range(4):
#         evaluation_4[i][j] = temp[j].mean()

# name = ['W', 'b', 'W', 'b']
# for i in range(1, 3):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.hist((evaluation_4[:, i - 1]).mean())
#     mean = round((evaluation_4[:, i - 1]).mean(), 3)
#     ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
#     ax.set_title("Histogram for {}_0 in 4.daphne from Evaluation-based Sampling\n Marginal expectation is {}".format(name[i - 1], mean))
#     fname = os.path.join("figs", "evaluation_4_%s" % i)
#     plt.savefig(fname)
#     print("\nFigure saved as '%s'" % fname)
#
# for i in range(3, 5):
#     fig, ax = plt.subplots(figsize=(8,6))
#     ax.hist(evaluation_4[:, i - 1])
#     mean = round((evaluation_4[:, i- 1]).mean(), 3)
#     ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
#     ax.set_title("Histogram for {}_1 in 4.daphne from Evaluation-based Sampling\n Marginal expectation is {}".format(name[i - 1], mean))
#     fname = os.path.join("figs", "evaluation_4_%s" %i)
#     plt.savefig(fname)
#     print("\nFigure saved as '%s'" % fname)
#
#
# graph1 = graph_set[0]
# graph_1 = np.zeros(1000)
# for i in range(1000):
#     graph_1[i] = sample_from_joint(graph1)
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(graph_1)
# mean = round(graph_1.mean(), 3)
# ax.set_title("Histogram for mean in 1.daphne from Graph-based Sampling\n Marginal expectation is %f" %mean)
# ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
# fname = os.path.join("figs", "graph_1")
# plt.savefig(fname)
# print("\nFigure saved as '%s'" % fname)
#
# graph2 = graph_set[1]
# graph_2 = np.zeros((1000, 2))
# for i in range(1000):
#     graph_2[i] = sample_from_joint(graph2)
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(graph_2[:, 0])
# mean = round((graph_2[:, 0]).mean(), 3)
# ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
# ax.set_title("Histogram for slope in 2.daphne from Graph-based Sampling\n Marginal expectation is %f" %mean)
# fname = os.path.join("figs", "graph_2_1")
# plt.savefig(fname)
# print("\nFigure saved as '%s'" % fname)
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(graph_2[:, 1])
# mean = round((graph_2[:, 1]).mean(), 3)
# ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
# ax.set_title("Histogram for bias in 2.daphne from Graph-based Sampling\n Marginal expectation is %f" %mean)
# fname = os.path.join("figs", "graph_2_2")
# plt.savefig(fname)
# print("\nFigure saved as '%s'" % fname)


graph3 = graph_set[2]
graph_3 = np.ones((1000, 17))
for i in range(1000):
    graph_3[i] = sample_from_joint(graph3)

trans = np.zeros((3, 3))

for i in range(len(graph_3)):
    for j in range(1, len(graph_3[1])):
        trans[int(graph_3[i][j-1])][int(graph_3[i][j])] += 1

stationary = np.sum(trans, axis = 0)/np.sum(trans)


for i in range(1, 18):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(graph_3[:, i - 1])
    mean = round((graph_3[:, i - 1]).mean(), 3)
    ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
    ax.set_title("Histogram for state {} in 3.daphne from Graph-based Sampling".format(i))
    fname = os.path.join("figs", "graph_3_%s" %i)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

# graph4 = graph_set[3]
# graph_4 = np.ones((1000, 4))
# for i in range(1000):
#     temp = sample_from_joint(graph4)
#     for j in range(4):
#         graph_4[i][j] = temp[j].mean()
#
# name = ['W', 'b', 'W', 'b']
# for i in range(1, 3):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.hist(graph_4[:, i - 1])
#     mean = round((graph_4[:, i - 1]).mean(), 3)
#     ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
#     ax.set_title("Histogram for {}_0 in 4.daphne from Graph-based Sampling\n Marginal expectation is {}".format(name[i - 1], mean))
#     fname = os.path.join("figs", "graph_4_%s" % i)
#     plt.savefig(fname)
#     print("\nFigure saved as '%s'" % fname)
#
# for i in range(3, 5):
#     fig, ax = plt.subplots(figsize=(8,6))
#     ax.hist(graph_4[:, i - 1])
#     print((graph_4[:, i - 1]).mean())
#     ax.set(ylabel='Frequency', xlabel='Samplings from the prior distribution')
#     ax.set_title("Histogram for {}_1 in 4.daphne from Graph-based Sampling\n Marginal expectation is {}".format(name[i-1], mean))
#     fname = os.path.join("figs", "graph_4_%s" %i)
#     plt.savefig(fname)
#     print("\nFigure saved as '%s'" % fname)