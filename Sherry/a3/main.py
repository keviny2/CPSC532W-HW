from graph_based_sampling import sample_from_joint
from importance_sampling import importance_Sampling
from mh_Gibbs import latent_observed, gibbs
from daphne import daphne
import os
from matplotlib import pyplot as plt
import torch

def plot_trace(sample, title, ylabel, type, file_name):
    fig, ax = plt.subplots(figsize = (8,6))
    ax.plot(sample)
    ax.set_title(title)
    ax.set(ylabel = ylabel, xlabel = "Num_iterations")
    fname = os.path.join("figs", type, file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

def plot_histogram(sample, title, xlabel, type, file_name):
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.hist(sample)
    ax.set_title(title)
    ax.set(ylabel = "Weights", xlabel = xlabel)
    fname = os.path.join("figs", type, file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

def plot_histogram_IS(sample, weights, title, xlabel, type, file_name):
    fig, ax = plt.subplots(figsize = (8,6))
    ax.hist(sample, weights = weights)
    ax.set_title(title)
    ax.set(ylabel="Frequency", xlabel=xlabel)
    fname = os.path.join("figs", type, file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


#TODO: 1.daphne
# i = 1
# ast = daphne(['desugar', '-i',
#               '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
# graph = daphne(['graph','-i',
#                 '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
#
# mean, variance, sample, weights = importance_Sampling(ast, 10000)
# # posterior distribution for IS:
# plot_histogram_IS(sample.numpy(), weights.numpy(), "Posterior distribution of mu in 1.daphne using IS\n mean is {} \n variance is {}".format(mean, variance),
#                "mu", "IS", "posterior_histogram_1_daphne")
#
# sample, variables_dict = sample_from_joint(graph)
# edges = graph[1]['A']
# links = graph[1]['P']
# returnings = graph[2]
# latent, observed = latent_observed(links)
# sample = gibbs(latent, variables_dict, 1000, edges, links, returnings)
# mean = torch.mean(sample.float(), dim = 0)
# variance = torch.var(sample.float(), dim = 0)
# print('posterior mean is {}'.format(mean))
# print('posterior variance is {}'.format(variance))
# # posterior distribution for Gibbs:
# plot_histogram(sample.numpy(), "Posterior distribution for mu in 1.daphne using Gibbs \n mean is {} \n variance is {}".format(mean, variance),
#                "mu", "Gibbs", "posterior_histogram_1_daphne")
# # trace plot for Gibbs:
# plot_trace(sample.numpy(), "Trace plot for mu in 1.daphne using Gibbs", "mu", "Gibbs", "trace_plot_1_daphne")


#TODO: 2.daphne
# i = 2
# ast = daphne(['desugar', '-i',
#               '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
# graph = daphne(['graph','-i',
#                 '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
#
# mean, variance, sample, weights = importance_Sampling(ast, 10000)
# # posterior distribution for IS:
# plot_histogram_IS(sample[:, 0].numpy(), weights.numpy(), "Posterior distribution of slope in 2.daphne using IS",
#                "slope", "IS", "posterior_histogram_slope_2_daphne")
# plot_histogram_IS(sample[:, 1].numpy(), weights.numpy(), "Posterior distribution of slope in 2.daphne using IS",
#                "bias", "IS", "posterior_histogram_bias_2_daphne")
#
# sample, variables_dict = sample_from_joint(graph)
# edges = graph[1]['A']
# links = graph[1]['P']
# returnings = graph[2]
# latent, observed = latent_observed(links)
# sample = gibbs(latent, variables_dict, 10000, edges, links, returnings)
# mean = torch.mean(sample.float(), dim = 0)
# variance = torch.var(sample.float(), dim = 0)
# print('posterior mean is {}'.format(mean))
# print('posterior variance is {}'.format(variance))
# # posterior distribution for Gibbs:
# plot_histogram(sample[:, 0].numpy(), "Posterior distribution for slope in 1.daphne using Gibbs",
#                "slope", "Gibbs", "posterior_histogram_slope_2_daphne")
# plot_histogram(sample[:, 1].numpy(), "Posterior distribution for bias in 2.daphne using Gibbs",
#                "bias", "Gibbs", "posterior_histogram_bias_2_daphne")
# # trace plot for Gibbs:
# plot_trace(sample[:, 0].numpy(), "Trace plot for slope in 2.daphne using Gibbs", "slope", "Gibbs", "trace_plot_slope_2_daphne")
# plot_trace(sample[:, 1].numpy(), "Trace plot for bias in 2.daphne using Gibbs", "bias", "Gibbs", "trace_plot_bias_2_dapphne")


#TODO: 3.daphne
i = 3
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
print(ast)
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
print(graph)

mean, variance, sample, weights = importance_Sampling(ast, 100000)
# posterior distribution for IS:
plot_histogram_IS(sample.numpy(), weights.numpy(), "Posterior distribution of the indicator in 3.daphne using IS",
               "indicator", "IS", "posterior_histogram_3_daphne")

sample, variables_dict = sample_from_joint(graph)
edges = graph[1]['A']
links = graph[1]['P']
returnings = graph[2]
latent, observed = latent_observed(links)
sample = gibbs(latent, variables_dict, 10000, edges, links, returnings)
mean = torch.mean(sample.float(), dim = 0)
variance = torch.var(sample.float(), dim = 0)
print('posterior mean is {}'.format(mean))
print('posterior variance is {}'.format(variance))
# posterior distribution for Gibbs:
plot_histogram(sample.numpy(), "Posterior distribution of the indicator in 3.daphne using Gibbs",
               "indicator", "Gibbs", "posterior_histogram_3_daphne")


#TODO: 4.daphne
# i = 4
# ast = daphne(['desugar', '-i',
#               '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
# graph = daphne(['graph','-i',
#                 '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
#
# mean, variance, sample, weights = importance_Sampling(ast, 10000)
# # posterior distribution for IS:
# plot_histogram_IS(sample.numpy(), weights.numpy(), "Posterior distribution of the indicator in 4.daphne using IS",
#                "indicator", "IS", "posterior_histogram_4_daphne")
#
# sample, variables_dict = sample_from_joint(graph)
# edges = graph[1]['A']
# links = graph[1]['P']
# returnings = graph[2]
# latent, observed = latent_observed(links)
# sample = gibbs(latent, variables_dict, 10000, edges, links, returnings)
# mean = torch.mean(sample.float(), dim = 0)
# variance = torch.var(sample.float(), dim = 0)
# print('posterior mean is {}'.format(mean))
# print('posterior variance is {}'.format(variance))
# # posterior distribution for Gibbs:
# plot_histogram(sample.numpy(), "Posterior distribution of the indicator in 4.daphne using Gibbs",
#                "indicator", "Gibbs", "posterior_histogram_4_daphne")
