from daphne import daphne
import time
import torch

from graph_based_sampling import sample_from_joint
from importance_sampling import importance_Sampling, importance_Sampling_mean, importance_Sampling_variance
from mh_Gibbs import latent_observed, gibbs
from plot import plot_histogram_IS, plot_histogram


i = 3
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])


_, variables_dict, _ = sample_from_joint(graph)
vertices = graph[1]['V']
edges = graph[1]['A']
links = graph[1]['P']
returnings = graph[2]
latent, observed = latent_observed(links)


'''IS'''
t = time.time()
samples_IS, weights = importance_Sampling(ast, 20000)
print("\nImportance sampling for 3.daphne took %f seconds" % (time.time() - t))

mean_IS = importance_Sampling_mean(samples_IS, weights, 20000)
variance_IS = importance_Sampling_variance(samples_IS, weights, mean_IS)
print('\nposterior probability (mean) that the first and second datapoint are in the same cluster using Important sampling is {}'.format(mean_IS))
print('\nposterior probability (variance) that the first and second datapoint are in the same cluster using Important sampling is {}'.format(variance_IS))

'''posterior distribution for IS:'''
plot_histogram_IS(samples_IS.numpy(), weights.numpy(), "Posterior distribution of the first and second datapoint in the same cluster in 3.daphne using IS",
               "indicator", "IS", "posterior_histogram_3_daphne")


'''Gibbs'''
t = time.time()
sample, variables_dict_set = gibbs(latent, variables_dict, 20000, edges, links, returnings)
print("\nGibbs sampling for 3.daphne took %f seconds" % (time.time() - t))
mean = torch.mean(sample.float(), dim = 0)
print('\nposterior probability that the first and second datapoint are in the same cluster using MH Gibbs is {}'.format(mean))


'''posterior distribution for Gibbs:'''
plot_histogram(sample.numpy(), "Posterior distribution of the first and second datapoint in the same cluster in 3.daphne using Gibbs",
               "indicator", "Gibbs", "posterior_histogram_3_daphne")


