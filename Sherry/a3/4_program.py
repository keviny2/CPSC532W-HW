from daphne import daphne
import time
import torch

from graph_based_sampling import sample_from_joint
from importance_sampling import importance_Sampling, importance_Sampling_mean, importance_Sampling_variance
from mh_Gibbs import latent_observed, gibbs
from plot import plot_histogram_IS, plot_histogram

i = 4
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])

num_samples = 50000
_, variables_dict, _ = sample_from_joint(graph)
vertices = graph[1]['V']
edges = graph[1]['A']
links = graph[1]['P']
returnings = graph[2]
latent, observed = latent_observed(links)

'''IS'''
t = time.time()
samples_IS, weights = importance_Sampling(ast, num_samples)
print("\nImportance sampling for 4.daphne took %f seconds" % (time.time() - t))

mean_IS = importance_Sampling_mean(samples_IS, weights, num_samples)
variance_IS = importance_Sampling_variance(samples_IS, weights, mean_IS)
print('\nposterior probability of raining in 4.daphne using Importance sampling is {}'.format(mean_IS))


'''posterior distribution for IS'''
plot_histogram_IS(samples_IS.numpy(),  (weights / torch.sum(weights) * num_samples).numpy(), "Posterior probablity of raining in 4.daphne using Important Sampling",
               "raining", "IS", "posterior_histogram_4_daphne")


'''Gibbs'''

t = time.time()
sample_gibbs, variables_dict_set = gibbs(latent, variables_dict, num_samples, edges, links, returnings)
print("\nGibbs sampling for 4.daphne took %f seconds" % (time.time() - t))
mean = torch.mean(sample_gibbs.float(), dim = 0)
print('\nposterior probability of raining in 4.daphne using MH Gibbs is {}'.format(mean))

'''posterior distribution for Gibbs:'''
plot_histogram(sample_gibbs.numpy(), "Posterior probability of raining in 4.daphne using MH Gibbs",
               "raining", "Gibbs", "posterior_histogram_4_daphne")



