from daphne import daphne
import time
import torch
import numpy as np

from graph_based_sampling import sample_from_joint, deterministic_eval, evaluate
from importance_sampling import importance_Sampling, importance_Sampling_mean, importance_Sampling_variance
from mh_Gibbs import latent_observed, gibbs, joint_log_likelihood
from HMC import HMC, joint_log_likelihood_HMC
from plot import plot_histogram_IS, plot_histogram, plot_trace, plot_joint_loglik


i = 5
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])

_, variables_dict, _= sample_from_joint(graph)
vertices = graph[1]['V']
edges = graph[1]['A']
links = graph[1]['P']
returnings = graph[2]
latent, observed = latent_observed(links)

num_samples = 50000

'''IS'''
# t = time.time()
# samples_IS, weights = importance_Sampling(ast, num_samples)
# print("\nImportance sampling for 5.daphne took %f seconds" % (time.time() - t))
#
# mean_IS = importance_Sampling_mean(samples_IS, weights, num_samples)
# variance_IS = importance_Sampling_variance(samples_IS, weights, mean_IS)
#
# print('\nposterior mean of x in 5.daphne using Importance Sampling is {}'.format(mean_IS[0]))
# print('\nposterior mean of y in 5.daphne using Importance Sampling is {}'.format(mean_IS[1]))
# print('\nposterior variance of x in 5.daphne using Importance Sampling is {}'.format(variance_IS[0]))
# print('\nposterior variance of y in 5.daphne using Importance Sampling is {}'.format(variance_IS[1]))
#
# '''posterior distribution for IS:'''
# plot_histogram_IS(samples_IS[:, 0].numpy(), (weights / torch.sum(weights) * num_samples).numpy(), "Posterior distribution of x in 5.daphne using IS",
#                "x", "IS", "posterior_histogram_5_x_daphne")
# plot_histogram_IS(samples_IS[:, 1].numpy(), (weights / torch.sum(weights) * num_samples).numpy(), "Posterior distribution of y in 5.daphne using IS",
#                   "y", "IS", "posterior_histogram_5_y_daphne")

'''Gibbs:'''

t = time.time()
samples_Gibbs, variables_dict_set = gibbs(latent, variables_dict, num_samples, edges, links, returnings)
print("\nGibbs sampling for 5.daphne took %f seconds" % (time.time() - t))

mean = torch.mean(samples_Gibbs.float(), dim = 0)
variance = torch.var(samples_Gibbs.float(), dim = 0)
print('\nposterior mean of x in 5.daphne using Gibbs is {}'.format(mean[0]))
print('\nposterior mean of y in 5.daphne using Gibbs is {}'.format(mean[1]))
print('\nposterior variance of x in 5.daphne using Gibbs is {}'.format(variance[0]))
print('\nposterior variance of y in 5.daphne using Gibbs is {}'.format(variance[1]))

'''posterior distribution for Gibbs:'''
plot_histogram(samples_Gibbs[:, 0].numpy(), "Posterior distribution for x in 5.daphne using Gibbs",
               "x", "Gibbs", "posterior_histogram_5_x_daphne")
plot_histogram(samples_Gibbs[:, 1].numpy(), "Posterior distribution for y in 5.daphne using Gibbs",
               "y", "Gibbs", "posterior_histogram_5_y_daphne")


'''trace plot for Gibbs:'''
plot_trace(samples_Gibbs[:, 0].numpy(), "Trace plot for slope in 5.daphne using Gibbs", "x", "Gibbs",
           "trace_plot_5_x_daphne")
plot_trace(samples_Gibbs[:, 1].numpy(), "Trace plot for bias in 5.daphne using Gibbs", "y", "Gibbs",
           "trace_plot_5_y_daphne")

'''Joint_log_likelihood for Gibbs:'''
joint_latent_set1 = latent.copy()
joint_latent_set2 = latent.copy()
joint_latent_set1.append(returnings[1])
joint_latent_set2.append(returnings[2])
logPs1 = joint_log_likelihood(joint_latent_set1, links, variables_dict_set, num_samples)
logPs2 = joint_log_likelihood(joint_latent_set2, links, variables_dict_set, num_samples)
logPs3 = joint_log_likelihood(vertices, links, variables_dict_set, num_samples)
plot_joint_loglik(logPs1, "Joint Log-Likelihood for x for 5.daphne", "Gibbs", "joint_log_likelihood_5_x_daphne")
plot_joint_loglik(logPs2, "Joint Log-Likelihood for y for 5.daphne", "Gibbs", "joint_log_likelihood_5_y_daphne")
plot_joint_loglik(logPs3, "Joint Log-Likelihood for 5.daphne", "Gibbs", "joint_log_likelihood_5_daphne")

# '''HMC:'''
# num_samples = 10000
# latents_dict = {}
# observes_dict = {}
# for lat in latent:
#     latents_dict[lat] = variables_dict[lat]
#     latents_dict[lat].requires_grad = True
#
# for obs in observed:
#     observes_dict[obs] = variables_dict[obs]
#
# t = time.time()
# samples_HMC, variables_dict_set = HMC(latents_dict, num_samples, 10, 0.1, torch.eye(len(latent)), observes_dict, links)
# print("\nHamiltonian monte carlo for 5.daphne took %f seconds" % (time.time() - t))
#
# extract_samples = []
# for i in range(num_samples):
#     extract_samples.append(deterministic_eval(evaluate(returnings, samples_HMC[i])))
#
# extract_samples = torch.stack(extract_samples)
# mean = torch.mean(extract_samples, dim = 0)
# variance = torch.var(extract_samples, dim = 0)
# print('\nposterior mean of x in 5.daphne using HMC is {}'.format(mean[0]))
# print('\nposterior mean of y in 5.daphne using HMC is {}'.format(mean[1]))
# print('\nposterior variance of x in 5.daphne using HMC is {}'.format(variance[0]))
# print('\nposterior variance of y in 5.daphne using HMC is {}'.format(variance[1]))
#
# '''posterior distribution for HMC:'''
# plot_histogram(extract_samples.detach().numpy()[:, 0], "Posterior distribution for x in 5.daphne using HMC",
#                "slope", "HMC", "posterior_histogram_5_x_daphne")
# plot_histogram(extract_samples.detach().numpy()[:, 1], "Posterior distribution for y in 5.daphne using HMC",
#                "bias", "HMC", "posterior_histogram_5_y_daphne")
#
# '''trace plot for HMC:'''
# plot_trace(extract_samples.detach().numpy()[:, 0], "Trace plot for slope in 5.daphne using HMC", "x", "HMC",
#            "trace_plot_5_x_daphne")
# plot_trace(extract_samples.detach().numpy()[:, 1], "Trace plot for bias in 5.daphne using HMC", "y", "HMC",
#            "trace_plot_5_y_daphne")
#
# '''Joint_log_likelihood for HMC:'''
# joint_latent_set1 = latent.copy()
# joint_latent_set2 = latent.copy()
# joint_latent_set1.append(returnings[1])
# joint_latent_set2.append(returnings[2])
# logPs1 = joint_log_likelihood_HMC(joint_latent_set1, links, variables_dict_set, num_samples)
# logPs2 = joint_log_likelihood_HMC(joint_latent_set2, links, variables_dict_set, num_samples)
# logPs3 = joint_log_likelihood_HMC(vertices, links, variables_dict_set, num_samples)
#
# plot_joint_loglik(logPs1.numpy(), "Joint Log-Likelihood for x for 5.daphne using HMC", "HMC", "joint_log_likelihood_5_x_daphne")
# plot_joint_loglik(logPs2.numpy(), "Joint Log-Likelihood for y for 5.daphne using HMC", "HMC", "joint_log_likelihood_5_y_daphne")
# plot_joint_loglik(logPs3.numpy(), "Joint Log-Likelihood for 5.daphne using HMC", "HMC", "joint_log_likelihood_5_daphne")
