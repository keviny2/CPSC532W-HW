from daphne import daphne
import time
import torch
import numpy as np

from graph_based_sampling import sample_from_joint, deterministic_eval, evaluate
from importance_sampling import importance_Sampling, importance_Sampling_mean, importance_Sampling_Covariance
from mh_Gibbs import latent_observed, gibbs, joint_log_likelihood
from HMC import HMC, joint_log_likelihood_HMC
from plot import plot_histogram_IS, plot_histogram, plot_trace, plot_joint_loglik

num_samples = 50000
i = 2
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
print(ast)
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
samples_IS, weights = importance_Sampling(ast, num_samples)
print("\nImportance sampling for 2.daphne took %f seconds" % (time.time() - t))

mean_IS = importance_Sampling_mean(samples_IS, weights, num_samples)
covariance_IS = importance_Sampling_Covariance(samples_IS, weights)


print('\nposterior means of slope and bias in 2.daphne using Importance Sampling are {}'.format(mean_IS))
print('\nposterior covariance of slopa and bias in 2.daphne using Importance Sampling is {}'.format(covariance_IS))

'''posterior distribution for IS:'''
plot_histogram_IS(samples_IS[:, 0].numpy(),  (weights / torch.sum(weights) * num_samples).numpy(), "Posterior distribution of slope in 2.daphne using Importance Sampling",
               "slope", "IS", "posterior_histogram_2_slope__daphne")
plot_histogram_IS(samples_IS[:, 1].numpy(),  (weights / torch.sum(weights) * num_samples).numpy(), "Posterior distribution of slope in 2.daphne using Imporance Sampling",
               "bias", "IS", "posterior_histogram_2_bias__daphne")


'''Gibbs:'''

t = time.time()
sample_gibbs, variables_dict_set = gibbs(latent, variables_dict, num_samples, edges, links, returnings)
print("\nGibbs sampling for 2.daphne took %f seconds" % (time.time() - t))

mean = torch.mean(sample_gibbs.float(), dim = 0)
covariance = np.cov(sample_gibbs.numpy().T)
print('\nposterior means of slope and bias in 2.daphne using MH Gibbs Sampling is {}'.format(mean))
print('\nposterior covariance of slope and bias in 2.daphne using MH Gibbs Sampling is {}'.format(covariance))

'''posterior distribution for Gibbs:'''
plot_histogram(sample_gibbs[:, 0].numpy(), "Posterior distribution for slope in 2.daphne using Gibbs",
               "slope", "Gibbs", "posterior_histogram_2_slope_daphne")
plot_histogram(sample_gibbs[:, 1].numpy(), "Posterior distribution for bias in 2.daphne using Gibbs",
               "bias", "Gibbs", "posterior_histogram_2_bias_daphne")

'''trace plot for Gibbs:'''
plot_trace(sample_gibbs[:, 0].numpy(), "Trace plot for slope in 2.daphne using Gibbs", "slope", "Gibbs", "trace_plot_2_slope_daphne")
plot_trace(sample_gibbs[:, 1].numpy(), "Trace plot for bias in 2.daphne using Gibbs", "bias", "Gibbs", "trace_plot_2_bias_daphne")

'''joint_loglik for Gibbs'''
joint_latent_set1 = latent.copy()
joint_latent_set2 = latent.copy()
joint_latent_set1.append(returnings[1])
joint_latent_set2.append(returnings[2])
logPs1 = joint_log_likelihood(joint_latent_set1, links, variables_dict_set, num_samples)
logPs2 = joint_log_likelihood(joint_latent_set2, links, variables_dict_set, num_samples)
logPs3 = joint_log_likelihood(vertices, links, variables_dict_set, num_samples)
plot_joint_loglik(logPs1, "Joint Log-Likelihood for slope for 2.daphne using Gibbs", "Gibbs", "joint_log_likelihood_2_slope_daphne")
plot_joint_loglik(logPs2, "Joint Log-Likelihood for bias for 2.daphne using Gibbs", "Gibbs", "joint_log_likelihood_2_bias_daphne")
plot_joint_loglik(logPs3, "Joint Log-Likelihood for 2.daphne using Gibbs", "Gibbs", "joint_log_likelihood_2_daphne")

'''HMC'''

num_samples = 15000
latents_dict = {}
observes_dict = {}
for lat in latent:
    latents_dict[lat] = variables_dict[lat]
    latents_dict[lat].requires_grad = True

for obs in observed:
    observes_dict[obs] = variables_dict[obs]

t = time.time()
samples_HMC, variables_dict_set = HMC(latents_dict, num_samples, 10, 0.1, torch.eye(len(latent)), observes_dict, links)
print("\nHamiltonian monte carlo for 2.daphne took %f seconds" % (time.time() - t))

extract_samples = []
for i in range(num_samples):
    extract_samples.append(deterministic_eval(evaluate(returnings, samples_HMC[i])))

extract_samples = torch.stack(extract_samples)
mean = torch.mean(extract_samples, dim = 0)
covariance = np.cov(extract_samples.detach().numpy().T)
print('\nposterior means of slope and bias in 2.daphne using HMC is {}'.format(mean))
print('\nposterior covariance of slope and bias in 2.daphne using HMC is {}'.format(covariance))

'''posterior distribution for HMC:'''
plot_histogram(extract_samples.detach().numpy()[:, 0], "Posterior distribution for slope in 2.daphne using HMC",
               "slope", "HMC", "posterior_histogram_slope_2_daphne")
plot_histogram(extract_samples.detach().numpy()[:, 1], "Posterior distribution for bias in 2.daphne using HMC",
               "bias", "HMC", "posterior_histogram_bias_2_daphne")


'''trace plot for HMC:'''
plot_trace(extract_samples.detach().numpy()[:, 0], "Trace plot for slope in 2.daphne using HMC", "slope", "HMC", "trace_plot_2_slope_daphne")
plot_trace(extract_samples.detach().numpy()[:, 1], "Trace plot for bias in 2.daphne using HMC", "bias", "HMC", "trace_plot_2_bias_daphne")

''' Joint_log_likelihood for HMC:'''
joint_latent_set1 = latent.copy()
joint_latent_set2 = latent.copy()
joint_latent_set1.append(returnings[1])
joint_latent_set2.append(returnings[2])
logPs1 = joint_log_likelihood_HMC(joint_latent_set1, links, variables_dict_set, num_samples)
logPs2 = joint_log_likelihood_HMC(joint_latent_set2, links, variables_dict_set, num_samples)
logPs3 = joint_log_likelihood_HMC(vertices, links, variables_dict_set, num_samples)
plot_joint_loglik(logPs1.numpy(), "Joint Log-Likelihood for slope in 2.daphne using HMC", "HMC", "joint_log_likelihood_2_slope_daphne")
plot_joint_loglik(logPs2.numpy(), "Joint Log-Likelihood for bias in 2.daphne using HMC", "HMC", "joint_log_likelihood_2_bias_daphne")
plot_joint_loglik(logPs3.numpy(), "Joint Log-Likelihood for 2.daphne using HMC", "HMC", "joint_log_likelihood_2_daphne")
