from daphne import daphne
import time
import torch

from graph_based_sampling import sample_from_joint
from importance_sampling import importance_Sampling, importance_Sampling_mean, importance_Sampling_variance
from mh_Gibbs import latent_observed, gibbs, joint_log_likelihood
from HMC import HMC, joint_log_likelihood_HMC
from plot import plot_histogram_IS, plot_histogram, plot_trace, plot_joint_loglik


i = 1
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
print(ast)
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
print("\nImportance sampling for 1.daphne took %f seconds" % (time.time() - t))

mean_IS = importance_Sampling_mean(samples_IS, weights, num_samples)
variance_IS = importance_Sampling_variance(samples_IS, weights, mean_IS)


print('\nposterior mean of mu in 1.daphne using Importance Sampling is {}'.format(mean_IS))
print('\nposterior variance of mu in 1.daphne using Importance Sampling is {}'.format(variance_IS))


'''posterior distribution for IS'''
plot_histogram_IS(samples_IS.numpy(), (weights / torch.sum(weights) * num_samples).numpy(), "Posterior distribution of mu in 1.daphne using IS",
               "mu", "IS", "posterior_histogram_1_daphne")


'''Gibbs'''
t = time.time()
sample_gibbs, variables_dict_set = gibbs(latent, variables_dict, num_samples, edges, links, returnings)
print("\nGibbs sampling for 1.daphne took %f seconds" % (time.time() - t))

mean = torch.mean(sample_gibbs.float(), dim = 0)
variance = torch.var(sample_gibbs.float(), dim = 0)
print('\nposterior mean of mu in 1.daphne using Gibbs is {}'.format(mean))
print('\nposterior variance of mu in 1.daphne using Gibbs is {}'.format(variance))

'''posterior distribution for Gibbs:'''
plot_histogram(sample_gibbs.numpy(), "Posterior distribution for mu in 1.daphne using Gibbs",
               "mu", "Gibbs", "posterior_histogram_1_daphne")

'''trace plot for Gibbs:'''
plot_trace(sample_gibbs.numpy(), "Trace plot for mu in 1.daphne using Gibbs", "mu", "Gibbs", "trace_plot_1_daphne")

'''Joint_log_likelihood for Gibbs:'''
joint_latent_set = latent.copy()
joint_latent_set.append(returnings)
logPs = joint_log_likelihood(joint_latent_set, links, variables_dict_set, num_samples)
plot_joint_loglik(logPs, "Joint Log-Likelihood for 1.daphne using Gibbs", "Gibbs", "joint_log_likelihood_1_daphne")

'''HMC'''
latents_dict = {}
observes_dict = {}
for lat in latent:
    latents_dict[lat] = variables_dict[lat]
    latents_dict[lat].requires_grad = True

for obs in observed:
    observes_dict[obs] = variables_dict[obs]

t = time.time()
samples_HMC, variables_dict_set = HMC(latents_dict, num_samples, 10, 0.1, torch.eye(len(latent)), observes_dict, links)
print("\nHamiltonian monte carlo for 1.daphne took %f seconds" % (time.time() - t))

extract_samples = []
for i in range(num_samples):
    extract_samples.append(samples_HMC[i][returnings])
extract_samples = torch.stack(extract_samples)
mean = torch.mean(extract_samples, dim = 0)
variance = torch.var(extract_samples, dim = 0)
print('\nposterior mean of mu in 1.daphne using HMC is {}'.format(mean))
print('\nposterior variance of mu in 1.daphne using HMC is {}'.format(variance))

'''posterior distribution for HMC:'''
plot_histogram(extract_samples.detach().numpy(), "Posterior distribution for mu in 1.daphne using HMC",
               "mu", "HMC", "posterior_histogram_1_daphne")

'''trace plot for HMC:'''
plot_trace(extract_samples.detach().numpy(), "Trace plot for mu in 1.daphne using HMC", "mu", "HMC", "trace_plot_1_daphne")

''' Joint_log_likelihood for HMC:'''
joint_latent_set = latent.copy()
joint_latent_set.append(returnings)
logPs = joint_log_likelihood_HMC(joint_latent_set, links, variables_dict_set, num_samples)
plot_joint_loglik(logPs.detach().numpy(), "Joint Log-Likelihood for 1.daphne using HMC", "HMC", "joint_log_likelihood_1_daphne")
