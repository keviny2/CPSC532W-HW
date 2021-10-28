from graph_based_sampling import sample_from_joint, deterministic_eval, evaluate
from importance_sampling import importance_Sampling
from mh_Gibbs import latent_observed, gibbs
from HMC import HMC
from daphne import daphne
import os
from matplotlib import pyplot as plt
import torch
import time
import numpy as np

def plot_trace(sample, title, ylabel, type, file_name):
    fig, ax = plt.subplots(figsize = (8,6))
    ax.plot(sample)
    ax.set_title(title)
    ax.set(ylabel = ylabel, xlabel = "Num_iterations")
    fname = os.path.join("figs", type, file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

def plot_joint_loglik(logPs, title, type, file_name):
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot(logPs)
    ax.set_title(title)
    ax.set(ylabel = "Log-Likelihood", xlabel = "Num_iterations")
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

def joint_log_likelihood(vertices, links, variables_dict):
    logP = 0
    for vertex in vertices:
        logP += deterministic_eval(evaluate(links[vertex][1], variables_dict)).log_prob(variables_dict[vertex])

    return logP


#TODO: 1.daphne

i = 1
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
print(ast)
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])

'''IS'''
t = time.time()
mean, variance, sample, weights = importance_Sampling(ast, 20000)
print("Importance sampling for 1.daphne took %f seconds" % (time.time() - t))

print('posterior mean of mu in 1.daphne using Importance Sampling is {}'.format(mean))
print('posterior variance of mu in 1.daphne using Importance Sampling is {}'.format(variance))


'''posterior distribution for IS'''
plot_histogram_IS(sample.numpy(), weights.numpy(), "Posterior distribution of mu in 1.daphne using IS",
               "mu", "IS", "posterior_histogram_1_daphne")

sample, variables_dict, _ = sample_from_joint(graph)
vertices = graph[1]['V']
edges = graph[1]['A']
links = graph[1]['P']
returnings = graph[2]
latent, observed = latent_observed(links)

'''Gibbs'''
t = time.time()
sample, logPs = gibbs(latent, variables_dict, 20000, vertices, edges, links, returnings)
print("Gibbs sampling for 1.daphne took %f seconds" % (time.time() - t))

mean = torch.mean(sample.float(), dim = 0)
variance = torch.var(sample.float(), dim = 0)
print('posterior mean of mu in 1.daphne using Gibbs is {}'.format(mean))
print('posterior variance of mu in 1.daphne using Gibbs is {}'.format(variance))

'''posterior distribution for Gibbs:'''
plot_histogram(sample.numpy(), "Posterior distribution for mu in 1.daphne using Gibbs",
               "mu", "Gibbs", "posterior_histogram_1_daphne")

'''trace plot for Gibbs:'''
plot_trace(sample.numpy(), "Trace plot for mu in 1.daphne using Gibbs", "mu", "Gibbs", "trace_plot_1_daphne")

'''Joint_log_likelihood for Gibbs:'''
plot_joint_loglik(logPs, "Joint Log-Likelihood for 1.daphne", "Gibbs", "joint_log_likelihood_1_daphne")

'''HMC'''
latents_dict = {}
observes_dict = {}
for lat in latent:
    latents_dict[lat] = variables_dict[lat]
    latents_dict[lat].requires_grad = True

for obs in observed:
    observes_dict[obs] = variables_dict[obs]

t = time.time()
samples, logPs = HMC(latents_dict, 20000, 10, 0.1, torch.eye(len(latent)), observes_dict, links, vertices)
print("Hamiltonian monte carlo for 1.daphne took %f seconds" % (time.time() - t))

extract_samples = []
for i in range(20000):
    extract_samples.append(samples[i][returnings])
extract_samples = torch.stack(extract_samples)
mean = torch.mean(extract_samples, dim = 0)
variance = torch.var(extract_samples, dim = 0)
print('posterior mean of mu in 1.daphne using HMC is {}'.format(mean))
print('posterior variance of mu in 1.daphne using HMC is {}'.format(variance))

'''posterior distribution for HMC:'''
plot_histogram(extract_samples.detach().numpy(), "Posterior distribution for mu in 1.daphne using HMC",
               "mu", "HMC", "posterior_histogram_1_daphne")

'''trace plot for HMC:'''
plot_trace(extract_samples.detach().numpy(), "Trace plot for mu in 1.daphne using HMC", "mu", "HMC", "trace_plot_1_daphne")

''' Joint_log_likelihood for HMC:'''
plot_joint_loglik(logPs.detach().numpy(), "Joint Log-Likelihood for 1.daphne", "HMC", "joint_log_likelihood_1_daphne")

#TODO: 2.daphne

i = 2
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
print(ast)
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])

'''IS'''
t = time.time()
mean, variance, sample, weights = importance_Sampling(ast, 50000)
print("Importance sampling for 2.daphne took %f seconds" % (time.time() - t))
covariance = np.cov(sample.numpy().T, aweights = weights.numpy())

print('posterior means of slope and bias in 2.daphne using Importance Sampling are {}'.format(mean))
print('posterior covariance of slopa and bias in 2.daphne using Importance Sampling is {}'.format(covariance))

'''posterior distribution for IS:'''
plot_histogram_IS(sample[:, 0].numpy(), weights.numpy(), "Posterior distribution of slope in 2.daphne using Importance Sampling",
               "slope", "IS", "posterior_histogram_slope_2_daphne")
plot_histogram_IS(sample[:, 1].numpy(), weights.numpy(), "Posterior distribution of slope in 2.daphne using Imporance Sampling",
               "bias", "IS", "posterior_histogram_bias_2_daphne")

sample, variables_dict, _ = sample_from_joint(graph)
vertices = graph[1]['V']
edges = graph[1]['A']
links = graph[1]['P']
returnings = graph[2]
latent, observed = latent_observed(links)

'''Gibbs:'''
t = time.time()
sample, logPs = gibbs(latent, variables_dict, 50000, vertices, edges, links, returnings)
print("Gibbs sampling for 2.daphne took %f seconds" % (time.time() - t))
mean = torch.mean(sample.float(), dim = 0)
covariance = np.cov(sample.numpy().T)
print('posterior means of slope and bias in 2.daphne using MH Gibbs Sampling is {}'.format(mean))
print('posterior covariance of slope and bias in 2.daphne using MH Gibbs Sampling is {}'.format(covariance))

'''posterior distribution for Gibbs:'''
plot_histogram(sample[:, 0].numpy(), "Posterior distribution for slope in 2.daphne using Gibbs",
               "slope", "Gibbs", "posterior_histogram_slope_2_daphne")
plot_histogram(sample[:, 1].numpy(), "Posterior distribution for bias in 2.daphne using Gibbs",
               "bias", "Gibbs", "posterior_histogram_bias_2_daphne")

'''trace plot for Gibbs:'''
plot_trace(sample[:, 0].numpy(), "Trace plot for slope in 2.daphne using Gibbs", "slope", "Gibbs", "trace_plot_slope_2_daphne")
plot_trace(sample[:, 1].numpy(), "Trace plot for bias in 2.daphne using Gibbs", "bias", "Gibbs", "trace_plot_bias_2_dapphne")
plot_joint_loglik(logPs, "Joint Log-Likelihood for 2.daphne", "Gibbs", "joint_log_likelihood_2_daphne")

'''HMC'''
latents_dict = {}
observes_dict = {}
for lat in latent:
    latents_dict[lat] = variables_dict[lat]
    latents_dict[lat].requires_grad = True

for obs in observed:
    observes_dict[obs] = variables_dict[obs]

t = time.time()
samples, logPs = HMC(latents_dict, 20000, 10, 0.1, torch.eye(len(latent)), observes_dict, links, vertices)
print("Hamiltonian monte carlo for 2.daphne took %f seconds" % (time.time() - t))

extract_samples = []
for i in range(20000):
    extract_samples.append(deterministic_eval(evaluate(returnings, samples[i])))

extract_samples = torch.stack(extract_samples)
mean = torch.mean(extract_samples, dim = 0)
covariance = np.cov(extract_samples.detach().numpy().T)
print('posterior means of slope and bias in 2.daphne using HMC is {}'.format(mean))
print('posterior covariance of slope and bias in 2.daphne using HMC is {}'.format(variance))

'''posterior distribution for HMC:'''
plot_histogram(extract_samples.detach().numpy()[:, 0], "Posterior distribution for slope in 2.daphne using HMC",
               "slope", "HMC", "posterior_histogram_slope_2_daphne")
plot_histogram(extract_samples.detach().numpy()[:, 1], "Posterior distribution for bias in 2.daphne using HMC",
               "bias", "HMC", "posterior_histogram_bias_2_daphne")


'''trace plot for HMC:'''
plot_trace(extract_samples.detach().numpy()[:, 0], "Trace plot for slope in 2.daphne using HMC", "slope", "HMC", "trace_plot_slope_2_daphne")
plot_trace(extract_samples.detach().numpy()[:, 1], "Trace plot for bias in 2.daphne using HMC", "bias", "HMC", "trace_plot_bias_2_daphne")

''' Joint_log_likelihood for HMC:'''
plot_joint_loglik(logPs.detach().numpy(), "Joint Log-Likelihood for 2.daphne", "HMC", "joint_log_likelihood_2_daphne")


#TODO: 3.daphne

i = 3
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])

'''IS'''
t = time.time()
mean, variance, sample, weights = importance_Sampling(ast, 20000)
print("Importance sampling for 3.daphne took %f seconds" % (time.time() - t))
print('posterior probability that the first and second datapoint are in the same cluster using Important sampling is {}'.format(mean))

'''posterior distribution for IS:'''
plot_histogram_IS(sample.numpy(), weights.numpy(), "Posterior distribution of the first and second datapoint in the same cluster in 3.daphne using IS",
               "indicator", "IS", "posterior_histogram_3_daphne")

sample, variables_dict, topological_sort = sample_from_joint(graph)
vertices = graph[1]['V']
edges = graph[1]['A']
links = graph[1]['P']
returnings = graph[2]
latent, observed = latent_observed(links)

'''Gibbs'''
t = time.time()
sample, logPs = gibbs(latent, variables_dict, 20000, vertices, edges, links, returnings)
print("Gibbs sampling for 3.daphne took %f seconds" % (time.time() - t))
mean = torch.mean(sample.float(), dim = 0)
print('posterior probability that the first and second datapoint are in the same cluster using MH Gibbs is {}'.format(mean))


'''posterior distribution for Gibbs:'''
plot_histogram(sample.numpy(), "Posterior distribution of the first and second datapoint in the same cluster in 3.daphne using Gibbs",
               "indicator", "Gibbs", "posterior_histogram_3_daphne")

'''Joint_log_like plot for Gibbs:'''
plot_joint_loglik(logPs, "Joint Log-Likelihood plot for 3.daphne", "Gibbs", "Joint_log_lik_3_daphne")


#TODO: 4.daphne

i = 4
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])

'''IS'''
t = time.time()
mean, variance, sample, weights = importance_Sampling(ast, 20000)
print("Sampling sampling for 4.daphne took %f seconds" % (time.time() - t))

print('posterior probability of raining in 4.da;hne using Importance sampling is {}'.format(mean))
# posterior distribution for IS:
plot_histogram_IS(sample.numpy(), weights.numpy(), "Posterior probablity of raining in 4.daphne using Important Sampling",
               "indicator", "IS", "posterior_histogram_4_daphne")

sample, variables_dict, _ = sample_from_joint(graph)
vertices = graph[1]['V']
edges = graph[1]['A']
links = graph[1]['P']
returnings = graph[2]
latent, observed = latent_observed(links)

'''Gibbs'''
t = time.time()
sample, logPs = gibbs(latent, variables_dict, 20000, vertices, edges, links, returnings)
print("Gibbs sampling for 4.daphne took %f seconds" % (time.time() - t))
mean = torch.mean(sample.float(), dim = 0)
print('posterior probability of raining in 4.daphne using MH Gibbs is {}'.format(mean))

'''posterior distribution for Gibbs:'''
plot_histogram(sample.numpy(), "Posterior probability of raining in 4.daphne using MH Gibbs",
               "indicator", "Gibbs", "posterior_histogram_4_daphne")

'''joint_log_like plot for Gibbs'''


plot_joint_loglik(logPs, "Joint Log-Likelihood plot for 4.daphne", "Gibbs", "Joint_log_lik_4_daphne")


# Todo: 5.daphne

i = 5
ast = daphne(['desugar', '-i',
              '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
graph = daphne(['graph','-i',
                '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])

mean, variance, sample, weights = importance_Sampling(ast, 10000)

'''posterior distribution for IS:'''
plot_histogram_IS(sample[:, 0].numpy(), weights.numpy(), "Posterior distribution of x in 5.daphne using IS",
               "x", "IS", "posterior_histogram_x_5_daphne")
plot_histogram_IS(sample[:, 1].numpy(), weights.numpy(), "Posterior distribution of y in 5.daphne using IS",
                  "y", "IS", "posterior_histogram_y_5_daphne")

sample, variables_dict, topological_sort= sample_from_joint(graph)
vertices = graph[1]['V']
edges = graph[1]['A']
links = graph[1]['P']
returnings = graph[2]
latent, observed = latent_observed(links)
sample, logPs = gibbs(latent, variables_dict, 10000, vertices, edges, links, returnings)
mean = torch.mean(sample.float(), dim = 0)
variance = torch.var(sample.float(), dim = 0)
print('posterior mean is {}'.format(mean))
print('posterior variance is {}'.format(variance))
# posterior distribution for Gibbs:
plot_histogram(sample.numpy(), "Posterior distribution of the indicator in 3.daphne using Gibbs",
               "indicator", "Gibbs", "posterior_histogram_3_daphne")

# joint_log_like plot for Gibbs:
plot_joint_loglik(logPs, "Joint Log-Likelihood plot for 3.daphne", "Gibbs", "Joint_log_lik_3_daphne")
