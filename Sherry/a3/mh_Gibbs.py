from graph_based_sampling import sample_from_joint, deterministic_eval, evaluate
from daphne import daphne
import torch
from torch import distributions as dist
from matplotlib import pyplot as plt
import os

def accept(x, cal_X, cal_X_prime, edges, links):
    q = deterministic_eval(evaluate(links[x][1], cal_X))
    q_prime = deterministic_eval(evaluate(links[x][1], cal_X_prime))
    log_alpha = q_prime.log_prob(cal_X[x]) - q.log_prob(cal_X_prime[x])
    free_variables = edges[x]
    for v in free_variables:
        if type(cal_X_prime[v]) is bool:
            cal_X_prime[v] = torch.tensor(float(cal_X_prime[v]))
        if type(cal_X[v]) is bool:
            cal_X[v] = torch.tensor(float(cal_X[v]))
        log_alpha += deterministic_eval(evaluate(links[v][1], cal_X_prime)).log_prob(cal_X_prime[v])
        log_alpha -= deterministic_eval(evaluate(links[v][1], cal_X)).log_prob(cal_X[v])

    alpha = torch.exp(log_alpha)
    return alpha

def gibbs_Step(X, cal_X, edges, links):
    for x in X:
        q = deterministic_eval(evaluate(links[x][1], cal_X))
        cal_X_prime = cal_X.copy()
        cal_X_prime[x] = q.sample()
        alpha = accept(x, cal_X, cal_X_prime, edges, links)
        u = dist.uniform.Uniform(0, 1).sample()
        if u < alpha:
            cal_X = cal_X_prime
    return cal_X

def gibbs(X, cal_X, num_samples, edges, links, returnings):
    samples = []
    for i in range(num_samples):
        cal_X_prime = gibbs_Step(X, cal_X, edges, links)
        sample = deterministic_eval(evaluate(returnings, cal_X_prime))
        samples.append(sample)
        cal_X = cal_X_prime
    samples = torch.stack(samples)
    return samples

def latent_observed(links):
    latent = []
    observed = []
    for variable in links:
        if links[variable][0] == 'sample*':
            latent.append(variable)

        elif links[variable][0] == 'observe*':
            observed.append(variable)
    return latent, observed



# samples = []
# for i in range(1, 5):
#     i = 1
#     graph = daphne(['graph', '-i',
#                     '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
#     print(graph)
#     sample, variables_dict = sample_from_joint(graph)
#     edges = graph[1]['A']
#     links = graph[1]['P']
#     returnings = graph[2]
#     latent, observed = latent_observed(links)
#     sample = gibbs(latent, variables_dict, 1000, edges, links, returnings)
#
#     sample = torch.stack(sample)
#     samples.append(sample)
#
#     mean = torch.mean(sample.float(), dim = 0)
#     variance = torch.var(sample.float(), dim = 0)
#     print('posterior mean is {}'.format(mean))
#     print('posterior variance is {}'.format(variance))
#
#
# samples

# fig, ax = plt.subplots(figsize = (8,6))
# ax.plot(samples[0])
# ax.set_title("Trace plot for 1.daphne")
# fname = os.path.join("figs", "gibbs_1")
# plt.savefig(fname)
#
# fig, ax = plt.subplots(figsize = (8,6))
# ax.plot(samples[1][:, 0])
# ax.set_title("Trace plot for slope for 2.daphne")
# ax.set(ylabel = "slope", xlabel = "Num_iterations")
# fname = os.path.join("figs", "gibbs_2_slope")
# plt.savefig(fname)
#
# fig,ax = plt.subplots(figsize = (8,6))
# ax.plot(samples[1][:, 1])
# ax.set_title("Trace plot for bias for 2.daphne")
# ax.set(ylabel = "bias", xlabel = "Num_iterations")
# fname = os.path.join("figs", "gibbs_2_bias")
# plt.savefig(fname)
#
# samples