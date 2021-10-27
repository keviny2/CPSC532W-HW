from graph_based_sampling import sample_from_joint, deterministic_eval, evaluate
import torch
from torch import distributions as dist


def accept(x, cal_X, cal_X_prime, edges, links):
    q = deterministic_eval(evaluate(links[x][1], cal_X))
    q_prime = deterministic_eval(evaluate(links[x][1], cal_X_prime))
    log_alpha = q_prime.log_prob(cal_X[x]) - q.log_prob(cal_X_prime[x])
    free_variables = edges[x].copy()
    free_variables.append(x)
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
    cal_Xs = []
    for i in range(num_samples):
        # print(i)
        cal_X_prime = gibbs_Step(X, cal_X, edges, links)
        sample = deterministic_eval(evaluate(returnings, cal_X_prime))
        samples.append(sample)
        cal_X = cal_X_prime
        cal_Xs.append(cal_X)
    samples = torch.stack(samples)
    return samples, cal_Xs

def joint_log_likelihood(vertices, links, variables_dict_set, num_samples):
    logPs = []
    for i in range(num_samples):
        logP = 0
        variables_dict = variables_dict_set[i]
        for vertex in vertices:
            logP += deterministic_eval(evaluate(links[vertex][1], variables_dict)).log_prob(variables_dict[vertex]).float()
        logPs.append(logP)
    return logPs

def latent_observed(links):
    latent = []
    observed = []
    for variable in links:
        if links[variable][0] == 'sample*':
            latent.append(variable)

        elif links[variable][0] == 'observe*':
            observed.append(variable)
    return latent, observed


