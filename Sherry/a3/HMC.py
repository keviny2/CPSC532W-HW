import torch
from graph_based_sampling import deterministic_eval, evaluate
from torch import distributions as dist
import copy

def H(cal_X, cal_Y, links, R, M):
    return U(cal_X, cal_Y, links) + 0.5 * torch.matmul(R.T, torch.matmul(torch.inverse(M), R))

def U(cal_X, cal_Y, links):
    logP = 0
    for observe in cal_Y:
        logP += deterministic_eval(evaluate(links[observe][1], {**cal_X, **cal_Y})).log_prob(cal_Y[observe])

    for latent in cal_X:
        logP += deterministic_eval(evaluate(links[latent][1], {**cal_X, **cal_Y})).log_prob(cal_X[latent])


    return -logP

def grad_U(cal_X, cal_Y, links):
    U_cal_X = U(cal_X, cal_Y, links)
    U_cal_X.backward()
    grads = torch.zeros(len(cal_X))
    i = 0
    for latent in cal_X:
        grads[i] = (cal_X[latent].grad)
        if torch.isinf(grads[i]):
            grads[i] = torch.tensor(2**63 - 1)
        if torch.isneginf(grads[i]):
            grads[i] = torch.tensor(-2**63)
        i += 1
    return grads


def leapfrog(cal_X0, cal_R0, T, epsilon, cal_Y, links):
    cal_X = {0 : cal_X0}
    cal_R = {0 : cal_R0}
    cal_R[0.5] = cal_R0 - 0.5 * epsilon * grad_U(cal_X0, cal_Y, links)
    for t in range(1, T):
        cal_X[t] = add_dict(cal_X[t - 1], epsilon * cal_R[t - 0.5])
        cal_R[t + 0.5] = cal_R[t - 0.5] - epsilon * grad_U(cal_X[t], cal_Y, links)
    cal_X[T] = add_dict(cal_X[T - 1], epsilon * cal_R[T - 0.5])
    cal_R[T] = cal_R[T - 0.5] - 0.5 * epsilon * grad_U(cal_X[T], cal_Y, links)
    return cal_X[T], cal_R[T]


def add_dict(cal_inside, cal_R):
    dicts = {}
    keys = list(cal_inside.keys())
    for i in range(len(keys)):
        dicts[keys[i]] = cal_inside[keys[i]].detach() + cal_R[i]
        dicts[keys[i]].requires_grad = True

    return dicts


def HMC(cal_X, S, T, epsilon, M, cal_Y, links):
    samples = []
    cal_XYs = []
    for s in range(1, S + 1):
        R = dist.multivariate_normal.MultivariateNormal(torch.zeros(len(cal_X)), M).sample()
        cal_X_prime, R_prime = leapfrog(copy.deepcopy(cal_X), R, T, epsilon, cal_Y, links)
        u = dist.uniform.Uniform(0, 1).sample()
        if u < torch.exp(-H(cal_X_prime, cal_Y, links, R_prime, M) + H(cal_X, cal_Y, links, R, M)):
            cal_X = cal_X_prime
        samples.append(cal_X)
        cal_XYs.append({**cal_X, **cal_Y})

    return samples, cal_XYs

def joint_log_likelihood_HMC(vertices, links, variables_dict_set, num_samples):
    logPs = []
    for i in range(num_samples):
        logP = 0
        variables_dict = variables_dict_set[i]
        for vertex in vertices:
            logP += deterministic_eval(evaluate(links[vertex][1], variables_dict)).log_prob(variables_dict[vertex])
        logPs.append(logP)
    logPs = torch.stack(logPs).detach()

    return logPs




