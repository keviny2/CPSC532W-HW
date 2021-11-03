import torch
from graph_based_sampling import sample_from_joint_with_sorted
import numpy as np
from copy import deepcopy

def optimizer_step(Q, hat_g):
    parameters = []
    for v in hat_g:
        lambda_v = Q[v].Parameters()
        optimizer = torch.optim.Adam(lambda_v, lr=1e-2)

        for i in range(len(lambda_v)):
            lambda_v[i].grad = -hat_g[v][i]
        optimizer.step()
        optimizer.zero_grad()
        parameters.append(Q[v].Parameters())
    return Q

def elbo_gradients(Gs, logWs):
    L = len(Gs)
    domains = []
    for l in range(L):
        keys = Gs[l].keys()
        for key in keys:
            if key not in domains:
                domains.append(key)

    dict_hat_g = {}
    for v in domains:
        Fs = []
        gs = []
        for l in range(L):
            if v in Gs[l]:
                F = Gs[l][v] * logWs[l]
                Fs.append(F)
                gs.append(Gs[l][v])
            else:
                Fs.append(torch.zeros(2))
                gs.append(torch.zeros(2))
        Fs = torch.stack(Fs).detach()
        gs = torch.stack(gs).detach()

        hat_gs = []
        for i in range(np.shape(Fs)[1]):
            hat_b = np.sum(np.cov(Fs.numpy()[:, i], gs.numpy()[:, i])) / torch.sum(torch.var(gs, dim = 0)[i])
            hat_g = torch.sum(Fs[:, i] - hat_b * gs[:, i]) / L
            hat_gs.append(hat_g)
        dict_hat_g[v]  = hat_gs

    return dict_hat_g

def bbvi(T, L, graph, topological_orderings):
    rs = []
    Ws = []

    for t in range(T):
        print(t)
        sigma = {}
        sigma['Q'] = {}
        sigma['G'] = {}
        sigma_prime = {}
        sigma_prime['G'] = {}

        logWs = []
        Gs = []

        for l in range(L):
            sigma['logW'] = 0
            x = 0
            r, sigma = sample_from_joint_with_sorted(deepcopy(graph), topological_orderings, x, sigma)
            Gs.append(sigma['G'].copy())
            rs.append(r)
            logW = sigma['logW']
            logWs.append(logW)
            Ws.append(torch.exp(logW))
        hat_g = elbo_gradients(Gs, logWs)
        sigma['Q'] = optimizer_step(sigma['Q'], hat_g)


    rs = torch.stack(rs)
    Ws = torch.stack(Ws).detach()
    return rs, Ws

def calculate_mean(rs, Ws, num_samples):
    unnormlized_mean = 0
    for i in range(num_samples):
        unnormlized_mean += rs[i] * Ws[i]
    mean = unnormlized_mean / sum(Ws)

    return mean

def calculate_variance(rs, Ws, mean):
    variance = torch.matmul(((rs - mean)**2).T, Ws) / sum(Ws)

    return variance