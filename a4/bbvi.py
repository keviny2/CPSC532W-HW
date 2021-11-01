import distributions
import torch
from evaluation_based_sampling import evaluate_program_with_sigma
import numpy as np
import copy

def optimizer_step(Q, hat_g):
    for v in hat_g:
        lambda_v = Q[v].Parameters()
        optimizer = torch.optim.Adam(lambda_v, lr=1e-2)
        for i in range(5):
            nlp = -Q[v].log_prob(hat_g[v])
            nlp.backward()
            optimizer.step()
            optimizer.zero_grad()

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

        hat_b = torch.sum(torch.tensor(np.cov(Fs.numpy(), gs.numpy()))) / torch.sum(torch.var(gs, dim = 0))
        hat_g = torch.sum(Fs - hat_b * gs) / L
        dict_hat_g[v]  = hat_g

    return dict_hat_g

def bbvi(T, L, ast):
    # sigma = {}
    # sigma['logW'] = 0
    # sigma['Q'] = {}
    # sigma['G'] = {}
    rs = []
    Ws = []
    # sigma_prime = {}
    # sigma_prime['G'] = {}
    for t in range(T):
        print(t)
        sigma = {}
        sigma['logW'] = 0
        sigma['Q'] = {}
        sigma['G'] = {}
        sigma_prime = {}
        sigma_prime['G'] = {}

        logWs = []
        Gs = []
        x = 0
        for l in range(L):
            r, sigma = evaluate_program_with_sigma(ast, sigma, x)
            x += 1
            dict = {}
            for key in set(sigma['G']) - set(sigma_prime['G']):
                dict[key] = sigma['G'][key]
            Gs.append(dict)
            rs.append(r)
            logW = sigma['logW']
            logWs.append(logW)
            Ws.append(torch.exp(logW))
            sigma_prime['G'] = copy.deepcopy(sigma['G'])
        hat_g = elbo_gradients(Gs, logWs)
        sigma['Q'] = optimizer_step(sigma['Q'], hat_g)

    rs = torch.stack(rs)
    Ws = torch.stack(Ws).detach()
    return rs, Ws
