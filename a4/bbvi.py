import distributions
import torch
from evaluation_based_sampling import evaluate_program_with_sigma
import numpy as np

def optimizer_step(Q, hat_g):
    for v in hat_g:
        lambda_v = Q[v].Parameters()
        optimizer = torch.optim.Adam(lambda_v, lr=1e-2)
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
                Fs.append(0)
                gs.append(0)
        Fs = torch.stack(Fs).detach()
        gs = torch.stack(gs).detach()

        hat_b = torch.sum(torch.tensor(np.cov(Fs.numpy(), gs.numpy()))) / torch.sum(torch.var(gs, dim = 0))
        hat_g = torch.sum(Fs - hat_b * gs) / L


def bbvi(T, L, ast):
    sigma = {}
    sigma['logW'] = 0
    sigma['Q'] = {}
    sigma['G'] = {}
    results = []
    for t in range(T):
        Gs = []
        logWs = []

        for l in range(L):
            r, sigma = evaluate_program_with_sigma(ast, sigma)
            G = sigma['G'].copy()
            logW = sigma['logW']
            Gs.append(G)
            logWs.append(logW)
            results.append([r, logW])

        hat_g = elbo_gradients(Gs, logWs)
        sigma['Q'] = optimizer_step(sigma['Q'], hat_g)

    return results
