import distributions
import torch
from evaluation_based_sampling import evaluate_program_with_sigma
import numpy as np

def optimizer_step(Q, hat_g):
    for v in hat_g:
        lambda_v = Q[v].Parameters()
        optimizer = torch.optim.Adam(lambda_v, lr=1e-2)
        #Todo: call logP.backward() first but don't know what logP is
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

    for v in domains:
        Fs = []
        Gs = []
        for l in range(L):
            if v in Gs[l]:
                F = Gs[l][v] * logWs[l]
                Fs.append(F)
                Gs.append(Gs[l][v])
            else:
                Fs.append(0)
                Gs.append(0)
        Fs = torch.stack(Fs)
        Gs = torch.stack(Gs)
        hat_b = torch.sum(torch.tensor(np.cov(Fs, Gs))) / torch.sum(torch.var(Gs, dim = 0))
        hat_g = torch.sum(Fs - torch.matmul(hat_b.T, Gs)) / L

        return hat_g


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
            G = sigma['G']
            logW = sigma['logW']
            Gs.append(G)
            logWs.append(logW)
            results.append([r, logW])

        hat_g = elbo_gradients(Gs, logWs)
        sigma['Q'] = optimizer_step(sigma['Q'], hat_g)

    return results
