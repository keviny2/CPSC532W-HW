import torch
from graph_based_sampling import sample_from_joint_with_sorted
import numpy as np
from copy import deepcopy

def optimizer_step(hat_g, sigma):
    for v in list(hat_g.keys()):
        lambda_v = sigma['Q'][v].Parameters()

        try:
            len(lambda_v[0])
            lambda_v[0].grad = -torch.FloatTensor(hat_g[v])
        except:
            for i in range(len(lambda_v)):
                lambda_v[i].grad = -torch.tensor(float(hat_g[v][i]))


        sigma['lambda'][v].step()
        sigma['lambda'][v].zero_grad()


    return sigma['Q']

def elbo_gradients(Gs, logW):

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
            num_params = len(Gs[l][v])
            if v in list(Gs[l].keys()):
                F = Gs[l][v] * logW[l]

            else:
                F = torch.zeros(num_params)
                Gs[l][v] = torch.zeros(num_params)

            Fs.append(F)
            gs.append(Gs[l][v])

        Fs = torch.stack(Fs)
        gs = torch.stack(gs)

        hat_b = []
        for i in range(np.shape(Fs)[1]):
            covariance = np.cov(Fs[:, i].numpy(), gs[:, i].numpy())
            hat_b.append(torch.nan_to_num(torch.tensor(float(covariance[0, 1] / covariance[1, 1])), 1))

        hat_b = torch.stack(hat_b)
        dict_hat_g[v] = torch.sum(Fs - hat_b * gs, dim=0) / L

    return dict_hat_g


def bbvi(T, L, graph, topological_orderings):
    rs = []
    Ws = []
    bbvi_losses = []
    sigma = {}
    sigma['Q'] = {}
    sigma['lambda'] = {}
    # sigma['optimizer'] = torch.optim.Adam
    sigma['optimizer'] = torch.optim.SGD

    for t in range(T):
        print(t)

        bbvi_loss = []
        logWs = []
        Gs = []

        for l in range(L):
            sigma['logW'] = 0
            sigma['G'] = {}
            x = 0
            r, sigma = sample_from_joint_with_sorted(deepcopy(graph), topological_orderings, x, sigma)
            G = deepcopy(sigma['G'])
            Gs.append(G)
            rs.append(r)
            logWs.append(sigma['logW'].clone().detach())
            bbvi_loss.append(sigma['logW'].clone().detach())
            Ws.append(torch.exp(sigma['logW'].clone().detach()))

        hat_g = elbo_gradients(Gs, logWs)
        bbvi_loss = torch.mean(torch.stack(bbvi_loss).clone().detach())
        print(bbvi_loss)
        bbvi_losses.append(bbvi_loss)
        sigma['Q'] = optimizer_step(hat_g, sigma)

    Ws = torch.stack(Ws)
    return rs, Ws, bbvi_losses

def calculate_mean(rs, Ws, num_samples):
    unnormlized_mean = 0
    for i in range(num_samples):
        unnormlized_mean += rs[i] * Ws[i]
    mean = unnormlized_mean / sum(Ws)

    return mean

def calculate_variance(rs, Ws, mean):
    variance = torch.matmul(((rs - mean)**2).T, Ws) / sum(Ws)

    return variance
