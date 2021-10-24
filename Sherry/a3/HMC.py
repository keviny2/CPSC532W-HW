import torch
from graph_based_sampling import deterministic_eval, evaluate
from torch import distributions as dist

def H(cal_X, cal_Y, links, R, M):
    return  U(cal_X, cal_Y, links) + 0.5 * torch.matmul(R.T, torch.matmul(torch.inverse(M), R))

def U(cal_X, cal_Y, links):
    logP = 0
    for observe in cal_X:
        logP += deterministic_eval(evaluate(links[observe][1], {cal_X, cal_Y})).log_prob(cal_X[observe])

    return -logP

def grad_U(cal_X, cal_Y, links):
    U_cal_X = U(cal_X, cal_Y, links)
    torch.backward(U_cal_X)
    grad = torch.tensor([])
    for observe in cal_X:
        torch.cat((grad, cal_X[observe].grad), 1)

    return grad

def leapfrog(X0, R0, T, epsilon, Y, links):
    X = {0 : X0}
    R = {0 : R0}
    R[0.5] = R0 - 0.5 * epsilon * grad_U(X0, Y, links)
    for t in range(1, T):
        X[t] = X[t - 1] + epsilon * R[t - 0.5]
        R[t + 0.5] = R[t - 0.5] - epsilon * grad_U(X[t], Y, links)
    X[T] = X[T - 1] + epsilon * R[T - 0.5]
    R[T] = R0 - 0.5 * epsilon * grad_U(X[T - 0.5], Y, links)
    return X[T], R[T]


def HMC(X, S, T, epsilon, M, Y, links):
    samples = []
    for s in range(1, S + 1):
        R = dist.normal.Normal(0, M).sample()
        X_prime, R_prime = leapfrog(X[s - 1], R, T, epsilon, Y, links)
        u = dist.uniform.Uniform(0, 1).sample()
        if u < torch.exp(-H(X_prime, Y, links, R_prime, M) + H(X[s - 1], Y, links, R[s - 1], M)):
            X[s] = X_prime
        samples.append(X_prime)

    return samples
