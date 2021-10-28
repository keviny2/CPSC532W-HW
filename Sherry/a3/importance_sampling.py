from evaluation_based_sampling import evaluate_program
import torch
import numpy as np

def importance_Sampling(ast, num_samples):
    samples = []
    Ws = []
    for i in range(num_samples):
        sample, sigma = evaluate_program(ast)
        samples.append(sample)
        W = torch.exp(sigma['logW'])
        Ws.append(W)

    samples = torch.stack(samples)
    Ws = torch.stack(Ws)

    return samples, Ws


def importance_Sampling_mean(samples, Ws, num_samples):
    unnormlized_mean = 0
    for i in range(num_samples):
        unnormlized_mean += samples[i] * Ws[i]
    mean = unnormlized_mean / sum(Ws)

    return mean

def importance_Sampling_variance(samples, Ws, mean):
    variance = torch.matmul(((samples - mean)**2).T, Ws) / sum(Ws)

    return variance

def importance_Sampling_Covariance(samples, Ws):
    covariance = np.cov(samples.numpy().T, aweights = Ws.numpy(), ddof = 0)

    return covariance
