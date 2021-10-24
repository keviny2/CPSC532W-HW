from evaluation_based_sampling import evaluate_program
import torch


def importance_Sampling(ast, num_samples):
    samples = []
    Ws = []
    unnormalized_mean = 0
    for i in range(num_samples):
        sample, sigma = evaluate_program(ast)
        samples.append(sample)
        W = torch.exp(sigma['logW'])
        Ws.append(W)
        unnormalized_mean += sample * W

    mean = unnormalized_mean / sum(Ws)
    samples = torch.stack(samples)
    Ws = torch.stack(Ws)
    variance = torch.matmul(((samples - mean)**2).T, Ws) / sum(Ws)
    print('posterior mean is {}'.format(mean))
    print('posterior variance is {}'.format(variance))
    # Ws = Ws / torch.sum(Ws) * num_samples
    return mean, variance, samples, Ws



