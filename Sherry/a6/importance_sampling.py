from evaluator import evaluate
import torch
import numpy as np
from daphne import daphne
import json
import sys



def get_IS_sample(exp):
    #init calc:
    output = lambda x: x
    res =  evaluate(exp, env=None)('addr_start', output)
    logW = 0
    #TODO : hint, "get_sample_from_prior" as a basis for your solution
    while type(res) is tuple:
        cont, args, sigma = res
        res = cont(*args)
        logW = sigma['logW']
    return logW, res


def sample_from_prior(exp):
    #init calc:
    output = lambda x: x #The output is the identity
    res =  evaluate(exp, env=None)('addr_start', output) #set up the initial call
    while type(res) is tuple: #if there are continuations, the res will be a tuple
        cont, args, sigma = res #res is contininuation, arguments, and a map, which you can use to pass back some additional stuff
        res = cont(*args) #call the continuation
    #when res is not a tuple, the calculation has finished
    return res


if __name__ == '__main__':

    for i in range(1,5):
        exp = daphne(['desugar-hoppl-cps', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a6/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        log_weights = []
        values = []
        for i in range(10000):
            logW, sample = get_IS_sample(exp)
            log_weights.append(logW)
            values.append(sample)

        log_weights = torch.tensor(log_weights)

        values = torch.stack(values)
        values = values.reshape((values.shape[0],values.size().numel()//values.shape[0]))
        log_Z = torch.logsumexp(log_weights,0) - torch.log(torch.tensor(log_weights.shape[0],dtype=float))

        log_norm_weights = log_weights - log_Z
        weights = torch.exp(log_norm_weights).detach().numpy()
        weighted_samples = (torch.exp(log_norm_weights).reshape((-1,1))*values.float()).detach().numpy()
    
        print('covariance: ', np.cov(values.float().detach().numpy(),rowvar=False, aweights=weights))    
        print('posterior mean:', weighted_samples.mean(axis=0))
