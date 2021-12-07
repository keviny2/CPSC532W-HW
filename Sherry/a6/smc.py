from evaluator import evaluate
import torch
from daphne import daphne
import numpy as np
import json
import sys



def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights):
    new_particles = []
    weights = torch.exp(torch.FloatTensor(log_weights))
    normalization_weights = weights / torch.sum(weights)
    samples = torch.multinomial(normalization_weights, len(particles), True)
    for sample in samples:
        new_particles.append(particles[sample])
    logZ = torch.log(torch.mean(weights))

    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.


        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    address = ''
    while not done:
        print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): #Even though this can be parallelized, we run it serially

            if i % 100 == 0:
                print(i)

            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                #TODO: check particle addresses, and get weights and continuations
                particles[i] = res
                if i == 0:
                    address = res[2]['addr']
                else:
                    test_address = res[2]['addr']
                    if test_address != address:
                        raise RuntimeError("Failed SMC, different addresses")
                logW = res[2]['logW']
                weights[i] = logW

        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


# if __name__ == '__main__':
#
#     for i in range(1,5):
#         exp = daphne(['desugar-hoppl-cps', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a6/programs/{}.daphne'.format(i)])
#         n_particles = np.array([1, 10, 100, 1000, 10000, 100000])
#         samples = []#TODO
#         for i in range(len(n_particles)):
#             logZ, particles = SMC(n_particles[i], exp)
#
#             print('logZ: ', logZ)
#
#             values = torch.stack(particles)
#             samples.append(values)


