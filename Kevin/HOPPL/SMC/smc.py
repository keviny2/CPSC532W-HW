from evaluator import evaluate
import torch


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
    logZ = torch.log(torch.mean(torch.exp(log_weights)))

    weights = torch.exp(log_weights)
    weights /= torch.sum(weights)  # normalize
    
    new_particles_indices = torch.multinomial(weights, len(particles), True)
    new_particles = []
    for index in new_particles_indices:
        # BUG: maybe the particle addresses aren't being carried over???
        new_particles.append(particles[index])

    return logZ, new_particles


def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    # 'prepare' each particle to be executed
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
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]:   # this checks if the calculation is done
                
                particles[i] = res[0]
                if i == 0:
                    done = True  # and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:  # this is the observe case
                #TODO: check particle addresses, and get weights and continuations
                particles[i] = res

                weights[i] = res[2]['logW']

                if i == 0:
                    address = res[2]['addr']
                else:
                    assert address == res[2]['addr']

        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, torch.tensor(weights))
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles




