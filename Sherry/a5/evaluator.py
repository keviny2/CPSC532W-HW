from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist
import torch

def standard_env():
    "An environment with some Scheme standard procedures."
    env = pmap(penv)
    env = env.update({'alpha' : ''})

    return env


class Env(dict):
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def get(self, var):
        return self[var] if (var in self) else self.outer.get(var)


class Procedure(object):
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env

    def __call__(self, *args):
        return evaluate(self.body, Env(self.parms, args, self.env))


def evaluate(exp, env=None):#TODO: add sigma, or something
    if env is None:
        env = standard_env()
    if isinstance(exp, str):
        if env.get(exp) is not None:
            return env.get(exp)
        return exp
    elif not isinstance(exp, list):
        return torch.tensor(float(exp))
    op, *args = exp
    if op == 'if':
        (test, conseq, alt) = args
        exp = (conseq if evaluate(test, env) else alt)
        return evaluate(exp, env)
    elif op == 'fn':
        parms, body = args
        if len(parms) > 1:
            parms = parms[1:]
        else:
            parms = []
        return Procedure(parms, body, env)

    elif op == 'sample':
        dist = evaluate(args[1], env)
        value = dist.sample()
        return value

    elif op == 'observe':
        return evaluate(args[-1], env)


    else:
        proc = evaluate(op, env)
        vals = [evaluate(arg, env) for arg in args[1:]]
        return proc(*vals)





def get_stream(exp):
    while True:
        yield evaluate(exp)([""])


def run_deterministic_tests():
    
    for i in range(1,14):
        exp = daphne(['desugar-hoppl', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)([""])
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))

        print('FOPPL Tests passed')

    for i in range(1,13):
        exp = daphne(['desugar-hoppl', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)([""])
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples = 1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



# if __name__ == '__main__':
    
    # run_deterministic_tests()
    # run_probabilistic_tests()
    

    # for i in range(1,4):
    #     print(i)
    #     exp = daphne(['desugar-hoppl', '-i', '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a5/programs/{}.daphne'.format(i)])
    #     print('\n\n\nSample of prior of program {}:'.format(i))
    #     print(evaluate(exp)([""]))
