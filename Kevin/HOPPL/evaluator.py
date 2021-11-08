from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth

from utils import save_ast, load_ast
from pyrsistent import pmap, plist, PClass, field
import torch


class Env(PClass):
    local_map = field()
    outer = field()

    def find(self, var):
        "Find the innermost Env where var appears."
        return self.local_map if (var in self.local_map) else self.outer.find(var)


class Procedure(PClass):
    "A user-defined Scheme procedure."
    parms = field()
    body = field()
    env = field()

    def __call__(self, *args):
        return evaluate(self.body, Env(local_map=pmap(dict(zip(self.parms, args))), outer=self.env))


def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env(local_map=pmap(penv), outer=None)
    return env


def evaluate(exp, env=None, sig=None): #TODO: add sigma, or something

    # evaluate the wrapper fn
    if env is None:
        env = standard_env()
        proc = evaluate(exp, env)
        return proc('')  # list with an empty string for pmap constructor for __call__ function in Procedure class

    # simply return constants and variables
    if isinstance(exp, str):         # variable reference
        try:
            return env.find(exp)[exp]
        except AttributeError:  # not found in environment, so just be a string primitive
            return exp
    elif not isinstance(exp, list):  # constant
        return torch.tensor(exp)

    op, *args = exp
    if op == 'sample':
        # TODO: maybe use sigma to pass around sample and observe stuff
        pass
    if op == 'observe':
        pass
    if op == 'if':             # conditional
        (test, conseq, alt) = args
        exp = (conseq if evaluate(test, env) else alt)
        return evaluate(exp, env)
    elif op == 'define':         # definition
        (symbol, exp) = args
        env[symbol] = evaluate(exp, env)
    elif op == 'fn':             # procedure
        (parms, body) = args
        return Procedure(parms=parms[1:], body=body, env=env)  # the first element will be the address which we can ignore?
    else:                        # procedure call
        proc = evaluate(op, env, sig)
        vals = [evaluate(arg, env) for arg in args[1:]]  # NOTE: I think we can skip the address here b/c we'll catch
                                                            # the sample and observe case earlier
        return proc(*vals)


def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():

    for i in range(1,14):

        # exp = daphne(['desugar-hoppl', '-i', '../CPSC532W-HW/Kevin/HOPPL/programs/tests/deterministic/test_{}.daphne'.format(i)])
        exp = load_ast('programs/saved_tests/deterministic/test_{}.daphne'.format(i))

        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))

        print('FOPPL Test {} passed'.format(i))

    for i in range(1,13):

        # exp = daphne(['desugar-hoppl', '-i', '../CPSC532W-HW/Kevin/HOPPL/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        exp = load_ast('programs/saved_tests/hoppl-deterministic/test_{}.daphne'.format(i))

        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('HOPPL Test {} passed'.format(i))
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        #exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        exp = load_ast('programs/saved_tests/probabilistic/test_{}.daphne'.format(i))
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    # run_deterministic_tests()  # TODO: still need to finish test 12; get more information about conj first
    run_probabilistic_tests()
    

    for i in range(1,4):
        print(i)
        # exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/{}.daphne'.format(i)])
        exp = load_ast('programs/saved_tests/{}.daphne'.format(i))
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate(exp))        
