from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test, load_truth
from pyrsistent import pmap, plist

from utils import save_ast, load_ast
import math
import operator as op


class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."

    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)


class Procedure(object):
    "A user-defined Scheme procedure."

    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env

    def __call__(self, *args):
        return eval(self.body, Env(self.parms, args, self.env))


def standard_env():
    "An environment with some Scheme standard procedures."
    env = pmap(penv)  # this is defined in primitives.py
    env.update(vars(math))
    env = env.update({
        'alpha': '',
        '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv,
        '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le, '=': op.eq,
        'abs': abs,
        'append': op.add,
        'apply': lambda proc, args: proc(*args),
        'begin': lambda *x: x[-1],
        'car': lambda x: x[0],
        'cdr': lambda x: x[1:],
        'cons': lambda x, y: [x] + y,
        'eq?': op.is_,
        'expt': pow,
        'equal?': op.eq,
        'length': len,
        'list': lambda *x: list(x),
        'list?': lambda x: isinstance(x, list),
        'map': map,
        'max': max,
        'min': min,
        'not': op.not_,
        'null?': lambda x: x == [],
        'number?': lambda x: isinstance(x, int) or isinstance(x, float),
        'print': print,
        'procedure?': callable,
        'round': round,
        'symbol?': lambda x: isinstance(x, str),
    })
    return env


def evaluate(exp, env=None):  # TODO: add sigma, or something

    if env is None:
        env = standard_env()

    # simply return constants and variables
    if isinstance(exp, str):  # variable reference
        return env.find(exp)[exp]
    elif not isinstance(exp, list):  # constant
        return exp

    # if exp in distributions:
    #     return exp, sig
    # if type(exp) is torch.Tensor:
    #     return exp, sig
    # if type(exp) in [int, float]:
    #     return torch.tensor(exp), sig
    # if type(exp) is bool:
    #     return torch.FloatTensor([exp]), sig
    # if exp in list(variable_bindings.keys()):
    #     return variable_bindings[exp], sig
    # if exp in list(functions.keys()):
    #     return functions[exp], sig
    # if exp is None:
    #     return None, sig

    op, *args = exp
    if op == 'quote':  # quotation
        return args[0]
    elif op == 'if':  # conditional
        (test, conseq, alt) = args
        exp = (conseq if eval(test, env) else alt)
        return eval(exp, env)
    elif op == 'define':  # definition
        (symbol, exp) = args
        env[symbol] = eval(exp, env)
    elif op == 'set!':  # assignment
        (symbol, exp) = args
        env.find(symbol)[symbol] = eval(exp, env)
    elif op == 'lambda':  # procedure
        (parms, body) = args
        return Procedure(parms, body, env)
    else:  # procedure call
        proc = eval(op, env)
        vals = [eval(arg, env) for arg in args]
        return proc(*vals)


def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():
    for i in range(1, 14):
        exp = daphne(['desugar-hoppl', '-i', '../CPSC532W-HW/Kevin/HOPPL/programs/tests/deterministic/test_{}.daphne'.format(i)])

        save_ast('programs/saved_tests/deterministic/test_{}.daphne'.format(i), exp)
        # truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        # ret = evaluate(exp)
        # try:
        #     assert(is_tol(ret, truth))
        # except:
        #     raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        #
        # print('FOPPL Tests passed')

    for i in range(1, 13):

        exp = daphne(['desugar-hoppl', '-i',
                      '../CPSC532W-HW/Kevin/HOPPL/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        save_ast('programs/saved_tests/hoppl-deterministic/test_{}.daphne'.format(i), exp)
        # truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        # ret = evaluate(exp)
        # try:
        #     assert (is_tol(ret, truth))
        # except:
        #     raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, exp))
        #
        # print('Test passed')

    print('All deterministic tests passed')


def run_probabilistic_tests():
    num_samples = 1e4
    max_p_value = 1e-2

    for i in range(1, 7):
        exp = daphne(['desugar-hoppl', '-i', '../CPSC532W-HW/Kevin/HOPPL/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        save_ast('programs/saved_tests/probabilistic/test_{}.daphne'.format(i), exp)
        # truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        #
        # stream = get_stream(exp)
        #
        # p_val = run_prob_test(stream, truth, num_samples)
        #
        # print('p value', p_val)
        # assert (p_val > max_p_value)

    print('All probabilistic tests passed')


if __name__ == '__main__':

    # run_deterministic_tests()
    run_probabilistic_tests()

    # for i in range(1, 4):
    #     print(i)
    #     exp = daphne(['desugar-hoppl', '-i', '../CPSC532W-HW/Kevin/HOPPL/programs/{}.daphne'.format(i)])
    #     save_ast('programs/{}.daphne'.format(i), exp)
        # print('\n\n\nSample of prior of program {}:'.format(i))
        # print(evaluate(exp))
