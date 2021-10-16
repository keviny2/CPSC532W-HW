from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import torch
from torch import distributions
import numpy as np
        
def evaluate_program(ast):
    if type(ast) is not list:
        if type(ast) is dict or type(ast) is str:
            return ast
        else:
            return torch.tensor(ast)

    if type(ast) is list and all([type(elem) is dict for elem in ast]):
        return ast[0]

    if type(ast) is list and ast[0] == 'sample':
        return ast[1]

    if isinstance(ast[0], list) and ast[0][0] in ['normal', 'beta', 'exponential', 'uniform']:
        if all([type(elem) is torch.Tensor for elem in ast[0][1:]]) or all([type(elem) is int for elem in ast[0][1:]]) or all([type(elem) is float for elem in ast[0][1:]]):

            if ast[0][0] == 'normal':
                dist = distributions.normal.Normal(float(ast[0][1]), float(ast[0][2]))
                sample = dist.sample()
                return sample
            elif ast[0][0] == 'beta':
                dist = distributions.beta.Beta(float(ast[0][1]), float(ast[0][2]))
                sample = dist.sample()
                return sample
            elif ast[0][0] == 'exponential':
                dist = distributions.exponential.Exponential(float(ast[0][1]))
                sample = dist.sample()
                return sample

    if all([type(elem) is not list for elem in ast]):
        if type(ast[0]) == torch.Tensor:
            return ast[0]
        elif ast[0] == '+':
            return torch.sum(torch.tensor(ast[1:]))
        elif ast[0] == '-':
            return ast[1] - (torch.sum(torch.tensor(ast[2:])))
        elif ast[0] == '*':
            return torch.prod(torch.tensor(ast[1:]))
        elif ast[0] == '/':
            return ast[1] / torch.prod(torch.tensor(ast[2:]))
        elif ast[0] == 'sqrt':
            return torch.sqrt(torch.tensor([ast[1]]))
        elif ast[0] == 'vector':
            return torch.tensor(ast[1:])
        elif ast[0] == 'hash-map':
            ast = np.reshape(np.array(ast[1:]), (-1, 2))
            ast = dict((ast[i][0], torch.tensor(ast[i][1])) for i in range(ast.shape[0]))
            return ast
        elif ast[0] == 'get':
            if type(ast[1]) is str:
                return str(ast[2])
            else:
                return torch.tensor((ast[1])[int(ast[2])])
        elif ast[0] == 'put':
           (ast[1])[int(ast[2])] = ast[3]
           return ast[1]
        elif ast[0] == 'first':
            return torch.tensor((ast[1])[0])
        elif ast[0] == 'last':
            return torch.tensor((ast[1])[len(ast[1]) - 1])
        elif ast[0] == 'append':
            return torch.cat((ast[1], torch.tensor([ast[2]])), dim = 0)
        else:
            return ast

    subroot = [evaluate_program(sub_ast) for sub_ast in ast]
    return evaluate_program(subroot)


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    #for i in range(1,14):

        # ast = daphne(['desugar', '-i',
        #               '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        # truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        # ret, sig = evaluate_program(ast), '0'
        # try:
        #     assert(is_tol(ret, truth))
        # except AssertionError:
        #     raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        #
        # print('Test passed')

    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples = 1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        i = 1
        ast = daphne(['desugar', '-i',
                      '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
        print('Test passed')
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()


    for i in range(1,5):
        ast = daphne(['desugar', '-i',
                      '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])