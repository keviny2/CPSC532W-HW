from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import torch
import numpy as np
        
def evaluate_program(ast):
    if type(ast) is not list:
        return ast

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
            ast = dict((ast[i][0], ast[i][1]) for i in range(ast.shape[0]))
            return ast
        elif ast[0] == 'get':
            return (ast[1])[ast[2]]
        elif ast[0] == 'put':
            if type(ast[1]) == torch.Tensor:
                (ast[1])[ast[2]] = ast[3]
                return ast[1]
            elif type(ast[1]) == dict:
                (ast[1])[ast[2]] = ast[3]
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

    return torch.tensor(evaluate_program(subroot))


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        print(i)
        ast = daphne(['desugar', '-i',
                      '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast), '0'
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        ast = daphne(['desugar', '-i',
                      '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/tests/probabilitstic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()


    for i in range(1,5):
        ast = daphne(['desugar', '-i',
                      '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])