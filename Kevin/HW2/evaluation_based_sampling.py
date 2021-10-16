from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import torch
from torch.distributions import normal, beta, exponential

variable_bindings = {}

def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """

    if type(ast) is not list:
        return ast

    # BUG: THERE IS A BUG IN MY CODE FOR LET!!!!!
    # substitute bound variables for their values

    # construct boolean array indicating which elements are variables that are bound to a value using the global
    # variable_bindings dictionary
    variables = list(variable_bindings.keys())
    variable_bindings_bool = [elem in variables for elem in ast]

    # substitute the variable name in the ast with the actual value
    if any(variable_bindings_bool):
        for idx, val in enumerate(variable_bindings_bool):
            if val:
                ast[idx] = variable_bindings[ast[idx]]


    if ast[0] == 'let':
        # we will successfully assign a variable a value after this block of code executes
        # BUG: weird behavior
        binding_obj = evaluate_program(ast[1])
        variable_bindings[binding_obj[0]] = binding_obj[1]

        # throw away let statement and continue evaluation
        return evaluate_program(ast[2:])

    if ast[0] == 'sample':
        distribution_obj = evaluate_program(ast[1])
        if distribution_obj[0] == 'normal':
            return normal.Normal(distribution_obj[1], distribution_obj[2]).sample()
        if distribution_obj[0] == 'beta':
            return beta.Beta(distribution_obj[1], distribution_obj[2]).sample()
        if distribution_obj[0] == 'exponential':
            return exponential.Exponential(distribution_obj[1]).sample()

    # [+ 3 4]
    if all([type(elem) is not list for elem in ast]):
        if type(ast[0]) in [int, float, torch.Tensor]:
            return torch.tensor(ast[0])
        if type(ast[0]) is dict:
            return ast[0]
        if ast[0] in ['normal', 'beta', 'exponential']:
            return ast
        elif ast[0] == '+':
            return torch.sum(torch.tensor(ast[1:]))
        elif ast[0] == 'sqrt':
            return torch.sqrt(torch.tensor(ast[1]))
        elif ast[0] == '-':
            return ast[1] - torch.sum(torch.tensor(ast[2:]))
        elif ast[0] == '*':
            return torch.prod(torch.tensor(ast[1:]))
        elif ast[0] == '/':
            return ast[1] / torch.prod(torch.tensor(ast[2:]))
        elif ast[0] == 'vector':
            return torch.tensor(ast[1:])
        elif ast[0] == 'get':
            if type(ast[1]) is dict:
                return torch.tensor(ast[1][ast[2]])
            return torch.tensor(ast[1][ast[2]])
        elif ast[0] == 'put':
            if type(ast[1]) is dict:
                ast[1][ast[2]] = torch.tensor(ast[3])
            else:
                ast[1][ast[2]] = torch.tensor(ast[3])
            return ast[1]
        elif ast[0] == 'first':
            return torch.tensor(ast[1][0])
        elif ast[0] == 'last':
            return torch.tensor(ast[1][-1])
        elif ast[0] == 'append':
            res = ast[1].tolist()
            res.extend([ast[2]])
            return torch.tensor(res)
        elif ast[0] == 'hash-map':
            return dict(zip(ast[1:][::2], torch.tensor(ast[1:][1::2])))
        else:
            return ast

    # [+ [- 3 1] [* 2 4]]
    subroot = [evaluate_program(sub_ast) for sub_ast in ast]
    return evaluate_program(subroot)


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():

    debug_start = 1
    for i in range(debug_start,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast), '0'
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed', ast, 'test', i)
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4

    debug_start = 4
    for i in range(debug_start,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
        print('Test passed', ast, 'test', i)
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    # run_deterministic_tests()
    
    run_probabilistic_tests()


    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])