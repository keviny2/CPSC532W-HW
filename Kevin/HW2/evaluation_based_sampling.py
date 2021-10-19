import copy
import pickle

from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from distributions import my_distributions, Distribution
import torch


variable_bindings = {}
functions = {}

def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """

    # make a deep copy so that substituting variables with actual values doesn't change
    # the variable for the next execution of the function by the generator
    ast_curr = copy.deepcopy(ast)

    if type(ast_curr) is not list:
        return ast_curr

    # substitute bound variables for their values

    # construct boolean array indicating which elements are variables that are bound to a value using the global
    # variable_bindings dictionary
    variables = list(variable_bindings.keys())
    variable_bindings_bool = [elem in variables for elem in ast_curr]

    # substitute the variable name in ast_curr with the actual value
    # BUG: sometimes works on tensors, at other times fails
    if any(variable_bindings_bool):
        for idx, val in enumerate(variable_bindings_bool):
            if val:
                ast_curr[idx] = variable_bindings[ast_curr[idx]]


    try:
        # always check whether there is a function definition in order to continue running the program after
        # saving the function def in the functions dict
        if ast_curr[0][0] == 'defn':
            functions[ast_curr[0][1]] = ast_curr[0][2:]
            key = ast_curr[0][1]

            # remove the defn from ast, some weird bugs arise if I keep it there
            ast_curr = ast_curr[1:]
            res = evaluate_program(ast_curr)

            # clear all functions
            del functions[key]

            return res
    except:
        pass

    if ast_curr[0] == 'if':
        e1 = evaluate_program(ast_curr[1])
        e2 = evaluate_program(ast_curr[2])
        e3 = evaluate_program(ast_curr[3])
        if e1:
            return e2
        else:
            return e3

    if ast_curr[0] == 'let':
        # we will successfully assign a variable a value after this block of code executes

        # check if the variable will be assigned to a distribution object
        if type(ast_curr[1][0]) is str and ast_curr[1][1][0] in my_distributions:
            distribution_object = evaluate_program(ast_curr[1][1])
            variable_bindings[ast_curr[1][0]] = distribution_object
        else:
            binding_obj = evaluate_program(ast_curr[1])
            variable_bindings[binding_obj[0]] = binding_obj[1]  # 33

        # throw away let statement and continue evaluation
        res = evaluate_program(ast_curr[2:])

        # clear all variable bindings
        del variable_bindings[binding_obj[0]]

        # make sure that a list is returned
        try:
            temp = res[0]  # try subscripting res
            return res
        except (TypeError, ValueError, IndexError):
            return torch.tensor(res)

    if ast_curr[0] == 'sample':
        distribution_obj = evaluate_program(ast_curr[1])
        return distribution_obj.sample()

    if ast_curr[0] == 'observe':
        return None

    if ast_curr[0] == 'vector':
        sublists = [evaluate_program(elem) for elem in ast_curr[1:]]
        try:
            ret = torch.tensor(sublists)
        except (ValueError, RuntimeError):
            ret = sublists
        return ret

    if all(item in my_distributions for item in list(list(zip(*ast_curr))[0])):
        return ast_curr

    # [+ 3 4]
    if all([type(elem) is not list for elem in ast_curr]):
        if type(ast_curr[0]) in [int, float, torch.Tensor]:
            return torch.tensor(ast_curr[0])
        if type(ast_curr[0]) is dict:
            return ast_curr[0]
        if ast_curr[0] in list(functions.keys()):
            function_params = dict(zip(functions[ast_curr[0]][0], ast_curr[1:]))
            function_body = functions[ast_curr[0]][1]

            try:
                assert(len(function_params) == (len(ast_curr) - 1))
            except AssertionError:
                raise AssertionError('Invalid number of parameters')

            processed_function_body = substitute_params(function_params, function_body)
            return evaluate_program(processed_function_body)

        if ast_curr[0] in my_distributions:
            distribution_obj = Distribution(ast_curr[0], ast_curr[1:])
            return distribution_obj
        elif ast_curr[0] == '+':
            return torch.sum(torch.tensor(ast_curr[1:]))
        elif ast_curr[0] == '-':
            return ast_curr[1] - torch.sum(torch.tensor(ast_curr[2:]))
        elif ast_curr[0] == '*':
            return torch.prod(torch.tensor(ast_curr[1:]))
        elif ast_curr[0] == '/':
            return ast_curr[1] / torch.prod(torch.tensor(ast_curr[2:]))
        elif ast_curr[0] == 'sqrt':
            return torch.sqrt(torch.tensor(ast_curr[1]))
        elif ast_curr[0] == '<':
            return ast_curr[1] < ast_curr[2]
        elif ast_curr[0] == '>':
            return ast_curr[1] > ast_curr[2]
        elif ast_curr[0] == 'get':
            if type(ast_curr[1]) is dict:
                return torch.tensor(ast_curr[1][ast_curr[2]])
            return torch.tensor(ast_curr[1][ast_curr[2]])
        elif ast_curr[0] == 'put':
            if type(ast_curr[1]) is dict:
                ast_curr[1][ast_curr[2]] = torch.tensor(ast_curr[3])
            else:
                ast_curr[1][ast_curr[2]] = torch.tensor(ast_curr[3])
            return ast_curr[1]
        elif ast_curr[0] == 'remove':
            del ast_curr[1][ast_curr[2]]
            return ast_curr[1]
        elif ast_curr[0] == 'first':
            return torch.tensor(ast_curr[1][0])
        elif ast_curr[0] == 'second':
            return torch.tensor(ast_curr[1][1])
        elif ast_curr[0] == 'last':
            return torch.tensor(ast_curr[1][-1])
        elif ast_curr[0] == 'rest':
            return torch.tensor(ast_curr[1][1:])
        elif ast_curr[0] == 'append':
            res = ast_curr[1].tolist()
            res.extend([ast_curr[2]])
            return torch.tensor(res)
        elif ast_curr[0] == 'hash-map':
            return dict(zip(ast_curr[1:][::2], torch.tensor(ast_curr[1:][1::2])))
        else:
            return ast_curr

    # [+ [- 3 1] [* 2 4]]
    subroot = [evaluate_program(sub_ast_curr) for sub_ast_curr in ast_curr]
    return evaluate_program(subroot)


def substitute_params_helper(params, body):
    res = []
    for elem in body:
        if elem in list(params.keys()):
            res.append(params[elem])
        else:
            res.append(elem)

    if len(res) == 1:
        return res[0]
    else:
        return res

def substitute_params(params, body):
    if not isinstance(body, list):
        if body in list(params.keys()):
            return params[body]
        else:
            return body

    if all([type(elem) is not list for elem in body]):
        return substitute_params_helper(params, body)

    curr = [substitute_params(params, elem) for elem in body]
    return curr

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():

    debug_start = 1
    for i in range(debug_start,14):
        #note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        ast = load_ast('programs/saved_asts/det{}_ast.pkl'.format(i))
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

    debug_start = 1
    for i in range(debug_start,7):
        #note: this path should be with respect to the daphne path!        
        # ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        ast = load_ast('programs/saved_asts/prob{}_ast.pkl'.format(i))
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(ast)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)
        print('Test passed', ast, 'test', i)
    
    print('All probabilistic tests passed')

def save_ast(file_name, my_ast):
    # faster load

    # saving the list
    open_file = open(file_name, "wb")
    pickle.dump(my_ast, open_file)
    open_file.close()

def load_ast(file_name):

    # loading the list
    open_file = open(file_name, 'rb')
    ret = pickle.load(open_file)
    open_file.close()
    return ret

if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()

    debug_start = 1
    for i in range(debug_start,5):
        # ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/HW2/programs/{}.daphne'.format(i)])

        ast = load_ast('programs/saved_asts/daphne{}_ast.pkl'.format(i))
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])