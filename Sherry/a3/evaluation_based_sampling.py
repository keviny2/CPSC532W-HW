from daphne import daphne
from tests import is_tol, run_prob_test, load_truth
import torch
from primitives import primitives_evaluation, distributions_evaluation
import numpy as np

def evaluate_program(ast):
    variables_dict = {}
    functions_dict = {}
    sigma = {}
    sigma['logW'] = 0

    if type(ast[0]) is list and ast[0][0] == 'defn':
        function_expression = [ast[0][1], ast[0][2], ast[0][3]]
        functions_dict[ast[0][1]] = function_expression
        ast = ast[1]
    else:
        ast = ast[0]
    return evaluate_variable(ast, variables_dict, functions_dict, sigma)

primitives_operations = ['+', '-', '*', '/', 'sqrt', 'vector', 'hash-map', 'get', 'put', 'first', 'second', 'rest',
                         'last', 'append', '<', '<=', '>', '>=', '=', 'mat-transpose', 'mat-tanh', 'mat-mul', 'mat-add',
                         'mat-repmat', 'and', 'or']
distribution_types = ['normal', 'beta', 'exponential', 'uniform', 'discrete', 'gamma', 'dirichlet', 'flip', 'dirac']
condition_types = ['sample', 'let', 'if', 'defn', 'observe']

def evaluate_variable(ast, variables_dict, functions_dict, sigma):
    if type(ast) is not list:
        if ast in primitives_operations:
            return ast, sigma
        elif type(ast) is torch.Tensor:
            return ast, sigma
        elif type(ast) is int:
            return torch.tensor(ast), sigma
        elif type(ast) is float:
            return torch.tensor(ast), sigma
        elif type(ast) is bool:
            return ast, sigma
        elif ast in distribution_types:
            return ast, sigma
        elif ast in variables_dict:
            return variables_dict[ast], sigma
        elif ast in functions_dict:
            return functions_dict[ast], sigma
        elif ast is None:
            pass

    elif type(ast) is list:
        if ast[0] in condition_types:
            return conditions_evaluation(ast, variables_dict, functions_dict, sigma)
        else:
            sub_ast = []
            for elem in ast:
                elem, sigma = evaluate_variable(elem, variables_dict, functions_dict, sigma)
                sub_ast.append(elem)
            if sub_ast[0] in primitives_operations:
                return primitives_evaluation(sub_ast), sigma
            elif sub_ast[0] in distribution_types:
                return distributions_evaluation(sub_ast), sigma
            elif type(sub_ast[0]) is list and sub_ast[0][0] in functions_dict:
                variables = sub_ast[0][1]
                values = sub_ast[1:]
                for i in range(len(variables)):
                    variables_dict[variables[i]] = values[i]
                return evaluate_variable(sub_ast[0][2], variables_dict, functions_dict, sigma)


def conditions_evaluation(ast, variables_dict, functions_dict, sigma):
    if ast[0] == 'sample':
        object, sigma = evaluate_variable(ast[1], variables_dict, functions_dict, sigma)
        sample = object.sample()
        return sample, sigma
    elif ast[0] == 'let':
        variable_value, sigma = evaluate_variable(ast[1][1], variables_dict, functions_dict, sigma)
        variables_dict[ast[1][0]] = variable_value
        return evaluate_variable(ast[2], variables_dict, functions_dict, sigma)

    elif ast[0] == 'if':
        boolean, sigma = evaluate_variable(ast[1], variables_dict, functions_dict, sigma)
        if boolean:
            variable_type, sigma = evaluate_variable(ast[2], variables_dict, functions_dict, sigma)
            return variable_type, sigma
        else:
            variable_type, sigma = evaluate_variable(ast[3], variables_dict, functions_dict, sigma)
            return variable_type, sigma

    elif ast[0] == 'observe':
        dist, sigma = evaluate_variable(ast[1], variables_dict, functions_dict, sigma)
        observation, sigma = evaluate_variable(ast[2], variables_dict, functions_dict, sigma)
        if type(observation) is not torch.Tensor:
            observation = torch.tensor(float(observation))
        sigma['logW'] = sigma['logW'] + dist.log_prob(observation)
        return observation, sigma

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)


# def run_deterministic_tests():
#     # for i in range(1, 14):
#     #     ast = daphne(['desugar', '-i',
#     #                   '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/tests/deterministic/test_{}.daphne'.format(i)])
#     #     truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
#     #     ret, sig = evaluate_program(ast), '0'
#     #     try:
#     #         assert (is_tol(ret, truth))
#     #     except AssertionError:
#     #         raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, ast))
#     #
#     #     print('Test passed')
#
#     print('All deterministic tests passed')


# def run_probabilistic_tests():
#     num_samples = 1e4
#     max_p_value = 1e-4
#
#     for i in range(1, 7):
#         i = 5
#         ast = daphne(['desugar', '-i',
#                       '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
#         truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
#
#         stream = get_stream(ast)
#
#         p_val = run_prob_test(stream, truth, num_samples)
#
#         print('p value', p_val)
#         assert (p_val > max_p_value)
#         print('Test passed')
#
#     print('All probabilistic tests passed')


# if __name__ == '__main__':
#
#     run_deterministic_tests()
#
#     run_probabilistic_tests()
#
#     for i in range(1, 5):
#         ast = daphne(['desugar', '-i',
#                       '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a3/programs/{}.daphne'.format(i)])
#
#         print('\n\n\nSample of prior of program {}:'.format(i))
#         print(evaluate_program(ast))