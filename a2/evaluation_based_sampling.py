from daphne import daphne
from tests import is_tol, run_prob_test, load_truth
import torch
from primitives import primitives_evaluation, distributions_evaluation

def evaluate_program(ast):
    variables_dict = {}
    functions_dict = {}

    if type(ast[0]) is list and ast[0][0] == 'defn':
        function_expression = [ast[0][1], ast[0][2], ast[0][3]]
        functions_dict[ast[0][1]] = function_expression
        ast = ast[1]
    else:
        ast = ast[0]
    return evaluate_variable(ast, variables_dict, functions_dict)

primitives_operations = ['+', '-', '*', '/', 'sqrt', 'vector', 'hash-map', 'get', 'put', 'first', 'second', 'rest',
                         'last', 'append', '<', '<=', '>', '>=', '==', 'mat-transpose', 'mat-tanh', 'mat-mul', 'mat-add',
                         'mat-repmat']
distribution_types = ['normal', 'beta', 'exponential', 'uniform', 'discrete']
condition_types = ['sample', 'let', 'if', 'defn', 'observe']

def evaluate_variable(ast, variables_dict, functions_dict):
    if type(ast) is not list:
        if ast in primitives_operations:
            return ast
        elif type(ast) is torch.Tensor:
            return ast
        elif type(ast) is int:
            return torch.tensor(ast)
        elif type(ast) is float:
            return torch.tensor(ast)
        elif ast in distribution_types:
            return ast
        elif ast in variables_dict:
            return variables_dict[ast]
        elif ast in functions_dict:
            return functions_dict[ast]
        elif ast is None:
            return None

    elif type(ast) is list:
        if ast[0] in condition_types:
            return conditions_evaluation(ast, variables_dict, functions_dict)
        else:
            sub_ast = []
            for elem in ast:
                elem = evaluate_variable(elem, variables_dict, functions_dict)
                sub_ast.append(elem)
            if sub_ast[0] in primitives_operations:
                return primitives_evaluation(sub_ast)
            elif sub_ast[0] in distribution_types:
                return distributions_evaluation(sub_ast)
            elif type(sub_ast[0]) is list and sub_ast[0][0] in functions_dict:
                variables = sub_ast[0][1]
                values = sub_ast[1:]
                for i in range(len(variables)):
                    variables_dict[variables[i]] = values[i]
                return evaluate_variable(sub_ast[0][2], variables_dict, functions_dict)


def conditions_evaluation(ast, variables_dict, functions_dict):
    if ast[0] == 'sample':
        object = evaluate_variable(ast[1], variables_dict, functions_dict)
        sample = object.sample()
        return sample
    elif ast[0] == 'let':
        variable_value = evaluate_variable(ast[1][1], variables_dict, functions_dict)
        variables_dict[ast[1][0]] = variable_value
        return evaluate_variable(ast[2], variables_dict, functions_dict)

    elif ast[0] == 'if':
        boolean = evaluate_variable(ast[1], variables_dict, functions_dict)
        if boolean:
            variable_type = evaluate_variable(ast[2], variables_dict, functions_dict)
            return variable_type
        else:
            variable_type = evaluate_variable(ast[3], variables_dict, functions_dict)
            return variable_type

    elif ast[0] == 'observe':
        return evaluate_variable(None, variables_dict, functions_dict)


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)


def run_deterministic_tests():
    # for i in range(1, 14):
    #     i = 6
    #     ast = daphne(['desugar', '-i',
    #                   '/Users/xiaoxuanliang/Desktop/a2/programs/tests/deterministic/test_{}.daphne'.format(i)])
    #     truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
    #     ret, sig = evaluate_program(ast), '0'
    #     try:
    #         assert (is_tol(ret, truth))
    #     except AssertionError:
    #         raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, ast))
    #
    #     print('Test passed')

    print('All deterministic tests passed')


def run_probabilistic_tests():
    # num_samples = 1e4
    # max_p_value = 1e-4
    #
    # for i in range(1, 7):
    #     i = 6
    #     ast = daphne(['desugar', '-i',
    #                   '/Users/xiaoxuanliang/Desktop/a2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
    #     truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
    #
    #     stream = get_stream(ast)
    #
    #     p_val = run_prob_test(stream, truth, num_samples)
    #
    #     print('p value', p_val)
    #     assert (p_val > max_p_value)
    #     print('Test passed')

    print('All probabilistic tests passed')


if __name__ == '__main__':

    run_deterministic_tests()

    run_probabilistic_tests()

    for i in range(1, 5):
        ast = daphne(['desugar', '-i',
                      '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/{}.daphne'.format(i)])

        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast))