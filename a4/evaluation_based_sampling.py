from daphne import daphne
from tests import is_tol, run_prob_test, load_truth
import torch
from primitives import primitives_evaluation, distributions_evaluation
import numpy as np
import distributions


# x = 0
# def variable_name():
#     global x
#     x += 1
#     return "v_{}".format(x)


def evaluate_program(ast, x):
    variables_dict = {}
    functions_dict = {}
    sigma = {}
    sigma['logW'] = 0
    sigma['Q'] = {}
    sigma['G'] = {}

    if type(ast[0]) is list and ast[0][0] == 'defn':
        function_expression = [ast[0][1], ast[0][2], ast[0][3]]
        functions_dict[ast[0][1]] = function_expression
        ast = ast[1]
    else:
        ast = ast[0]
    return evaluate_variable(ast, variables_dict, functions_dict, sigma, x)

def evaluate_program_with_sigma(ast, sigma, x):
    variables_dict = {}
    functions_dict = {}

    if type(ast[0]) is list and ast[0][0] == 'defn':
        function_expression = [ast[0][1], ast[0][2], ast[0][3]]
        functions_dict[ast[0][1]] = function_expression
        ast = ast[1]
    else:
        ast = ast[0]
    return evaluate_variable(ast, variables_dict, functions_dict, sigma, x)

primitives_operations = ['+', '-', '*', '/', 'sqrt', 'vector', 'hash-map', 'get', 'put', 'first', 'second', 'rest',
                         'last', 'append', '<', '<=', '>', '>=', '=', 'mat-transpose', 'mat-tanh', 'mat-mul', 'mat-add',
                         'mat-repmat', 'and', 'or']
distribution_types = ['normal', 'beta', 'exponential', 'uniform', 'discrete', 'gamma', 'dirichlet', 'flip', 'dirac']
condition_types = ['sample', 'let', 'if', 'defn', 'observe']

def evaluate_variable(ast, variables_dict, functions_dict, sigma, x):
    if type(ast) is not list:
        if ast in primitives_operations:
            return ast, sigma
        elif type(ast) is torch.Tensor:
            return ast, sigma
        elif type(ast) is int:
            return torch.tensor(float(ast)), sigma
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
            return conditions_evaluation(ast, variables_dict, functions_dict, sigma, x)
        else:
            sub_ast = []
            for elem in ast:
                elem, sigma = evaluate_variable(elem, variables_dict, functions_dict, sigma, x)
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
                return evaluate_variable(sub_ast[0][2], variables_dict, functions_dict, sigma, x)


def conditions_evaluation(ast, variables_dict, functions_dict, sigma, x):
    if ast[0] == 'sample':
        # object, sigma = evaluate_variable(ast[1], variables_dict, functions_dict, sigma)
        # sample = object.sample()
        # return sample, sigma

        p, sigma = evaluate_variable(ast[1], variables_dict, functions_dict, sigma, x)
        v = "v_{}".format(x)
        if v not in sigma['Q']:
            sigma['Q'][v] = p.make_copy_with_grads()
        c = sigma['Q'][v].sample()
        logP = sigma['Q'][v].log_prob(c)
        logP.backward()
        params = sigma['Q'][v].Parameters()
        grads = torch.zeros(len(params))
        i = 0
        for param in params:
            grads[i] = param.grad
            i += 1
        sigma['G'][v] = grads
        logW_v = p.log_prob(c) - sigma['Q'][v].log_prob(c)
        sigma['logW'] += logW_v
        return c, sigma


    elif ast[0] == 'let':
        variable_value, sigma = evaluate_variable(ast[1][1], variables_dict, functions_dict, sigma, x)
        variables_dict[ast[1][0]] = variable_value
        return evaluate_variable(ast[2], variables_dict, functions_dict, sigma, x)

    elif ast[0] == 'if':
        boolean, sigma = evaluate_variable(ast[1], variables_dict, functions_dict, sigma,x)
        if boolean:
            variable_type, sigma = evaluate_variable(ast[2], variables_dict, functions_dict, sigma, x)
            return variable_type, sigma
        else:
            variable_type, sigma = evaluate_variable(ast[3], variables_dict, functions_dict, sigma, x)
            return variable_type, sigma

    elif ast[0] == 'observe':
        dist, sigma = evaluate_variable(ast[1], variables_dict, functions_dict, sigma, x)
        observation, sigma = evaluate_variable(ast[2], variables_dict, functions_dict, sigma, x)
        if type(observation) is not torch.Tensor:
            observation = torch.tensor(float(observation))
        sigma['logW'] = sigma['logW'] + dist.log_prob(observation)
        return observation, sigma

def get_stream(ast, x):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast, x)

