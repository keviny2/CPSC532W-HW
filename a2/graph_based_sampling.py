import torch
from torch import distributions as dist
from daphne import daphne
from evaluation_based_sampling import evaluate_program
import numpy as np

import primitives
from tests import is_tol, run_prob_test,load_truth


primitives_operations = ['+', '-', '*', '/', 'sqrt', 'vector', 'hash-map', 'get', 'put', 'first', 'second', 'rest',
                         'last', 'append', '<', '>', 'mat-transpose', 'mat-tanh', 'mat-mul', 'mat-add', 'mat-repmat']
distribution_types = ['normal', 'beta', 'exponential', 'uniform', 'discrete']
condition_types = ['sample', 'let', 'if', 'defn', 'observe']

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': dist.Normal,
       'beta': dist.Beta,
       'exponential': dist.Exponential,
       'uniform': dist.Uniform,
       'discrete': dist.Categorical,
       '+': torch.sum,
       '*': torch.multiply,
       '-': primitives.minus,
       '/': primitives.divide,
       'sqrt': torch.sqrt,
       'vector': primitives.vector,
       'hash-map': primitives.hashmap,
       'get': primitives.get,
       'put': primitives.put,
       'first': primitives.first,
       'second': primitives.second,
       'rest': primitives.rest,
       'last': primitives.last,
       'append': primitives.append,
       '<': primitives.smaller,
       '>': primitives.larger,
       'mat-transpose': primitives.mat_transpose,
       'mat-tanh': primitives.mat_tanh,
       'mat-mul': primitives.mat_mul,
       'mad-add': primitives.mat_add,
       'if': primitives.iff
       }


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))

    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    elif type(exp) is torch.Tensor:
        return exp
    else:
        raise("Expression type unknown.", exp)



def sample_from_joint(graph):

    "This function does ancestral sampling starting from the prior."

    vertices = graph[1]['V']
    edges = graph[1]['A']
    links = graph[1]['P']
    flows = graph[1]['Y']
    returnings = graph[2]
    variables_dict = {}

    unique_vertices = []
    degrees = {}
    for vertex in vertices:
        if vertex not in degrees:
            degrees[vertex] = 0
            unique_vertices.append(vertex)

    for vertex in unique_vertices:
        if vertex in edges:
            leaves = edges[vertex]
            for leave in leaves:
                degrees[leave] += 1

    ordering = []
    while len(ordering) != len(unique_vertices):
        for vertex in unique_vertices:
            degrees[vertex] -= 1
            if degrees[vertex] == -1:
                ordering.append(vertex)


    for vertex in ordering:
        link = links[vertex]
        if link[0] == 'sample*':
            record = evaluate(link[1], variables_dict)
            try:
                dist = deterministic_eval(record)
                value = dist.sample()
                variables_dict[vertex] = value
            except:
                ordering.append(vertex)
                pass

        elif link[0] == 'observe*':
            continue

    record = evaluate(returnings, variables_dict)
    return deterministic_eval(record)

def evaluate(exp, variables_dict):

    if type(exp) is not list:
        if exp in variables_dict:
            return variables_dict[exp]
        else:
            return exp
    else:
        record = []
        for sub_exp in exp:
            value = evaluate(sub_exp, variables_dict)
            record.append(value)
        return record

def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i',
                        '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))

        print('Test passed')

    print('All deterministic tests passed')
    


def run_probabilistic_tests():

    num_samples = 1
    max_p_value = 1e-4

    for i in range(1,7):
        graph = daphne(['graph', '-i',
                        '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(graph)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    run_deterministic_tests()
    run_probabilistic_tests()

    for i in range(1,5):
        graph = daphne(['graph','-i',
                        '/Users/xiaoxuanliang/Desktop/CPSC 532W/HW/a2/programs/{}.daphne'.format(i)])
        print(graph)
        print('\n\n\nSample of prior of program {}:'.format(i))

        print(sample_from_joint(graph))

    