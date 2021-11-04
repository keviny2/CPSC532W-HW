import torch
from torch import distributions as dist
from daphne import daphne
from evaluation_based_sampling import evaluate_program
import numpy as np
import distributions


import primitives
from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': distributions.Normal,
       'beta': dist.Beta,
       'exponential': dist.Exponential,
       'uniform': dist.Uniform,
       'uniform-continuous': dist.Uniform,
       'discrete': distributions.Categorical,
       'gamma': distributions.Gamma,
       'flip': distributions.Bernoulli,
       'dirichlet': distributions.Dirichlet,
       'dirac': primitives.Dirac,
       '+': primitives.plus,
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
       '=': primitives.equal,
       'mat-transpose': primitives.mat_transpose,
       'mat-tanh': primitives.mat_tanh,
       'mat-mul': primitives.mat_mul,
       'mat-add': primitives.mat_add,
       'mat-repmat': primitives.mat_repmat,
       'if': primitives.iff,
       'and': primitives.func_and,
       'or': primitives.func_or
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
    elif type(exp) is bool:
        return exp
    elif exp is None:
        pass
    else:
        raise("Expression type unknown.", exp)



def sample_from_joint(graph):
    vertices = graph[1]['V']
    edges = graph[1]['A']
    links = graph[1]['P']
    flows = graph[1]['Y']
    returnings = graph[2]
    variables_dict = {}

    sigma = {}
    sigma['logW'] = 0
    sigma['Q'] = {}
    sigma['G'] = {}

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

    topological_orderings = []
    for vertex in ordering:
        link = links[vertex]
        if link[0] == 'sample*':
            record = evaluate(link[1], variables_dict)
            try:
                dist = deterministic_eval(record)
                value = dist.sample()
                variables_dict[vertex] = value
                topological_orderings.append(vertex)
            except:
                ordering.append(vertex)
                pass

        elif link[0] == 'observe*':
            record = evaluate(link[2], variables_dict)
            value = deterministic_eval(record)
            variables_dict[vertex] = value
            topological_orderings.append(vertex)

    record = evaluate(returnings, variables_dict)
    return deterministic_eval(record), variables_dict, topological_orderings


def sample_from_joint_with_sorted(graph, topological_orderings, x, sigma):
    links = graph[1]['P']
    returnings = graph[2]
    variables_dict = {}

    for vertex in topological_orderings:
        link = links[vertex]
        if link[0] == 'sample*':
            v = 'v_{}'.format(x)
            link.insert(1, v)
            record = evaluate(link[2], variables_dict)
            p = deterministic_eval(record)
            if v not in sigma['Q']:
                sigma['Q'][v] = p.make_copy_with_grads()
                sigma['lambda'][v] = sigma['optimizer'](sigma['Q'][v].Parameters(), lr = 1e-4)

            c = sigma['Q'][v].sample()
            variables_dict[vertex] = c
            logP = sigma['Q'][v].log_prob(c)
            logP.backward()
            params = sigma['Q'][v].Parameters()
            try:
                l = len(params[0])
                grads = params[0].grad.clone().detach()
            except:
                grads = torch.zeros(len(params))
                i = 0
                for param in params:
                    grads[i] = param.grad.clone().detach()
                    i += 1
            sigma['G'][v] = grads
            sigma['lambda'][v].zero_grad()
            logW_v = p.log_prob(c) - sigma['Q'][v].log_prob(c)
            sigma['logW'] += logW_v
            x += 1

        elif link[0] == 'observe*':

            d = deterministic_eval(evaluate(link[1], variables_dict))
            c = deterministic_eval(evaluate(link[2], variables_dict))
            variables_dict[vertex] = c
            if type(c) is not torch.Tensor:
                c = torch.tensor(float(c))
            sigma['logW'] += d.log_prob(c)

    record = evaluate(returnings, variables_dict)
    return deterministic_eval(record), sigma


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



