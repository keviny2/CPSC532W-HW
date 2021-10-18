import torch
from torch import distributions
import numpy as np

primitives_operations = ['+', '-', '*', '/', 'sqrt', 'vector', 'hash-map', 'get', 'put', 'first', 'second', 'rest',
                         'last', 'append', '<', '>', 'mat-transpose', 'mat-tanh', 'mat-mul', 'mat-add', 'mat-repmat']
distribution_types = ['normal', 'beta', 'exponential', 'uniform', 'discrete']


def primitives_evaluation(ast):

    if ast[0] == '+':
        return torch.sum(torch.tensor(ast[1:]))
    elif ast[0] == '-':
        return ast[1] - (torch.sum(torch.tensor(ast[2:])))
    elif ast[0] == '*':
        return torch.prod(torch.tensor(ast[1:]))
    elif ast[0] == '/':
        return ast[1] / torch.prod(torch.tensor(ast[2:]))
    elif ast[0] == 'vector':
        try:
            return torch.stack(ast[1:])
        except:
            return ast[1:]
    elif ast[0] == 'sqrt':
        return torch.sqrt(torch.tensor([ast[1]]))
    elif ast[0] == 'hash-map':
        ast = np.reshape(np.array(ast[1:]), (-1, 2))
        ast = dict((ast[i][0], torch.tensor(ast[i][1])) for i in range(ast.shape[0]))
        return ast
    elif ast[0] == 'get':
        try:
            return ast[1][ast[2].item()]
        except:
            return ast[1][int(ast[2])]
    elif ast[0] == 'put':
        (ast[1])[int(ast[2])] = ast[3]
        return ast[1]
    elif ast[0] == 'first':
        return (ast[1])[0]
    elif ast[0] == 'second':
        return (ast[1])[1]
    elif ast[0] == 'rest':
        return (ast[1])[1:]
    elif ast[0] == 'last':
        return (ast[1])[len(ast[1]) - 1]
    elif ast[0] == 'append':
        return torch.cat((ast[1], torch.tensor([ast[2]])), dim=0)
    elif ast[0] == '<':
        return ast[1] < ast[2]
    elif ast[0] == '>':
        return ast[1] > ast[2]
    elif ast[0] == 'mat-transpose':
        return ast[1].T
    elif ast[0] == 'mat-tanh':
        return torch.tanh(ast[1])
    elif ast[0] == 'mat-mul':
        ast[1] = ast[1].float()
        ast[2] = ast[2].float()
        return torch.matmul(ast[1], ast[2])
    elif ast[0] == 'mat-add':
        return ast[1] + ast[2]
    elif ast[0] == 'mat-repmat':
        return torch.tensor(ast[1]).repeat(int(ast[2]), int(ast[3]))


def distributions_evaluation(ast):
    if ast[0] == 'normal':
        dist = distributions.normal.Normal(float(ast[1]), float(ast[2]))
        return dist
    elif ast[0] == 'beta':
        dist = distributions.beta.Beta(float(ast[1]), float(ast[2]))
        return dist
    elif ast[0] == 'exponential':
        dist = distributions.exponential.Exponential(float(ast[1]))
        return dist
    elif ast[0] == 'uniform':
        dist = distributions.uniform.Uniform(float(ast[1]), float(ast[2]))
        return dist
    elif ast[0] == 'discrete':
        for i in range(len(ast[1])):
            ast[1][i] = float(ast[1][i])
        dist = distributions.categorical.Categorical(ast[1])
        return dist
    else:
        print("need define distribution for: %s" % ast[0])



def minus(ast):
    return ast[0] - (torch.sum(torch.tensor(ast[1:])))

def divide(ast):
    return ast[0] / torch.prod(torch.tensor(ast[1:]))

def vector(ast):
    try:
        for i in range(len(ast)):
           ast[i] = torch.tensor(ast[i])
        return torch.stack(ast)
    except:
        return ast

def hashmap(ast):
    ast = np.reshape(np.array(ast), (-1, 2))
    ast = dict((ast[i][0], torch.tensor(ast[i][1])) for i in range(ast.shape[0]))
    return ast

def get(ast):
    try:
        return ast[0][ast[1].item()]
    except:
        return ast[0][int(ast[1])]

def put(ast):
    (ast[0])[int(ast[1])] = ast[2]
    return ast[0]

def first(ast):
    return (ast[0])[0]

def second(ast):
    return (ast[0])[1]

def rest(ast):
    return (ast[0])[1:]

def last(ast):
    return (ast[0])[len(ast[0]) - 1]

def append(ast):
    return torch.cat((ast[0], torch.tensor([ast[1]])), dim=0)

def smaller(ast):
    return ast[0] < ast[1]

def larger(ast):
    return ast[0] > ast[1]

def mat_transpose(ast):
    return ast[0].T

def mat_tanh(ast):
    return torch.tanh(ast[0])

def mat_mul(ast):
    ast[0] = ast[0].float()
    ast[1] = ast[1].float()
    return torch.matmul(ast[0], ast[1])

def mat_add(ast):
    return ast[0] + ast[1]

def mat_repmat(ast):
    return  torch.tensor(ast[0]).repeat(int(ast[1]), int(ast[2]))


