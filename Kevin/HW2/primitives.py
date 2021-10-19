import torch

simple_operations = ['+', '-', '*', '/', 'sqrt', 'vector', 'hash-map', 'get', 'put', 'first',
                     'second', 'rest', 'last', 'append', '<', '<=', '>', '>=', '==', 'mat-transpose',
                     'mat-tanh', 'mat-mul', 'mat-add', 'mat-repmat']

complex_operations = ['sample', 'let', 'if', 'defn', 'observe']


def evaluate_simple_operation(ast):
    if ast[0] == '+':
        return torch.sum(torch.tensor(ast[1:]))
    elif ast[0] == '-':
        return ast[1] - torch.sum(torch.tensor(ast[2:]))
    elif ast[0] == '*':
        return torch.prod(torch.tensor(ast[1:]))
    elif ast[0] == '/':
        return ast[1] / torch.prod(torch.tensor(ast[2:]))
    elif ast[0] == 'sqrt':
        return torch.sqrt(torch.tensor(ast[1]))
    elif ast[0] == '<':
        return ast[1] < ast[2]
    elif ast[0] == '>':
        return ast[1] > ast[2]
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
    elif ast[0] == 'remove':
        del ast[1][ast[2]]
        return ast[1]
    elif ast[0] == 'first':
        return torch.tensor(ast[1][0])
    elif ast[0] == 'second':
        return torch.tensor(ast[1][1])
    elif ast[0] == 'last':
        return torch.tensor(ast[1][-1])
    elif ast[0] == 'rest':
        return torch.tensor(ast[1][1:])
    elif ast[0] == 'append':
        res = ast[1].tolist()
        res.extend([ast[2]])
        return torch.tensor(res)
    elif ast[0] == 'hash-map':
        return dict(zip(ast[1:][::2], torch.tensor(ast[1:][1::2])))
#TODO