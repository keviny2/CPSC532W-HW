import torch

math_operations = ['+', '-', '*', '/', 'sqrt', '<', '<=', '>', '>=', '==']

data_structure_operations = ['vector', 'hash-map', 'get', 'put', 'first',
                             'second', 'rest', 'last', 'append']

matrix_operations = ['mat-transpose', 'mat-tanh', 'mat-mul', 'mat-add', 'mat-repmat']

complex_operations = ['sample', 'if', 'defn', 'observe']

# TODO: implement primitives for graph - these primitives will be called when evaluating
#  the link function while doing ancestral sampling

def vector(*args):
    return evaluate_data_structure_operation(['vector', *args])

def sample(*args):
    # TODO: implement sample
    return evaluate_complex_operation()

def evaluate_matrix_operation(ast):
    if ast[0] == 'mat-mul':
        return torch.matmul(ast[1].float(), ast[2].float())
    if ast[0] == 'mat-add':
        return ast[1] + ast[2]
    if ast[0] == 'mat-repmat':
        return torch.tensor(ast[1]).repeat(int(ast[2]), int(ast[3]))
    if ast[0] == 'mat-tanh':
        return torch.tanh(ast[1])
    if ast[0] == 'mat-transpose':
        return ast[1].T


def evaluate_complex_operation(ast):
    if ast[0] == 'if':
        if ast[1]:
            return ast[2]
        else:
            return ast[3]

    if ast[0] == 'sample':
        return ast[1].sample()

    if ast[0] == 'observe':
        return ast[1].sample()


def evaluate_math_operation(ast):
    if ast[0] == '+':
        return torch.sum(torch.tensor(ast[1:]))
    elif ast[0] == '-':
        return ast[1] - torch.sum(torch.tensor(ast[2:]))
    elif ast[0] == '*':
        return torch.prod(torch.tensor(ast[1:]))
    elif ast[0] == '/':
        return ast[1] / torch.prod(torch.tensor(ast[2:]))
    elif ast[0] == 'sqrt':
        return torch.sqrt(ast[1])
    elif ast[0] == '<':
        return ast[1] < ast[2]
    elif ast[0] == '>':
        return ast[1] > ast[2]


def evaluate_data_structure_operation(ast):
    if ast[0] == 'vector':
        try:
            for i in range(1, len(ast)):
                ast[i] = torch.tensor(ast[i])
            return torch.stack(ast[1:])
        except:
            return ast[1:]
    elif ast[0] == 'hash-map':
        ret = {}
        for idx, elem in enumerate(ast[1:]):
            if idx % 2 == 0:
                if type(elem) is torch.Tensor:
                    ret[elem.numpy().item()] = ast[1:][idx + 1]
        return ret
    elif ast[0] == 'get':
        if type(ast[1]) is dict:
            if type(ast[2]) is torch.Tensor:
                return ast[1][ast[2].numpy().item()]
            return ast[1][ast[2]]
        else:
            return ast[1][ast[2]]
    elif ast[0] == 'put':
        if type(ast[1]) is dict:
            if type(ast[2]) is torch.Tensor:
                ast[1][ast[2].numpy().item()] = ast[3]
        else:
            ast[1][ast[2]] = ast[3]
        return ast[1]
    elif ast[0] == 'remove':
        del ast[1][ast[2]]
        return ast[1]
    elif ast[0] == 'first':
        return ast[1][0]
    elif ast[0] == 'second':
        return ast[1][1]
    elif ast[0] == 'last':
        return ast[1][-1]
    elif ast[0] == 'rest':
        return ast[1][1:]
    elif ast[0] == 'append':
        return torch.cat((ast[1], torch.tensor([ast[2]])), dim=0)