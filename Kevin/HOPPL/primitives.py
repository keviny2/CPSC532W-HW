import torch
from distributions import Normal, Bernoulli, Gamma, Dirichlet, Categorical, Beta, Exponential, UniformContinuous


def push_addr(alpha, value):
    return alpha + value


def vector(*args):
    try:
        return torch.stack(args)
    except:
        return args


def hash_map(*args):
    ret = {}
    for idx, elem in enumerate(args):
        if idx % 2 == 0:
            if type(elem) is torch.Tensor:
                ret[elem.item()] = args[idx + 1]
            else:
                ret[elem] = args[idx + 1]
    return ret


def append(*args):
    try:
        return torch.cat((args[0], args[1]))
    except:
        return torch.cat((args[0], torch.unsqueeze(args[1], 0)))


def get(*args):
    if type(args[1]) is str:
        return args[0][args[1]]
    else:
        return args[0][args[1].item()]


def put(*args):
    if type(args[0]) is dict:
        ret = args[0].copy()
        if type(args[1]) is str:
            ret[args[1]] = args[2]
        else:
            ret[args[1].item()] = args[2]
    else:  # assuming args[0] will be a torch tensor
        ret = torch.clone(args[0])
        ret[args[1].item()] = args[2]
    return ret


def remove(*args):
    del args[0][args[1]]
    return args[0]


def first(*args):
    return args[0][0]


def second(*args):
    return args[0][1]


def last(*args):
    return args[0][-1]


def rest(*args):
    return args[0][1:]


def mat_repmat(*args):
    return args[0].repeat(args[1].item(), args[2].item())

def empty(*args):
    return len(args[0]) == 0


env = {
    'alpha': '',
    'normal': Normal,
    'gamma': Gamma,
    'dirichlet': Dirichlet,
    'discrete': Categorical,
    'beta': Beta,
    'exponential': Exponential,
    'flip': Bernoulli,
    'uniform-continuous': UniformContinuous,
    'push-address': push_addr,
    '+': torch.add, '-': torch.sub, '*': torch.mul, '/': torch.div,
    '>': torch.gt, '<': torch.lt, '>=': torch.ge, '<=': torch.le, '=': torch.eq,
    'log': torch.log,
    'sqrt': torch.sqrt,
    'abs': torch.abs,
    'and': torch.logical_and,
    'or': torch.logical_or,
    'vector': vector,
    'hash-map': hash_map,
    'empty?': empty,
    'get': get,
    'put': put,
    'remove': remove,
    'first': first,
    'second': second,
    'last': last,
    'rest': rest,
    'append': append,
    'mat-mul': torch.matmul,
    'mat-add': torch.add,
    'mat-repmat': mat_repmat,
    'mat-tanh': torch.tanh,
    'mat-transpose': torch.t
}






