import torch
import torch.distributions as dist


class Normal(dist.Normal):
    
    def __init__(self, alpha, loc, scale):
        
        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)
        

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
    'push-address': push_addr,
    '+': torch.add, '-': torch.sub, '*': torch.mul, '/': torch.div,
    '>': torch.gt, '<': torch.lt, '>=': torch.ge, '<=': torch.le, '=': torch.eq,
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






