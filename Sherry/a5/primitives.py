import torch
import torch.distributions as dist
import operator as op
import numpy as np


class Normal(dist.Normal):
    
    def __init__(self, loc, scale, copy = False):

        if not copy:
            if scale > 20.:
                self.optim_scale = scale.clone().detach().requires_grad_()
            else:
                self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        else:
            self.optim_scale = scale
        
        
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps, copy = True)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)

    def sample(self):
        return super().sample()


class Beta(dist.Beta):

    def __init__(self, concentration1, concentration0, copy=False):

        if not copy:
            if concentration0 > 20.:
                self.optim_concentration0 = concentration0.clone().detach().requires_grad_()
            else:
                self.optim_concentration0 = torch.log(torch.exp(concentration0) - 1).clone().detach().requires_grad_()
        else:
            self.optim_concentration0 = concentration0

        super().__init__(concentration1, torch.nn.functional.softplus(self.optim_concentration0))

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration1, self.optim_concentration0]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """

        concentration1, concentration0 = [p.clone().detach().requires_grad_() for p in self.Parameters()]

        return Beta(torch.FloatTensor([concentration1]), torch.FloatTensor([concentration0]), copy=True)


    def log_prob(self, x):
        # print (self.concentration0, self.concentration1)
        # self.concentration1 = 5
        # print ('here', torch.nn.functional.softplus(self.optim_concentration0))
        # print (se)

        # self.concentration0 = torch.nn.functional.softplus(self.optim_concentration0)
        # self.optim_concentration0 = torch.FloatTensor([torch.nn.functional.softplus(self.optim_concentration0)])

        self.concentration0 = torch.nn.functional.softplus(self.optim_concentration0)

        return super().log_prob(x)


    def sample(self):
        return super().sample()

        

def push_addr(alpha, value):
    return alpha + value

def plus(*exp):
    # return torch.sum(torch.tensor(ast))
    return exp[0] + exp[1]

def minus(*exp):
    return exp[0] - exp[1]


def vector(*exp):
    if type(exp) is tuple:
        exp = list(exp)
    try:
        return torch.stack(exp)
    except:
        return exp

def get(*exp):
    try:
        return exp[0][int(exp[1])]
    except:
        return exp[0][exp[1]]

def put(*exp):
    try:
        (exp[0])[int(exp[1])] = exp[2]
    except:
        (exp[0])[exp[1]] = exp[2]
    return exp[0]

def first(*exp):
    return (exp[0])[0]


def last(*exp):
    return (exp[0])[len(exp[0]) - 1]


def rest(*exp):
    return (exp[0])[1:]

def hashmap(*exp):
    try:
        exp = dict((float(exp[i]), exp[i + 1]) for i in range(0, len(exp), 2))
    except:
        exp = dict((exp[i], exp[i + 1]) for i in range(0, len(exp), 2))
    return exp


def append(*exp):
    return torch.cat((exp[0], torch.tensor([exp[1]])), dim=0)


def isempty(*exp):
    return len(exp[0]) == 0


def conj(*exp):
    try:
        return torch.cat((torch.tensor([exp[1]]), exp[0]), dim=0)
    except:
        return torch.cat((torch.tensor([exp[1]]), torch.tensor(exp[0])), dim = 0)

env = {
    'normal' : Normal,
    'beta' : Beta,
    'exponential': dist.Exponential,
    'uniform-continuous' : dist.Uniform,
    'flip' : dist.Bernoulli,
    'discrete': dist.Categorical,
    'push-address' : push_addr,
    '+' : plus,
    '-' : minus,
    '*' : op.mul,
    '/': op.truediv,
    '>' : op.gt,
    '<' : op.lt,
    'sqrt' : torch.sqrt,
    'log' : torch.log,
    'vector': vector,
    'get': get,
    'put': put,
    'first': first,
    'last': last,
    'peek' : last,
    'rest' : rest,
    'append': append,
    'hash-map' : hashmap,
    'procedure?' : callable,
    'empty?' : isempty,
    'conj' : conj
}







