import pickle
import torch
from torch.distributions import normal, beta, exponential, uniform, categorical, bernoulli, gamma, dirichlet
from dirac import Dirac

distributions = ['normal', 'beta', 'exponential', 'uniform', 'discrete', 'flip', 'gamma', 'dirichlet', 'dirac']

tasks = ['Gaussian unknown mean problem',
         'Bayesian linear regression',
         'Hidden Markov Model',
         'Bayesian Neural Net']

# dictionary of ordinal numbers for printing purposes
nth = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth"
}


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def get_distribution(dist_type, parameters):
    params = []
    for param in parameters:
        if type(param) is torch.Tensor:
            try:
                params.append(param.numpy().item())
            except:
                params.append(param)
        else:
            params.append(param)

    if dist_type == 'normal':
        return normal.Normal(params[0], params[1])
    if dist_type == 'beta':
        return beta.Beta(params[0], params[1])
    if dist_type == 'exponential':
        return exponential.Exponential(params[0])
    if dist_type == 'uniform':
        return uniform.Uniform(params[0], params[1])
    if dist_type == 'discrete':
        return categorical.Categorical(probs=params[0])
    if dist_type == 'flip':
        return bernoulli.Bernoulli(params[0])
    if dist_type == 'gamma':
        return gamma.Gamma(params[0], params[1])
    if dist_type == 'dirichlet':
        return dirichlet.Dirichlet(params[0])
    if dist_type == 'dirac':
        return Dirac(params[0])


def save_ast(file_name, my_ast):
    """
    saves an ast compiled by daphne in a pickle file

    :param file_name: path to where the saved ast will be saved
    :param my_ast: ast object to be stored
    :return:
    """

    open_file = open(file_name, "wb")
    pickle.dump(my_ast, open_file)
    open_file.close()


def load_ast(file_name):
    """
    loads a saved ast

    :param file_name: path to the pickle file
    :return:
    """

    open_file = open(file_name, 'rb')
    ret = pickle.load(open_file)
    open_file.close()
    return ret


def substitute_sampled_vertices(expression, variable_bindings):
    """
    given an ast, substitute all variable strings with their corresponding values

    :param expression: an ast
    :param variable_bindings: a map between variable names and variable values
    :return:
    """
    if type(expression) is not list:
        if isinstance(expression, str):
            if expression in list(variable_bindings.keys()):
                return variable_bindings[expression]
        return expression

    return [substitute_sampled_vertices(sub_expression, variable_bindings) for sub_expression in expression]


