import pickle
import torch
from torch.distributions import normal, beta, exponential, uniform, categorical, bernoulli

distributions = ['normal', 'beta', 'exponential', 'uniform', 'discrete', 'bernoulli']

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
    if dist_type == 'bernoulli':
        return bernoulli.Bernoulli(params[0])




def save_ast(file_name, my_ast):
    # faster load

    # saving the list
    open_file = open(file_name, "wb")
    pickle.dump(my_ast, open_file)
    open_file.close()


def load_ast(file_name):

    # loading the list
    open_file = open(file_name, 'rb')
    ret = pickle.load(open_file)
    open_file.close()
    return ret


def substitute_sampled_vertices(expression, variable_bindings):
    if type(expression) is not list:
        if isinstance(expression, str):
            if expression in list(variable_bindings.keys()):
                return variable_bindings[expression]
        return expression

    return [substitute_sampled_vertices(sub_expression, variable_bindings) for sub_expression in expression]


