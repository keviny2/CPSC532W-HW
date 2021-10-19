import torch
from torch.distributions import normal, beta, exponential, uniform, multinomial, bernoulli

my_distributions = ['normal', 'beta', 'exponential', 'uniform', 'discrete', 'bernoulli']

class Distribution:
    def __init__(self, dist_type, params):
        self.dist_type = dist_type
        self.params = params

    def sample(self):
        if self.dist_type == 'normal':
            return torch.tensor([normal.Normal(self.params[1], self.params[2]).sample()])
        if self.dist_type == 'beta':
            return torch.tensor([beta.Beta(self.params[1], self.params[2]).sample()])
        if self.dist_type == 'exponential':
            return torch.tensor([exponential.Exponential(self.params[1]).sample()])
        if self.dist_type == 'uniform':
            return torch.tensor([uniform.Uniform(self.params[1], self.params[2]).sample()])
        if self.dist_type == 'discrete':
            # TODO: don't know exactly what multinomial should return yet... should it be index? or something else?
            return torch.argmax(multinomial.Multinomial(total_count=1, probs=self.params[1]).sample())
        if self.dist_type == 'bernoulli':
            return torch.tensor([bernoulli.Bernoulli(self.params[1])])

