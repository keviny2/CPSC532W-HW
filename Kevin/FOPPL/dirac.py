import torch
import numpy as np


class Dirac:
    def __init__(self, param):
        """
        implementation of dirac function

        :param param: x + y
        """
        self.param = torch.tensor(param)

    def log_prob(self, obs):
        log_norm_const = torch.log(torch.FloatTensor([2])) + torch.lgamma(torch.FloatTensor([9/8]))

        log_prob = -(self.param - obs)**8 - log_norm_const
        return log_prob

