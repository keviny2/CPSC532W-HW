from evaluation_based_sampling import evaluate_program
from sampler import Sampler
from utils import load_ast
import copy
import torch


class ImportanceSampler(Sampler):

    def __init__(self, method):
        super().__init__(method)

    def sample(self, num_samples, num, summary=True, plot=True):
        """
        importance sampling procedure
        """

        print('=' * 10, 'Likelihood Weighting', '=' * 10)

        ast = load_ast('programs/saved_asts/hw3/program{}.pkl'.format(num))
        sig = {'logW': 0}
        samples = []
        for i in range(num_samples):
            r_i, sig_i = evaluate_program(ast, sig, 'IS')
            logW_i = copy.deepcopy(sig['logW'])
            samples.append([r_i, logW_i])
            sig['logW'] = 0

        if summary:
            self.summary(num, samples)

        if plot:
            pass

    def compute_statistics(self, samples, parameter_names):
        """
        method to compute the posterior expectation and variance for a sequence of samples and sample weights

        :param samples: samples obtained from the sampling procedure
        :param parameter_names: parameter names for printing / plotting
        :return:
        """
        # separate parameter observations and weights from samples
        temp = [elem[0] for elem in samples]
        weights = torch.FloatTensor([elem[1] for elem in samples])

        # initialize empty list that will contain lists of parameter observations
        parameter_traces = []

        # checks if samples only contains a single parameter
        if temp[0].size() == torch.Size([]):
            parameter_traces.append(torch.FloatTensor(temp))
        else:
            for i in range(len(parameter_names)):
                parameter_traces.append(torch.FloatTensor([elem[i] for elem in temp]))

        for i, obs in enumerate(parameter_traces):
            posterior_expectation = torch.dot(obs, torch.exp(weights)) / torch.sum(torch.exp(weights))
            print('Posterior Expectation {}:'.format(parameter_names[i]), posterior_expectation)

            # I think I should use posterior_expectation for the mean...
            posterior_var = torch.dot(torch.pow((obs - posterior_expectation), 2), weights) / torch.sum(weights)
            print('Posterior Variance {}:'.format(parameter_names[i]), posterior_var)


