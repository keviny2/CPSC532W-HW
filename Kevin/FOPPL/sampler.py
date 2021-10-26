from abc import ABC, abstractmethod


class Sampler(ABC):
    def __init__(self, method):
        self.method = method

    @abstractmethod
    def sample(self, num_samples, num, summary, plot):
        """
        abstract sampling procedure that is algorithm specific

        :param num_samples: number of samples
        :param num: number corresponding to which daphne program was loaded; will affect how we compute summary
        statistics and construct plots because programs have different numbers of return values
        :param summary: bool
        :param plot: bool
        """
        raise NotImplementedError('subclasses must override this method!')

    @abstractmethod
    def compute_statistics(self, samples, parameter_names):
        """
        method to compute the posterior expectation and variance for a sequence of samples and sample weights

        :param samples: samples obtained from the sampling procedure
        :param parameter_names: parameter names for printing / plotting
        :return:
        """
        raise NotImplementedError('subclasses must override this method!')

    def summary(self, num, samples):
        """
        computes posterior expectation and variance

        :param num: number corresponding to which daphne program was loaded; will affect how we compute summary
        :param samples: list of observations obtained from the sampling procedure
        :return:
        """

        if num == 1:
            self.compute_statistics(samples, ['mu'])

        if num == 2:
            self.compute_statistics(samples, ['slope', 'bias'])

        if num == 5:
            self.compute_statistics(samples, ['probability first and second datapoint are in the same cluster'])

        if num == 6:
            self.compute_statistics(samples, ['probability it is raining'])

        if num == 7:
            self.compute_statistics(samples, ['something'])

    def plot(self):
        """
        constructs plots

        """
        raise NotImplementedError('subclasses must override this method!')


