from sampler import Sampler
from graph_based_sampling import sample_from_joint, deterministic_eval
from utils import load_ast, substitute_sampled_vertices
import matplotlib.pyplot as plt
import torch
import re
import copy

class HMCSampler(Sampler):

    def __init__(self, method, T, epsilon, M):
        """

        :param method: sampling type
        :param T: number of leapfrog steps
        :param epsilon: step size
        :param M: prior for sampling momentum
        """
        super().__init__(method)
        self.T = T
        self.epsilon = epsilon
        self.M = torch.tensor(M)
        self.latent_vars = []

    def sample(self, num_samples, num, summary=True, plot=True):
        """
        hamiltonian monte carlo

        """

        print('=' * 10, 'Hamiltonian Monte Carlo', '=' * 10)
        graph = load_ast('programs/saved_asts/hw3/program{}_graph.pkl'.format(num))

        # find latent variables and store in class variable self.latent_vars
        regex_lat = re.compile(r'sample*')
        self.latent_vars = [vertex for vertex in graph[1]['V'] if regex_lat.match(vertex)]

        # initialize latent variables
        _ = sample_from_joint(graph)

        # make each latent variable a tensor of size 1 (for future computational ease) and set requires_grad=True
        for node in self.latent_vars:
            graph[1]['Y'][node] = torch.FloatTensor([graph[1]['Y'][node]])
            graph[1]['Y'][node].requires_grad = True

        # assign a pointer to nodes
        X_0 = graph[1]['Y']

        return self.hmc(X_0, graph, num_samples)


    def hmc(self, X_0, graph, num_samples):
        # make a copy of the original graph
        graph_old = copy.deepcopy(graph)

        samples = []
        for s in range(num_samples):
            R = torch.distributions.Normal(torch.zeros(1), self.M).sample()

            # X_new is a dictionary containing the proposed new variable bindings
            X_new, R_new = self.leapfrog(copy.deepcopy(X_0), R, graph)

            # MH step
            u = torch.rand(1)
            alpha = torch.exp(-self.H(graph, R_new) + self.H(graph_old, R))
            if u < alpha:
                # BUG: never accepts...
                graph[1]['Y'] = X_new

            samples.append(deterministic_eval(substitute_sampled_vertices(graph[2], graph[1]['Y'])))

        return samples


    def leapfrog(self, X, R, graph):
        """
        perform leapfrog integration

        :param X: initial value
        :param graph: graphical model
        :param R: momentum
        :return:
        """
        X_curr = X
        R_half = R - 0.5 * self.epsilon * self.grad_U(graph)

        for t in range(self.T):
            # each latent variable in X_curr will get updated
            for node in self.latent_vars:
                X_curr[node].requires_grad = False  # QUESTION: do we need to record this operation for gradient computation?
                X_curr[node] += self.epsilon * R_half

            R_half -= self.epsilon * self.grad_U(graph)

        for node in self.latent_vars:
            X_curr[node] += self.epsilon * R_half
        R_half -= self.epsilon * self.grad_U(graph)

        return X_curr, R_half

    def grad_U(self, graph):

        # regex patterns to be used to locate observed and latent nodes
        regex_lat = re.compile(r'sample*')

        # make torch track all computations on latent variables for gradient

        # zero out gradients so they don't accumulate
        for node in self.latent_vars:
            graph[1]['Y'][node].grad = None

        # backpropagation
        U = self.compute_U(graph)
        U.backward()  # BUG: the sampler never accepts, maybe it has something to do with gradients getting messed up

        # construct a list of the gradients for each latent variable
        grad = [graph[1]['Y'][node].grad for node in list(graph[1]['Y'].keys()) if regex_lat.match(node)]
        return torch.FloatTensor(grad)

    @staticmethod
    def compute_U(graph):
        """
        returns the log of the joint

        :param graph: graphical model
        :return:
        """
        regex_obs = re.compile(r'observe*')
        regex_lat = re.compile(r'sample*')

        # iterate over every node in the graph to obtain the joint
        log_gamma = 0  # accumulator
        for node in graph[1]['V']:
            # substitute variables with their values
            raw_expression = graph[1]['P'][node]
            expression = substitute_sampled_vertices(raw_expression, graph[1]['Y'])

            # different behavior when it is a sample or observed node
            if regex_obs.match(node):
                log_gamma += deterministic_eval(expression)
            if regex_lat.match(node):
                expression[0] = 'observe*'
                expression.append(graph[1]['Y'][node])
                log_gamma += deterministic_eval(expression)

        return -log_gamma

    def H(self, graph, R):
        """
        use auxiliary variable technique

        :param graph: graphical model
        :param R: momentum
        :return: exp{-U(X) + 0.5*R.T*inv(M)*R}
        """
        if self.M.size() == torch.Size([]):
            return torch.exp(-self.compute_U(graph) +
                             0.5 * torch.matmul(R.T,
                                                torch.FloatTensor([torch.matmul(torch.FloatTensor([self.M]), R)])))

        return torch.exp(-self.compute_U(graph) +
                         0.5 * torch.matmul(R.T,
                                            torch.matmul(torch.inverse(self.M), R)))

    def compute_statistics(self, samples, parameter_names):

        # initialize empty list that will contain lists of parameter observations
        parameter_traces = []

        # checks if samples only contains a single parameter
        if samples[0].size() == torch.Size([]):
            parameter_traces.append(torch.FloatTensor(samples))
        else:
            for i in range(len(parameter_names)):
                parameter_traces.append(torch.FloatTensor([elem[i] for elem in samples]))

        for i, obs in enumerate(parameter_traces):
            posterior_exp = torch.mean(obs)
            self.posterior_exp[parameter_names[i]] = posterior_exp
            print('Posterior Expectation {}:'.format(parameter_names[i]), posterior_exp)

            posterior_var = torch.var(obs)
            self.posterior_var[parameter_names[i]] = posterior_var
            print('Posterior Variance {}:'.format(parameter_names[i]), posterior_var)

    def plot_values(self, samples, parameter_names, num_points, save_plot, num):

        # initialize empty list that will contain lists of parameter observations
        parameter_traces = []

        # checks if samples only contains a single parameter
        if samples[0].size() == torch.Size([]):
            parameter_traces.append(torch.FloatTensor(samples))
        else:
            for i in range(len(parameter_names)):
                parameter_traces.append(torch.FloatTensor([elem[i] for elem in samples]))

        fig, axs = plt.subplots(len(parameter_names), figsize=(8, 6))
        if len(parameter_names) == 1:
            axs = [axs]

        # histograms
        for i, obs in enumerate(parameter_traces):
            axs[i].set_title('{2} posterior exp: {0:.2f}    var: {1:.2f}'.format(self.posterior_exp[parameter_names[i]],
                                                                                 self.posterior_var[parameter_names[i]],
                                                                                 parameter_names[i]))
            axs[i].hist(obs.numpy().flatten())
            axs[i].set(ylabel='frequency', xlabel=parameter_names[i])

        plt.suptitle('Histogram for Program {0} using {1}'.format(num, self.method))
        plt.tight_layout()

        if save_plot:
            plt.savefig('report/HW3/figures/{0}_program_{1}'.format(self.method, num))

        if num in [1, 2, 7]:

            fig, axs = plt.subplots(len(parameter_names), figsize=(8, 6))
            if len(parameter_names) == 1:
                axs = [axs]

            # trace plots
            for i, obs in enumerate(parameter_traces):
                axs[i].plot(obs[-num_points:].numpy().flatten())
                axs[i].set(ylabel=parameter_names[i], xlabel='iterations')

            plt.suptitle('Trace Plots for Program {0} using {1}'.format(num, self.method))
            plt.tight_layout()

            if save_plot:
                plt.savefig('report/HW3/figures/trace_{0}_program_{1}'.format(self.method, num))

            plt.clf()

            # log joint density
            log_p = self.compute_log_density(samples, num)

            # special case, don't want to render the wrong program number
            if num == 7:
                plt.title('Log Joint Density for Program {0} using {1}'.format(5, self.method))
            else:
                plt.title('Log Joint Density for Program {0} using {1}'.format(num, self.method))

            plt.xlabel('iterations')
            plt.ylabel('log joint density')

            plt.plot(log_p.numpy().flatten())

            if save_plot:
                plt.savefig('report/HW3/figures/log_joint_{0}_program_{1}'.format(self.method, num))

    @staticmethod
    def compute_log_density(samples, num):
        """
        computes log density for a set of observations
        :param obs: set of observations
        :param num: identification for which program to run
        :return: trace of log densities
        """
        # load the corresponding program
        graph = load_ast('programs/saved_asts/hw3/program{}_graph.pkl'.format(num))

        regex_obs = re.compile(r'observe*')
        regex_lat = re.compile(r'sample*')

        # iterate over every node in the graph to obtain the joint
        log_ps = []  # accumulator

        # iterate over each observation
        for x in samples:
            # assign parameter values
            if x.size() == torch.Size([]) or x.size() == torch.Size([1]):
                graph[1]['Y'][graph[2]] = x
            else:
                for i in range(len(x)):
                    # BUG: can't do this... not always the case that we have sample1, sample2, ...., samplen
                    graph[1]['Y']['sample{}'.format(i + 1)] = x[i]

            # iterate over nodes and compute log likelihood
            log_p = 0  # accumulator
            for node in graph[1]['V']:
                # substitute variables with their values
                raw_expression = graph[1]['P'][node]
                expression = substitute_sampled_vertices(raw_expression, graph[1]['Y'])

                # different behavior when it is a sample or observed node
                if regex_obs.match(node):
                    log_p += deterministic_eval(expression)
                elif regex_lat.match(node):
                    expression[0] = 'observe*'
                    expression.append(graph[1]['Y'][node])
                    log_p += deterministic_eval(expression)
                else:
                    raise KeyError('Cannot determine expression type')

            log_ps.append(log_p)

        return torch.FloatTensor(log_ps)


