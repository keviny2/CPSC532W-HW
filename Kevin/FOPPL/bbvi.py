from evaluation_based_sampling import evaluate_program
from sampler import Sampler
from utils import load_ast, cov

import torch


class BBVI(Sampler):
    """
    class for black-box variational inference
    """

    def __init__(self, lr=1e-2):
        super().__init__('BBVI')
        self.lr = lr

    def optimizer_step(self, Q, g_hat):
        """

        :param Q: maps variables to their corresponding distributions
        :param g_hat:
        :return:
        """
        for v in list(g_hat.keys()):
            optimizer = torch.optim.Adam(Q[v].Parameters(), lr=self.lr)
            g_hat[v].backward()
            optimizer.step()
            optimizer.zero_grad()

    def elbo_gradients(self, G, logW):
        """
        compute estimate for ELBO gradient

        :param G: list of gradients of log q -> \grad_{\lambda_{v,d}} \log{q(X_v^l ; \lambda_v}
        :param logW: list of logW of importance weight
        :return: dictionary g_hat that contains gradient components for each variable v
        """
        L = len(G)

        # obtain the union of all gradient maps
        union = []
        for G_i in G:
            union += G_i
        union = list(set(union))

        # dictionary containing our gradient estimates for each variable v
        g_hat = {}

        # create a list of empty dictionaries
        F = [{} for i in range(L)]
        for v in union:
            # tensors for computing b_hat afterwards
            F_one_to_L_v = torch.empty(0)
            G_one_to_L_v = torch.empty(0)
            for l in range(L):
                if v in list(G[l].keys()):
                    F[l][v] = G[l][v] * logW[l]  # TODO: I think the textbook has a typo on this one
                else:
                    F[l][v], G[l][v] = 0, 0
                # saving each F[l][v] and G[l][v] for future computations
                F_one_to_L_v = torch.cat((F_one_to_L_v, F[l][v]), 0)
                G_one_to_L_v = torch.cat((G_one_to_L_v, G[l][v]), 0)

            # this is just the equations on line 16 & 17 in Alg. 12
            b_hat = torch.sum(cov(torch.cat((F_one_to_L_v, G_one_to_L_v), 0))) / torch.sum(torch.var(G_one_to_L_v))
            g_hat[v] = torch.sum(F_one_to_L_v - b_hat * G_one_to_L_v) / L
        return g_hat

    def sample(self, T, L, num):
        """
        perform black-box variational inference

        :param T: number of iterations
        :param L: number of samples to draw when estimating the gradient
        :param num: specify which program we are evaluating
        :return: a weighted set of samples
        """

        print('=' * 10, 'Black-Box Variational Inference', '=' * 10)

        ast = load_ast('programs/saved_asts/hw3/program{}.pkl'.format(num))

        sig = {
            'logW': 0,
            'Q': {},
            'G': {}
        }

        samples = []
        bbvi_loss = []
        for t in range(T):
            G_t = torch.empty(0)
            logW_t = torch.empty(0)
            r_t = torch.empty(0)
            for l in range(L):
                # BUG: sig_tl['G'] will be empty because there was no 'sample' statement that had
                # a variable as one of its arguments
                r_tl, sig_tl = evaluate_program(ast, sig, self.method)
                G_tl, logW_tl = sig_tl['G'], sig_tl['logW']

                # add to return list
                samples.append([r_tl, logW_tl])

                # add to list for self.elbo_gradient and self.optimizer_step functions
                G_t = torch.cat((G_t, G_tl), 0)
                logW_t = torch.cat((logW_t, logW_tl), 0)
                r_t = torch.cat((r_t, r_tl), 0)

            g_hat = self.elbo_gradients(G_t, logW_t)
            bbvi_loss.append(g_hat)  # store loss value for plotting afterwards

            sig['Q'] = self.optimizer_step(sig['Q'], g_hat)

        return samples, bbvi_loss

    def plot_values(self, samples, parameter_names):
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

        fig, axs = plt.subplots(len(parameter_names), figsize=(8, 6))
        if len(parameter_names) == 1:
            axs = [axs]

        # only need to plot histograms of posterior for IS
        for i, obs in enumerate(parameter_traces):
            if parameter_names == ['slope', 'bias']:
                axs[i].set_title('{1} posterior exp: {0:.2f}'.format(self.posterior_exp[parameter_names[i]],
                                                                     parameter_names[i]))
            else:
                axs[i].set_title('{2} posterior exp: {0:.2f}    var: {1:.2f}'.format(self.posterior_exp[parameter_names[i]],
                                                                                     self.posterior_var[parameter_names[i]],
                                                                                     parameter_names[i]))

            if num == 5 or num == 6:
                bin_size = 5
            else:
                bin_size = 2

            axs[i].hist(obs.numpy().flatten(),
                        weights=torch.exp(weights).numpy().flatten(),
                        bins=bin_size * math.ceil(np.max(obs.numpy().flatten()) - np.min(obs.numpy().flatten())))
            axs[i].set(ylabel='frequency', xlabel=parameter_names[i])

        plt.suptitle('Histogram for Program {0} using {1}'.format(num, self.method))
        plt.tight_layout()

        if save_plot:
            plt.savefig('report/HW3/figures/{0}_program_{1}'.format(self.method, num))




