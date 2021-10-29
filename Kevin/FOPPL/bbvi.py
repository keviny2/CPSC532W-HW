from evaluation_based_sampling import evaluate_program
from sampler import Sampler
from utils import load_ast, cov

import torch


class BBVI(Sampler):
    """
    class for black-box variational inference
    """

    def __init__(self, lr=1e-2, optimizer=torch.optim.Adam):
        super().__init__('BBVI')
        self.lr = lr
        self.optimizer = optimizer  # TODO: have a dictionary for my optimizers!!! :)

    def optimizer_step(self, Q, g_hat):
        """

        :param Q: maps variables to their corresponding distributions
        :param g_hat:
        :return:
        """
        for v in list(g_hat.keys()):
            pass


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

    def BBVI(self, T, L, num):
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

        r = torch.empty(0)
        logW = torch.empty(0)
        for t in range(T):
            G_t = torch.empty(0)
            logW_t = torch.empty(0)
            r_t = torch.empty(0)
            for l in range(L):
                r_tl, sig_tl = evaluate_program(ast, sig, self.method)
                G_tl, logW_tl = sig_tl['G'], sig_tl['logW']

                # add return values to list
                G_t = torch.cat((G_t, G_tl), 0)
                logW_t = torch.cat((logW_t, logW_tl), 0)
                r_t = torch.cat((r_t, r_tl), 0)

            g_hat = self.elbo_gradients(G_t, logW_t)
            sig['Q'] = self.optimizer_step(sig['Q'], g_hat)

            # append to list that we will return
            r = torch.cat(r, r_t)
            logW = torch.cat(logW, logW_t)

        return r, logW

