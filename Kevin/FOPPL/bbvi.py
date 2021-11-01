from evaluation_based_sampling import evaluate_program
from sampler import Sampler
from utils import load_ast, cov, create_fresh_variables, clone
from distributions import Normal

import torch


class BBVI(Sampler):
    """
    class for black-box variational inference
    """

    def __init__(self, lr=1e-2, family=Normal, optimizer=torch.optim.Adam):
        super().__init__('BBVI')
        self.lr = lr
        self.family = family
        self.optimizer = optimizer

    def optimizer_step(self, sig, g_hat):
        """

        :param sig: map containing state of bbvi
        :param g_hat:
        :return:
        """
        for v in list(g_hat.keys()):
            sig['O'][v].step()
            sig['O'][v].zero_grad()

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
            union += list(G_i.keys())
        union = list(set(union))

        # dictionary containing our gradient estimates for each variable v
        g_hat = {}

        for v in union:
            # tensors for computing b_hat afterwards
            F_one_to_L_v = torch.empty(0)
            G_one_to_L_v = torch.empty(0)
            for l in range(L):
                if v in list(G[l].keys()):
                    F_l_v = G[l][v] * logW[l]  # TODO: I think the textbook has a typo on this one
                else:
                    F_l_v, G[l][v] = 0, 0
                # saving each F_l_v and G[l][v] for future computations
                F_one_to_L_v = torch.cat((F_one_to_L_v, F_l_v), 0)
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
        ast = create_fresh_variables(ast)

        sig = {
            'logW': 0,
            'Q': {},
            'G': {},
            'O': {},   # map to optimizer for each variable
            'family': self.family,
            'optimizer': self.optimizer,  # optimizer type (e.g. Adam)
            'lr': self.lr
        }

        samples = []
        bbvi_loss = []
        for t in range(T):
            G_t = []  # each element of G_t will contain the gradient FOR EACH parameter
                         # (so if family is normal, there will be 2 parameters; one for loc and one for scale)
            logW_t = []
            r_t = []
            for l in range(L):
                # r_tl is the return value of the expression (won't have a value for each parameter)
                r_tl, sig_tl = evaluate_program(ast, sig, self.method)

                G_tl = clone(sig_tl['G'])
                logW_tl = sig_tl['logW'].clone()

                # add to return list
                samples.append([r_tl, logW_tl])

                # add to list for self.elbo_gradient and self.optimizer_step functions
                G_t.append(G_tl)
                logW_t.append(logW_tl)
                r_t.append(r_tl)

            g_hat = self.elbo_gradients(G_t, logW_t)
            bbvi_loss.append(g_hat)

            self.optimizer_step(sig, g_hat)

        return samples, bbvi_loss

