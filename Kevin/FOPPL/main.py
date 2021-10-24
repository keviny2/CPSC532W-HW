from evaluation_based_sampling import evaluate_program
from daphne import daphne
import torch
from utils import save_ast, load_ast
import copy

if __name__ == "__main__":
    daphne_input_nums = [1, 2, 5, 6, 7]
    num_samples = 10000

    debug_start = 0
    for num in daphne_input_nums[debug_start:]:
        # ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/FOPPL/programs/{}.daphne'.format(num)])
        # save_ast('programs/saved_asts/hw3/program{}.pkl'.format(num), ast)
        ast = load_ast('programs/saved_asts/hw3/program{}.pkl'.format(num))

        # Likelihood weighting / Importance Sampling
        print('=' * 10, 'Likelihood Weighting', '=' * 10)
        sig = {'logW': 0}
        samples = []
        for i in range(num_samples):
            r_i, sig_i = evaluate_program(ast, sig, 'IS')
            logW_i = copy.deepcopy(sig['logW'])
            samples.append([r_i, logW_i])
            sig['logW'] = 0

        obs = torch.FloatTensor(list(list(zip(*samples))[0]))
        weights = torch.FloatTensor(list(list(zip(*samples))[1]))
        posterior_expectation = torch.dot(obs, torch.exp(weights)) / torch.sum(torch.exp(weights))
        print('Posterior Expectation of mu:', posterior_expectation)
