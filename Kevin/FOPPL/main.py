from Kevin.FOPPL.importance_sampler import ImportanceSampler
from daphne import daphne
from utils import save_ast, load_ast

if __name__ == "__main__":
    daphne_input_nums = [1, 2, 5, 6, 7]

    debug_start = 0
    for num in daphne_input_nums[debug_start:]:
        ast = daphne(['desugar', '-i', '../CPSC532W-HW/Kevin/FOPPL/programs/{}.daphne'.format(num)])
        # save_ast('programs/saved_asts/hw3/program{}.pkl'.format(num), ast)
        ast = load_ast('programs/saved_asts/hw3/program{}.pkl'.format(num))

        sampler = ImportanceSampler()
        sampler.sample(ast)
