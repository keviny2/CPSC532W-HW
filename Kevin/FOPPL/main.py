from importance_sampler import ImportanceSampler
from mh_gibbs_sampler import MHGibbsSampler
import time

if __name__ == "__main__":
    daphne_input_nums = [1, 2, 5, 6, 7]
    num_samples = 10000

    debug_start = 0
    importance_sampler = ImportanceSampler('IS')
    mh_gibbs_sampler = MHGibbsSampler('MH')
    for num in daphne_input_nums[debug_start:]:
        # Likelihood weighting / Importance Sampling
        importance_sampler.sample(num_samples, num)

        # MH within Gibbs
        mh_gibbs_sampler.sample(num_samples, num)




