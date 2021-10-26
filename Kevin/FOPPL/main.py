from importance_sampler import ImportanceSampler
from mh_gibbs_sampler import MHGibbsSampler
from hmc import HMCSampler
import time

if __name__ == "__main__":
    daphne_input_nums = [1, 2, 5, 6, 7]
    num_samples = 100000
    num_points = 100000  # number of points to plot

    debug_start = 0
    importance_sampler = ImportanceSampler('IS')
    mh_gibbs_sampler = MHGibbsSampler('MH')
    hmc_sampler = HMCSampler('HMC', T=100, epsilon=0.25, M=1)  # TODO: find better initializations
    for idx, num in enumerate(daphne_input_nums[debug_start:], 1):
        print()

        # Likelihood weighting / Importance Sampling
        start = time.time()
        samples = importance_sampler.sample(num_samples, num)
        end = time.time()
        print('Took {0} seconds to finish Program {1}'.format(end - start, idx))

        importance_sampler.summary(num, samples)
        importance_sampler.plot(num, samples, num_points, save_plot=True)

        if num == 2:
            num_samples = 12500

        if num == 5:
            num_samples = 3500


        # MH within Gibbs
        start = time.time()
        samples = mh_gibbs_sampler.sample(num_samples, num)
        end = time.time()
        print('Took {0} seconds to finish Program {1}'.format(end - start, num))

        mh_gibbs_sampler.summary(num, samples)
        mh_gibbs_sampler.plot(num, samples, num_points, save_plot=True)

        # HMC
        # start = time.time()
        # hmc_sampler.sample(num_samples, num)
        # end = time.time()
        #
        # print('Took {0} seconds to finish Program {1}'.format(end - start, num))
        #
        num_samples = 100000

