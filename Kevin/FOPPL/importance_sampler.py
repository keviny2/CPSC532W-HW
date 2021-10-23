from evaluation_based_sampling import evaluate_program


class ImportanceSampler:

    def __init__(self):
        pass

    def sample(self, ast, num_samples=1000):

        samples = []
        for i in range(num_samples):
            self.sample_helper(ast)

    def sample_helper(self, ast):
        evaluate_program(ast)
