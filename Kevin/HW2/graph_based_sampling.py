import torch
import torch.distributions as dist
import pickle

from daphne import daphne

import primitives
from graph import Graph
from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': dist.Normal,
       'sqrt': torch.sqrt,
       'vector': primitives.vector,
       'sample': primitives.sample
       }


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else:
        raise("Expression type unknown.", exp)


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO: use sampling order to perform ancestral sampling; I will have to evaluate the link functions which
    #  describe the distribution b/c those are your good old expressions from evaluation_based_sampling

    g = Graph(graph[1]['V'])
    for key, value in graph[1]['A'].items():
        g.addEdge(key, value[0])
    sampling_order = g.topologicalSort()

    return torch.tensor([0.0, 0.0, 0.0])


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():

    debug_start = 6
    for i in range(debug_start,13):
        #note: this path should be with respect to the daphne path!
        # graph = daphne(['graph','-i','../CPSC532W-HW/Kevin/HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        graph = load_ast('programs/saved_asts/graph_deterministic{}.pkl'.format(i))
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))

        print('Test passed', graph, 'test', i)
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    num_samples=1e4
    max_p_value = 1e-4

    debug_start = 1
    for i in range(debug_start,7):
        #note: this path should be with respect to the daphne path!        
        # graph = daphne(['graph', '-i', '../CPSC532W-HW/Kevin/HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        graph = load_ast('programs/saved_asts/graph_prob{}.pkl'.format(i))
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(graph)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    

def save_ast(file_name, my_ast):
    # faster load

    # saving the list
    open_file = open(file_name, "wb")
    pickle.dump(my_ast, open_file)
    open_file.close()


def load_ast(file_name):

    # loading the list
    open_file = open(file_name, 'rb')
    ret = pickle.load(open_file)
    open_file.close()
    return ret
        
        
if __name__ == '__main__':
    

    # run_deterministic_tests()
    run_probabilistic_tests()

    for i in range(1,5):
        # graph = daphne(['graph','-i','../CPSC532W-HW/Kevin/HW2/programs/{}.daphne'.format(i)])
        graph = load_ast('programs/saved_asts/daphne_graph{}.pkl'.format(i))
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))

    