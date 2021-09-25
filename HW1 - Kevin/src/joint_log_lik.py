from scipy.special import loggamma
import numba_scipy
from numba import jit

@jit(nopython=True)
def joint_log_lik(doc_counts, topic_counts, alpha, gamma):
    """
    Calculate the joint log likelihood of the model

    Args:
        doc_counts: n_docs x n_topics array of counts per document of unique topics
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.
    Returns:
        ll: the joint log likelihood of the model
    """

    ll = 0

    for j in range(doc_counts.shape[0]):
        for i in range(doc_counts.shape[1]):
            term_1 = loggamma(doc_counts[j][i] + alpha)
            term_2 = 0
            for k in range(doc_counts.shape[1]):
                term_2 += doc_counts[j][k] + alpha
            term_2 = loggamma(term_2)
            ll += (term_1 - term_2)

    for i in range(doc_counts.shape[1]):
        for v in range(topic_counts.shape[1]):
            term_1 = loggamma(topic_counts[i][v] + gamma)
            term_2 = 0
            for w in range(topic_counts.shape[1]):
                term_2 += topic_counts[i][v] + gamma
            term_2 = loggamma(term_2)
            ll += (term_1 - term_2)

    return ll