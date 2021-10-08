import numpy as np
from scipy.special import loggamma

def joint_log_lik(doc_counts, topic_counts, alpha, gamma):
    """
    Calculate the joint log likelihood of the model
    
    Args:
        doc_counts: n_docs x n_topics array of counts per document of unique topics
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribution over words.
    Returns:
        ll: the joint log likelihood of the model
    """
    n_topics = np.shape(doc_counts)[1]
    n_docs = np.shape(doc_counts)[0]
    alphabet_size = np.shape(topic_counts)[1]

    ll = 0

    for k in range(n_topics):

        ll = ll + loggamma(alphabet_size * gamma) + np.sum(loggamma(topic_counts[k] + gamma))
        ll = ll - alphabet_size * loggamma(gamma) - loggamma(np.sum(topic_counts[k] + gamma))


    for d in range(n_docs):
        ll = ll + loggamma(n_topics * alpha) + np.sum(loggamma(doc_counts[d] + alpha))
        ll = ll - n_topics * loggamma(alpha) - loggamma(np.sum(doc_counts[d] + alpha))

    return ll