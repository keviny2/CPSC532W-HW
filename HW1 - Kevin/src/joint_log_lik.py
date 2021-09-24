import numpy as np
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
        for k in range(doc_counts.shape[1]):
            theta_jk = (doc_counts[j][k] + alpha) / (np.sum(doc_counts[j]) + topic_counts.shape[0]*alpha)
            ll += (alpha - 1) * np.log(theta_jk)

    for j in range(doc_counts.shape[0]):
        for w in range(topic_counts.shape[1]):
            # TODO
            ll += 0

    for j in range(doc_counts.shape[0]):
        for w in range(topic_counts.shape[1]):
            # TODO
            ll += 0

    for k in range(doc_counts.shape[1]):
        for w in range(topic_counts.shape[1]):
            phi_kw = (topic_counts[k][w] + gamma) / (np.sum(topic_counts[k]) + topic_counts.shape[1] * gamma)

    return ll