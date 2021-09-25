import numpy as np
import numba as nb
from numba import jit


@jit(nopython=True)
def sample_topic_assignment(topic_assignment,
                            topic_counts,
                            doc_counts,
                            topic_N,
                            doc_N,
                            alpha,
                            gamma,
                            words,
                            document_assignment):
    """
    Sample the topic assignment for each word in the corpus, one at a time.

    Args:
        topic_assignment: size n array of topic assignments
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        doc_counts: n_docs x n_topics array of counts per document of unique topics

        topic_N: array of size n_topics count of total words assigned to each topic
        doc_N: array of size n_docs count of total words in each document, minus 1

        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distributions over words.

        words: size n array of words
        document_assignment: size n array of assignments of words to documents
    Returns:
        topic_assignment: updated topic_assignment array
        topic_counts: updated topic counts array
        doc_counts: updated doc_counts array
        topic_N: updated count of words assigned to each topic
    """
    for i in range(len(words)):
        doc_idx = document_assignment[i]
        top_idx = topic_assignment[i]

        probabilities = np.zeros(topic_N.shape[0])
        Z = 0
        for k in range(topic_counts.shape[0]):
            if top_idx == k:
                a_kj = doc_counts[doc_idx][k] - 1 + alpha
                b_wk = (len(words[topic_assignment == k]) - 1 + gamma) / (np.sum(topic_counts[k]) - 1 + len(words) * gamma)
            else:
                a_kj = doc_counts[doc_idx][k] + alpha
                b_wk = (len(words[topic_assignment == k]) + gamma) / (np.sum(topic_counts[k]) + len(words)*gamma)
            probabilities[k] = a_kj * b_wk
            Z += a_kj * b_wk

        probabilities /= Z

        # update topic assignment for word i
        topic_assignment[i] = np.argmax(np.random.multinomial(1, probabilities))

        # update topic_counts
        for k in range(len(topic_N)):
            w_k = words[topic_assignment == k]

            topic_counts[k] = np.histogram(w_k, bins=topic_counts.shape[1], range=(-0.5, topic_counts.shape[1] - 0.5))[0]

        # topic_N: array of size n_topics count of total words assigned to each topic
        topic_N = topic_counts.sum(axis=1)

        # update doc_counts
        for d in range(len(doc_N)):
            # histogram counts the number of occurrences in a certain defined bin
            doc_counts[d] = \
                np.histogram(topic_assignment[document_assignment == d], bins=len(topic_N), range=(-0.5, len(topic_N) - 0.5))[0]

    # return topic_assignment, topic_counts, doc_counts, topic_N
    return topic_assignment
