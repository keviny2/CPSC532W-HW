import numpy as np
from numpy.random import choice

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
        gamma: prior dirichlet parameter on topic specific distribuitions over words.

        words: size n array of words
        document_assignment: size n array of assignments of words to documents
    Returns:
        topic_assignment: updated topic_assignment array
        topic_counts: updated topic counts array
        doc_counts: updated doc_counts array
        topic_N: updated count of words assigned to each topic
    """

    n_topics = np.shape(topic_counts)[0]
    n_docs = np.shape(doc_counts)[0]

    for d in range(n_docs):
        z_d = topic_assignment[document_assignment == d]
        w_d = words[document_assignment == d]

        for i in range(int(doc_N[d])):
            z_di = z_d[i]
            topic_counts[z_di][w_d[i]] -= 1
            topic_N[z_di] -= 1
            doc_counts[d][z_di] -= 1

            if topic_counts[z_di][w_d[i]] == -1:
                print("topic_counts")
            elif topic_N[z_di] == -1:
                print("topic_N")
            elif doc_counts[d][z_di] == -1:
                print("doc_counts")

            p = (doc_counts[d] + alpha) / np.sum(doc_counts[d] + alpha) * (topic_counts[:, w_d[i]] + gamma) / (np.sum(topic_counts, axis = 1) + gamma)
            p = p / np.sum(p)
            topic = np.random.choice(n_topics,  p = p)
            z_d[i] = topic

            topic_counts[topic][w_d[i]] += 1
            topic_N[topic] += 1
            doc_counts[d][topic] += 1

    return topic_assignment, topic_counts, doc_counts, topic_N