import numpy as np
import time
from models import sentiments, subjs

K = len(SUBJECTIVITIES)
S = len(EMOTIONS)

# TODO
def update_alpha(blogs):
    pass

def train_subjlda(blog, iters=800):
    D = len(blog.docs)
    V = len(blog.lexicon)
    _, M, T = blog.shape()

    doc_assignment, 
    sent_label, subj_label = init_labels(blog) 

    if alpha is None:
        alpha = update_alpha()
    if beta is None:
        beta = np.ones((S, V)) * 0.01 
    if gamma is None:
        L = blog.avg_len()
        gamma = (0.05 * L ) / K

    # Gibbs sample
    for iter in xrange(iters):
        # Shuffle
        perm = np.random.permutation(blog.n_words)
        WW = blog.words[perm]
        DB = blog.doc_belong[perm]
        SB = blog.sent_belong[perm]
        SA = blog.sent_assign[perm]

        for i, wi in enumerate(blog.words):
            d = DB[i]
            m = SB[i]
            k = blog.subj_assign[m]
            j = 
            Ndk[d, k] -= 1
            Nd[d] -= 1
            # Equation 5.18
            t1 = (Ndk[d,:] + gamma) / (Nd[d] + np.sum(gamma))
            b = np.arange(0, Nmj[m,
                        
            



class HMM(object): pass

