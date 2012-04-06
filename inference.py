import numpy as np
import scipy.sparse
from models import sentiments, subjs

def init_alpha(blogs):
    pass

def init_beta(blogs):

    pass

def init_gamma(blogs):
    pass

class SubjLDA(object):
    def __init__(self): pass
    
    def train(self, blog, alpha=None, beta=None, gamma=None, max_iter=800): 
        D = len(blog.docs)
        K = len(SUBJECTIVITIES)
        S = len(EMOTIONS)
        V = len(blog.lexicon)
        _, M, T = blog.shape()
        subj_labels = scipy.sparse.lil_matrix(shape=(D, M))
        sent_labels = scipy.sparse.lil_matrix(shape=(D, M, T))
        # Initialize the labels
#        for d in blog.docs:
#            for m in d.i_sentences:
#
#
#        # Initializations follow those in [Lin03].
        if alpha is None:

        if beta is None:
            beta = np.ones((S, V)) * 0.01 
        if gamma is None:
            L = blog.avg_len()
            gamma = (0.05 * L ) / K

            


class HMM(object): pass
