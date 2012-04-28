import numpy as np
from scipy.special import psi, polygamma
import time
from models import sentiments, subjs

K = len(SUBJECTIVITIES)
S = len(EMOTIONS)

# TODO

def discrete_sample(pdf):
    # Unnormalized inverse CDF sampling
    cdf = np.cumsum(pdf)
    return np.flatnonzero(cdf > cdf[-1]*np.random.random())[0]

@np.vectorize
def inversepsi(y, tol=1e-14):
    # Appendix C in [Minka12]
    if y >= -2.22:
        x = np.exp(y) + 0.5
    else:
        y = -1.0 / (y - psi(1))
    
    # Newton
    while True:
        xold = x
        x = xold - (psi(x) - y) / polygamma(1, x)
        if abs(x - xold) < tol:
            return x

def update_alpha(alpha, Njr, Nd, tol=1e-6)
    while True:
        oldnorm = np.linalg.norm(alpha)
    
        for k in xrange(K):
            for s in xrange(S):
                alpha[k,s] *= (np.sum(psi(Njr[
        for i in xrange(np.size(alpha, 0)):
            alpha[i] = 

            
        if abs(np.linalg.norm(alpha) - oldnorm) < tol:
            return alpha

def train_subjlda(blog, iters=800):
    D = len(blog.docs)
    V = len(blog.lexicon)
    #_, M, T = blog.shape()

    doc_assignment, sent_label, subj_label = init_labels(blog) 

    if alpha is None: # Asymmetric
        # Set an initial guess
        alpha = np.random.rand((K, S))
        # normalize
        for k in xrange(K):
            alpha[k] /= np.sum(alpha[k])
    if beta is None: # Asymmetric
        beta = np.ones((S, V)) * 0.01 
    if gamma is None: # Symmetric
        L = blog.avg_len()
        gamma = (0.05 * L ) / K
    
    # Greek life ftw
    pi = np.zeros((D, K))
    theta = np.zeros((blog.n_sentences, S)) # use flat sentences indices
    phi = np.zeros((S, V))

    for iter in xrange(iters):
        # E-step: Gibbs sample
        # Shuffle
        perm = np.random.permutation(blog.n_words)
        WW = blog.words[perm]
        DB = blog.doc_belong[perm]
        SB = blog.sent_belong[perm]
        SA = blog.sent_assign[perm]

        for m in xrange(blog.n_sentences):
            d = blog.sent_to_doc[m]
            k = blog.subj_assign[m]  

            Ndk[d, k] -= 1
            Nd[d] -= 1

            # Equation 5.18
            e_518_t1 = (Ndk[d,:] + gamma/K) / (Nd[d] + gamma)
            b_range = np.arange(0, Nmj[m, j])
            e_518_t2_top = np.prod(np.prod(b_range + alpha[k, :]))
            e_518_t2_bot = np.prod(b_range + np.sum(alpha[k, :]))
            e_518_t2 = e_518_t2_top / e_518_t2_bot
            e_518_pdf = e_518_t1 * e_518_t2
            k_new = discrete_sample(e_518_pdf)

            blog.subj_assign[m] = k_new

            Ndk[d, k_new] += 1
            Nd[d] += 1

            Nm_excl = Nm[m] - 1
            for i in np.flatnonzero(SB == m):
                r = WW[i]
                j = SA[i]

                Nmj[m, j] -= 1
                Njr[j, r] -= 1
                Nj[j] -= 1

                e_520_t1 = (Nmj[m,:] + alpha[k_new,:]) / (Nm_excl + np.sum(alpha[k_new,:]))
                e_520_t2 = (Njr[:,r] + beta[:,r]) / (Nj + np.sum(beta[:,r]))
                e_520_pdf = e_520_t1 * e_520_t2

                j_new = discrete_sample(e_520_pdf)

                SA[i] = j_new
                Nmj[m, j_new] += 1
                Njr[j_new, r] += 1
                Nj[j_new] += 1
        
        # M-step: MLE estimates
        if iter % 40 == 0:
            alpha = update_alpha(blog)
        if iter % 200 == 0:
            # Equations 5.21, 22, 23
            # Array broadcasting ftw... this is way better than in MATLAB
            pi = (Ndk + gamma/K) / (Nd + gamma)
            alpha_assigned = alpha[blog.subj_assign,:]
            theta = (Nmj + alpha_assigned) / (Nm + np.sum(alpha_assigned))
            phi = (Njr + beta) / (Nj + np.sum(beta, 1))

class HMM(object): pass

