import numpy as np
from scipy.special import psi, polygamma
import time
from models import SENTIMENTS, SUBJECTIVITIES
from pprint import pprint

K = len(SUBJECTIVITIES)
S = len(SENTIMENTS)

# TODO

def discrete_sample(pdf):
    # Unnormalized inverse CDF sampling
#    assert(np.all(pdf >= 0))
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

def update_alpha(alpha, theta, tol=1e-10):
# Newton method in [Minka00]
    K = np.size(alpha, 0)
    M, S = np.shape(theta)

    # Work on one row at a time
    #pprint(np.log(theta))
    #pprint(np.sum(np.log(theta), 0))
    log_p = 1.0 / M * np.sum(np.log(theta), 0)
    #pprint(log_p)
    for k in xrange(K):
        while True:
            oldnorm = np.linalg.norm(alpha[k])
            g = M * psi(np.sum(alpha[k])) - M*psi(alpha[k]) + M*log_p
            q = -M * polygamma(1, alpha[k])
            z = M * polygamma(1, np.sum(alpha[k], 0))
            b = np.sum(g / q) / (1.0 / z + np.sum(1.0 / q))

            alpha[k] -= (g - b) / q
                
            if abs(np.linalg.norm(alpha[k]) - oldnorm) < tol:
                break

    return alpha

def train_subjlda(blog, iters=800, alpha=None, beta=None, gamma=None):
    D = len(blog.docs)
    V = len(blog.lexicon)
    #_, M, T = blog.shape()

    if alpha is None: # Asymmetric
        alpha = np.ones((K,S)) / S

    if beta is None: # Asymmetric
        beta = np.ones((S, V)) * 0.01 
    if gamma is None: # Symmetric
        L = blog.avg_len()
        gamma = (0.05 * L ) / K
    
    Nd, Nm, Ndk, Nmj, Njr, Nj = blog.counts

    def update_pi_theta_phi():
        pi = np.transpose(np.transpose(Ndk + gamma/K) / (Nd + gamma))
#        pprint(np.shape(pi))
        alpha_assigned = alpha[blog.subj_assign,:]
        theta = np.transpose(np.transpose(Nmj + alpha_assigned) / (Nm + np.sum(alpha_assigned)))
#        pprint(np.shape(theta))
        phi = np.transpose(np.transpose(Njr + beta) / (Nj + np.sum(beta, 1)))
#        pprint(np.shape(phi))
        return pi, theta, phi

    pi, theta, phi = update_pi_theta_phi()

    # TODO: Perhaps copy later
    WW, DB, SB, SA = blog.words, blog.doc_belong, blog.sent_belong, blog.sent_assign
    for iter in xrange(iters):
        start = time.time()
        # E-step: Gibbs sample
        # Shuffle
        perm = np.random.permutation(len(WW))
        WW = WW[perm]
        DB = DB[perm]
        SB = SB[perm]
        SA = SA[perm]

        for m in xrange(blog.n_sentences):
            d = blog.sent_to_doc[m]
            k = blog.subj_assign[m]  

            Ndk[d, k] -= 1
            Nd[d] -= 1

            # Equation 5.18
            e_518_t1 = (Ndk[d,:] + gamma/K) / (Nd[d] + gamma)
            e_518_t2_top = 1.0
            for j in xrange(S):
                e_518_t2_top *= np.prod(np.arange(0, Nmj[m, j]) + alpha[k, j])
            e_518_t2_bot = np.prod(np.arange(0, Nm[m]) + np.sum(alpha[k, :]))
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
        if iter % 20 == 0:
            alpha = update_alpha(alpha, theta)
        if iter % 100 == 0:
            # Equations 5.21, 22, 23
            # Array broadcasting ftw... this is way better than in MATLAB
            pi, theta, phi = update_pi_theta_phi()
        print "Iteration %d took %g seconds"%(iter, time.time() - start)
    return pi, theta, phi        

class HMM(object): pass

