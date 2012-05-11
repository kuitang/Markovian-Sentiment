import time, pdb, numpy as np
from scipy.special import psi, polygamma
from models import SENTIMENTS, SUBJECTIVITIES

K = len(SUBJECTIVITIES)
S = len(SENTIMENTS)

def discrete_sample(pdf):
    # Unnormalized inverse CDF sampling
    cdf = np.cumsum(pdf)
    return np.flatnonzero(cdf > cdf[-1]*np.random.random())[0] 

@np.vectorize
def inversepsi(y, tol=1e-8):
    # Appendix C in [Minka12]
    if y >= -2.22:
        x = np.exp(y) + 0.5
    else:
        x = -1.0 / (y - psi(1))
    
    # Newton
    while True:
        xold = x
        x = xold - (psi(x) - y) / polygamma(1, x)
        if abs(x - xold) < tol:
            return x

def update_alpha_fp(alpha, theta, sentence_subj, tol=1e-12):
# Fixed point method in [Minka00]
    K = np.size(alpha, 0)
    M, S = np.shape(theta)

    for k in xrange(K):
        theta_k = theta[sentence_subj == k, :]
        log_p = 1.0 / M * np.sum(np.log(theta_k), 0)
        print log_p
        while True:
            oldnorm = np.linalg.norm(alpha[k])
            alpha[k] = inversepsi(psi(np.sum(alpha[k])) + log_p)
            if abs(np.linalg.norm(alpha[k]) - oldnorm) < tol:
                break

    return alpha

#@profile
def update_alpha(alpha, theta, sentence_subj, reset_alpha=False, stepsize=.1, tol=1e-14):
# Newton method in [Minka00]
# NOTE: Need a small stepsize to prevent negative valued alpha
# (I haven't thought about why yet... isn't the log likelihood convex?)
    K = np.size(alpha, 0)
    M, S = np.shape(theta)

#    if reset_alpha:
#        # Use moments to find appropriate initialization
#        for k in xrange(K):
#            theta_k = theta[sentence_subj == k, :]
#            Ep1    = np.mean(theta_k[:, 1])
#            Ep1_sq = np.mean(theta_k[:, 1] ** 2)
#            print 'Ep1', Ep1, 'Ep1_sq', Ep1_sq
#            sum_alpha = (Ep1 - Ep1_sq) / (Ep1_sq - Ep1 ** 2)
#            Epk = np.mean(theta_k, 0)
#            print 'Epk', Epk, 'sum_alpha', sum_alpha
#            alpha[k] = Epk * sum_alpha
#
    for k in xrange(K):
        theta_k = theta[sentence_subj == k, :]
        log_p = 1.0 / M * np.sum(np.log(theta_k), 0)
        while True:
            oldnorm = np.linalg.norm(alpha[k])
#            print np.sum(alpha[k])
#            print log_p
#            print 'alpha[%d] = %s'%(k, alpha[k])
#            print 'log_p[%d] = %s'%(k, log_p)
#            print np.shape(M*psi(np.sum(alpha[k])))
#            print np.shape(M*psi(alpha[k]) )
#            print np.shape(M*log_p)
            g = M*psi(np.sum(alpha[k])) - M*psi(alpha[k]) + M*log_p
            # Diagonal
            q = -M * polygamma(1, alpha[k])
            z = M * polygamma(1, np.sum(alpha[k]))
            b = np.sum(g / q) / (1.0 / z + np.sum(1.0 / q))

            #print "%s - %s"%(alpha[k], stepsize * (g - b) / q)
            alpha[k] -= stepsize * (g - b) / q

            # stupid: flip negative signs
            # No, you can't do that; you break the code!
#            if np.any(alpha[k] < 0):
#                print "Warning: negative alpha; flipping sign", alpha[k]
#                alpha[k][alpha[k] < 0] = -alpha[k][alpha[k] < 0]
    
            if abs(np.linalg.norm(alpha[k]) - oldnorm) < tol:
                break

    assert(np.all(alpha > 0))
    return alpha

def update_one_alpha(alpha, theta, stepsize=.01, tol=1e-14):
# Newton method in [Minka00]
# NOTE: Need a small stepsize to prevent negative valued alpha
# (I haven't thought about why yet... isn't the log likelihood convex?)
    D, K = np.shape(theta)

    log_p = 1.0 / D * np.sum(np.log(theta), 1)
    while True:
        oldnorm = np.linalg.norm(alpha)
        g = D*psi(np.sum(alpha)) - D*psi(alpha) + D*log_p
#        print log_p.shape
        # Diagonal
        q = -D * polygamma(1, alpha)
        z = D * polygamma(1, np.sum(alpha))
        b = np.sum(g / q) / (1.0 / z + np.sum(1.0 / q))
#        print 'g = %s, q = %s, z = %s, b = %s'%(g, q, z, b)
#        print 'g - b = %s', (g - b)
#        print '(g - b) / q = %s', (g - b) / q 
#
#        print np.shape(stepsize)
#        print "%s - %s"%(alpha, stepsize * (g - b) / q)
        alpha -= stepsize * ((g - b) / q)

        if abs(np.linalg.norm(alpha) - oldnorm) < tol:
            break

    assert(np.all(alpha > 0))
    return alpha

def train_lda(blog, iters=400, gamma=None):
    D = len(blog.docs)
    W = len(blog.lexicon)

    # sample of proportions
    theta = blog.doc_counts.transpose() / blog.doc_N
    alpha = np.ones(S)
    alpha = update_one_alpha(alpha, theta, blog.doc_counts)

    print "alpha:", alpha

    if gamma is None: # Set it symmetric for now--data should override.
        gamma = 0.001 # standard value
        print "gamma was set to %g"%gamma

    # Translated from MATLAB for 4240 Homework #5
    print type(iters)
    print "Distribution:", np.histogram(blog.topic_assign, bins=range(8))
    for iter in xrange(iters):
        start = time.time()
        perm = np.random.permutation(blog.n_words)
        blog.words = blog.words[perm]
        blog.doc_belong = blog.doc_belong[perm]
        blog.topic_assign = blog.topic_assign[perm]

        for d in xrange(len(blog.docs)):
            w_d_idxs = blog.doc_belong == d
            w_d = np.flatnonzero(w_d_idxs)

            for i in w_d:
                ti = blog.topic_assign[i]
                wi = blog.words[i]

                # Subtract contribution from word i
                blog.topic_counts[ti, wi] -= 1
                blog.topic_N[ti] -= 1
                blog.doc_counts[d, ti] -= 1

                # Compute the probability vector
                t1 = alpha + blog.doc_counts[d,:].transpose()
                t2_top = gamma + blog.topic_counts[:,wi]
                t2_bot = W * gamma + blog.topic_N
                pdf = t1 * t2_top / t2_bot
                # Remove negative probabilities
#                pdf[pdf < 0] = 0
                z = discrete_sample(pdf)

                blog.topic_assign[i] = z

                blog.topic_counts[z, wi] += 1
                blog.doc_counts[d, z] += 1
                blog.topic_N[z] += 1

        print "Iteration %d: %g seconds"%(iter, time.time() - start)
        print "Distribution:", np.histogram(blog.topic_assign, bins=range(8))

#@profile
def train_subjlda(blog, iters=400, beta=None, gamma=None):
    D = len(blog.docs)
    V = len(blog.lexicon)
    #_, M, T = blog.shape()

    # TUNE PRIORS HERE!
    if beta is None: # Asymmetric
        # Favor neutral about 10x more
        beta = np.ones((S, V)) * 0.01
        beta *= blog.lam

    if gamma is None: # Symmetric
#        # Made this TINY
        L = blog.avg_len()
#        gamma = 0.001
        gamma = (0.05 * L ) / K
        print "gamma was set to %g"%gamma
    
    Nd, Nm, Ndk, Nmj, Njr, Nj = blog.counts

    def update_pi_phi():
        # Equations 5.21, 22, 23
        # Array broadcasting ftw... this is way better than in MATLAB
        pi = np.transpose(np.transpose(Ndk + gamma/K) / (Nd + gamma))
        phi = np.transpose(np.transpose(Njr + beta) / (Nj + np.sum(beta, 1)))
        return pi, phi
    
    pi, phi = update_pi_phi()

    # Initialize alpha from the data (IMPORTANT! Preserve asymmetry.)

    # Initialize theta first, without zero prior (modified Equation 5.22)
    print Nmj, Nm
    theta = np.transpose(np.transpose(Nmj) / Nm) + 1e-10 # prevent zeros
    alpha = np.ones((K,S)) * .01
#    alpha = update_alpha_fp(alpha, theta, blog.subj_assign)
    alpha = update_alpha(alpha, theta, blog.subj_assign)
    print "Initial alpha:"
    print alpha

    def update_theta():
        idxs = np.cast['int'](blog.subj_assign)
        alpha_assigned = alpha[idxs,:]
        assert(np.all(alpha_assigned > 0))
        theta = np.transpose(np.transpose(Nmj + alpha_assigned) / (Nm + np.sum(alpha_assigned)))
        assert(np.all(theta > 0))
        return theta

    # TODO: Perhaps copy later
    WW, DB, SB, SA = blog.words, blog.doc_belong, blog.sent_belong, blog.sent_assign

    # Count each time a word changed sentiments
#    sent_changes = np.zeros_like(WW)
#    # Count each time a sentence changed subjectivity
#    subj_changes = np.zeros_like(blog.sent_to_doc)
#
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
            # Can we vectorize this?
            for j in xrange(S):
                e_518_t2_top *= np.prod(np.arange(0, Nmj[m, j]) + alpha[k, j])
            e_518_t2_bot = np.prod(np.arange(0, Nm[m]) + np.sum(alpha[k, :]))
            e_518_t2 = e_518_t2_top / e_518_t2_bot
            e_518_pdf = e_518_t1 * e_518_t2

            if np.any(np.isnan(e_518_pdf)) or np.all(e_518_pdf == 0):
                # Overflow. TODO: Fix, and understand the equations!
                print "Overflow in sentence %d."%m
#                print "Nmj = %d, Nm = %d, t2_top = %g, t2_bot = %g, pdf = %s"%(
#                        Nmj[m,j], Nm[m], e_518_t2_top, e_518_t2_bot, e_518_pdf)
#                # We know this sentence 
#                del sentidxs[m]
                k_new = k
            else:
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
                e_520_t2 = (Njr[:,r] + beta[:,r]) / (Nj + np.sum(beta, 1))
                e_520_pdf = e_520_t1 * e_520_t2

                j_new = discrete_sample(e_520_pdf)
#                if j_new != j: sent_changes[i] += 1

                SA[i] = j_new
                Nmj[m, j_new] += 1
                Njr[j_new, r] += 1
                Nj[j_new] += 1
        
#        # try a theta/alpha update, after 80 iters
#        if iter % 80 == 79: # use -1 to avoid clobbering our original alpha
#            theta = update_theta()
#            print "New theta:", theta
#            alpha = update_alpha(alpha, theta, blog.subj_assign)
#            print "New alpha:", alpha
#        
        # M-step: MLE estimates
#        if iter % 40 == 39: # use -1 to avoid clobbering our original alpha
#            alpha = update_alpha(alpha, theta, blog.subj_assign)
#            print "New alpha:", alpha
#
#        if iter % 200 == 99:
#            theta = update_theta()
#            print "New theta:", theta
#            print "Min elt:", np.min(theta)

        print "Iteration %d: %g seconds"%(iter, time.time() - start)
        print "              %d subj, %d obj"%(np.sum(blog.subj_assign == 1),
                                               np.sum(blog.subj_assign == 0))
    theta = update_theta()
    pi, phi = update_pi_phi()
    return pi, theta, phi        

class HMM(object): pass

