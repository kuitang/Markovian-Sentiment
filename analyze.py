import cPickle, sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from unidecode import unidecode
from functools import wraps
from scipy.special import gamma, gammaln
import models, inference

parser = argparse.ArgumentParser(description='Produce output analyses')
parser.add_argument('resultfile')

Nwords = 10
K = len(models.SENTIMENTS)
S = len(models.SUBJECTIVITIES)

# Dimension 0 is sentiment.
# Dimension 1 is rank.

def highlight_sentiment(w):
    highlight_sentiment.k += 1
    if w in models.sentiments:
        kk = models.sentiments[w]
        if highlight_sentiment.k == kk:
            return r'\textcolor{ForestGreen}{\textbf{%s}}'%w
        else:
            return r'\textcolor{BrickRed}{\textbf{%s}}'%w
    return w

def print_words(words):
    print r'\begin{tabular}{ %s }'%('l ' * K)
    print r' \sc & '.join(('',) + models.SENTIMENTS), r'\\'
    print r'\hline'
    for words_by_rank in words.transpose():
        highlight_sentiment.k = -1
        print " & ".join(highlight_sentiment(w) for w in words_by_rank), r'\\'
    print r'\end{tabular}'

def make_baseline_ll(blog):
    logV = np.log(len(blog.lexicon))

    word_freq = np.zeros(len(blog.lexicon))
    sent_freq = np.zeros(K)
    # Recount everything.
    for w in (w for d in blog.docs for s in d for w in s):
        word_freq[blog.lexicon[w]] += 1
        if w in models.sentiments: sent_freq[models.sentiments[w]] +=1
        elif w in models.subjectivities: sent_freq[0] += 1

    word_freq[word_freq == 0] = 1 # prevent zero
    
    def log_likelihood(doc):
        ll = 0.0
        for w in (w for s in doc for w in s):
            ll -= logV
            if w in blog.lexicon.worddict:
                ll += np.log(word_freq[blog.lexicon[w]])

        return ll
    return log_likelihood


def make_common_ll(blog, alpha, phi):
    # My Equation (7). Move the products to the outside to be log'd.

    logV = np.log(len(blog.lexicon))

    def log_likelihood(doc):
        ll = 0.0
        for w in (w for s in doc for w in s):
            if w not in blog.lexicon.worddict: # Assign uniform probability
                ll -= logV
                continue

            v = blog.lexicon[w]
            phi_col = phi[:,v]
            ll -= np.log(np.sum(phi_col))

            this_like = 0

            if len(alpha.shape) == 1:
                alpha_ = [ alpha ]
            else:
                alpha_ = alpha

            for k, alpha_row in enumerate(alpha_):
                I = np.zeros(K)
                I[k] = 1
                t1 = gamma(np.sum(alpha_row)) / np.prod(gamma(alpha_row))
                t2 = np.prod(gamma(alpha_row + I)) / gamma(np.sum(alpha_row + K))
                this_like += t1 * t2
            ll -= this_like
        return ll

    return log_likelihood

def make_subjlda_ll(blog, theta, phi):
    alpha = inference.update_alpha(np.ones((S,K)) * .01, theta, blog.subj_assign)
    
    return make_common_ll(blog, alpha, phi)

def make_lda_ll(blog, mygamma=0.001):
    # Need to compute everything else
    V = len(blog.lexicon)
    D = len(blog.docs)
    phi = np.transpose(np.transpose(blog.topic_counts + mygamma) / (blog.topic_N + V*mygamma))
    theta = (blog.doc_counts.transpose() + 0.01) / (blog.doc_N + D*0.01)
    alpha = np.ones(K) * .01
    # do a few iters
    for i in xrange(10):
        alpha = inference.update_one_alpha(alpha, theta)
        theta = (blog.doc_counts + alpha).transpose() / (blog.doc_N + np.sum(alpha))
    
    print 'alpha = ', alpha
    return make_common_ll(blog, alpha, phi)

def perplexity(log_likelihood, blog):
    top = sum(log_likelihood(d) for d in blog.test_docs)
    bot = sum(len(d) for d in blog.test_docs)
    return np.exp(- top / bot)

def analyze_common(blog):
    n_docs  = len(blog.docs) + len(blog.test_docs)
    n_words = blog.n_words + sum(len(s) for d in blog.test_docs for s in d)
    vocab   = len(blog.lexicon)
    print r'%d & %d & %d \\'%(n_docs, n_words, vocab)
    print '================================'

    log_likelihood = make_baseline_ll(blog)
    print '   BASE PERPLEXITY:', perplexity(log_likelihood, blog)

def analyze_subjlda(blog):
    blog, theta, phi = b['blog'], b['theta'], b['phi']
    analyze_common(blog)

    log_likelihood = make_subjlda_ll(blog, theta, phi)
    print "SUBJLDA PERPLEXITY:", perplexity(log_likelihood, blog)

    phi_idxs = np.argsort(phi, 1)
    words = np.empty((K, Nwords), dtype=object)
    for k, phi_irow in enumerate(phi_idxs):
        words[k] = [ blog.lexicon.words[i] for i in phi_irow[::-1][:Nwords] ]

    print_words(words)

def analyze_lda(b):
    analyze_common(b) 

    log_likelihood = make_lda_ll(b)
    print "    LDA PERPLEXITY:", perplexity(log_likelihood, b)

    idxs = np.argsort(b.topic_counts, 1)
    words = np.empty((K, Nwords), dtype=object)
    for k, irow in enumerate(idxs):
        words[k] = [ b.lexicon.words[i] for i in irow[::-1][:Nwords] ]
    print_words(words)

if __name__ == '__main__':
    models.load()
    args = parser.parse_args()

    name = os.path.split(args.resultfile)[0]
    b = cPickle.load(open(args.resultfile))

    if isinstance(b, dict) and 'phi' in b: # we has subjLDA
        analyze_subjlda(b)
    else:
        analyze_lda(b)

