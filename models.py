from collections import Counter
import string, itertools

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk.tokenize, nltk.data
# For now, don't add stemming.

STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(words):
#    return words
    return [ w for w in words if string.lower(w) not in STOPWORDS ]

class Lexicon(object):
    class FrozenError(Exception):
        def __str__(self): return "Cannot update a frozen Lexicon."
    class NotFrozenError(Exception):
        def __str__(self): return "Cannot vectorize from a non-frozen Lexicon."

    def __init__(self):
        self.freqs = Counter()
        self.frozen = False

    def update(self, words):
        if self.frozen: raise FrozenError
        self.freqs.update(words)
    
    def freeze(self):
        words_and_counts = self.freqs.most_common()
        print "Lexicon.freeze: We have %d unique words"%len(words_and_counts)
        self.wordidx = [ w[0] for w in words_and_counts ]
        self.frozen = True

    # Returns a sparse vector w where 
    # w[i] = # of occurences of wordidx[i] in words
    def vectorize(self, words):
        pass

def word_transform(word):    return word.lower().rstrip(string.punctuation)

def sent_transform(sent):
    return [ word_transform(w) for w in remove_stopwords(word_tokenize(sent)) if
                not (all(c in string.punctuation for c in w)) ]

def get_sentences(text, min_words=20):
    sentences = filter(None, map(sent_transform, sent_tokenize(text)))
    wc = sum(len(s) for s in sentences)
    if wc < min_words:
        return None
    else:
        return sentences

class Blog(object):
    def __init__(self, min_words_per_post=20):
        self.docs = []
        self.lexicon = Lexicon()
        self.min_words_per_post = min_words_per_post

    # This should be the only function to see raw text.
    def add_doc(self, text):
        sentences = get_sentences(text, self.min_words_per_post) 
        if sentences:
            self.docs.append(Document(sentences, self.lexicon))
#        else:
#            print "Document rejected: did not have %d words."%(
#                    self.min_words_per_post, text)

    def freeze(self):
        print "Blog.freeze: %d posts added."%len(self.docs)
        self.lexicon.freeze()
        for d in self.docs: d.vector()

class Document(object):
    def __init__(self, sentences, lexicon):
        self.sentences = sentences
        self.lexicon = lexicon
        lexicon.update(w for s in self.sentences for w in s)
        print self.sentences
    
    def vector(self):
        if not hasattr(self, '_vector'):
            self._vector = self.lexicon.vectorize(w for s in self.sentences for w in s)
        return self._vector

