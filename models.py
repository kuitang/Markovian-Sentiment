from collections import Counter
from pprint import pprint
import array, string, itertools, os, csv
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import EnglishStemmer
import nltk.data

stemmer = EnglishStemmer()
STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(words):
#    return words
    return [ w for w in words if string.lower(w) not in STOPWORDS ]

def word_transform(word):
    # don't stem
    # return word.lower().rstrip(string.punctuation)
    return stemmer.stem(word.lower().rstrip(string.punctuation))

# stupid tokeniztion
#def word_tokenize(sent):
#    return sent.split()
#
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

underscore_tr = string.maketrans('_', ' ')
SENTIMENTS = ( 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise' )
sentiments = (set(),) * len(EMOTIONS)
def load_wordnetaffect_lite(dirpath=os.path.join('data', 'wordnetaffectlite')):
    for i, e in enumerate(SENTIMENTS):
        with open(os.path.join(dirpath, e+'.txt')) as f:
            for r in csv.reader(f, delimiter=' '):
                # the first column is junk; add the rest
                # treat bigrams as two unigrams
                for gram in r[1:]:
                    words = gram.split('_')
                    sentiments[i].update(word_transform(w) for w in words)

load_wordnetaffect_lite()
pprint(sentiments)

SUBJECTIVITIES = ( 'neutral', 'positive', 'negative' )
subjs = (set(),) * len(SUBJECTIVITIES)
SENTIWORD_POSSCORE_C = 2
SENTIWORD_NEGSCORE_C = 3
SENTIWORD_SYNSET_C = 4

def load_sentiwordnet(path=os.path.join('data', 'sentiwordnet')):
    for r in csv.reader(open(path), delimiter='\t'):
        try:
            ps, ns = float(r[SENTIWORD_POSSCORE_C]), float(r[SENTIWORD_NEGSCORE_C])
        except ValueError as e:
            print "Error: %s. Row was: %s"%(e, r)

        # remove _ grams
        unigrams = string.translate(r[SENTIWORD_SYNSET_C],
                                    underscore_tr)

        # Each word in the synset is wwww#n, so remove last two chars
        words = [ word_transform(w[:-2]) for w in unigrams ]
        for w in words:
            if abs(ps - ns) >= 0.5:
                if ps > ns:
                    subjs[1].add(w)
                else:
                    subjs[2].add(w)
            else:
                subjs[0].add(w)

# Load stuff
load_sentiwordnet()
pprint(subjs)

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
    
    def __len__(self):
        return len(self.freqs)

    def freeze(self):
        words_and_counts = self.freqs.most_common()
        print "Lexicon.freeze: We have %d unique words"%len(words_and_counts)
        self.wordidx = [ w[0] for w in words_and_counts ]
        self.frozen = True
    
    def indices_for_words(words):
        return array.array('i', (wordidx[w] for w in words))

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

    def avg_len(self):
        return sum(len(d) for d in self.docs) / float(len(self.docs))

    def shape(self):
        D = len(self) # number of docs
        M = max(len(d) for d in self.docs) # number of sentences
        T = max(len(s) for s in d for d in self.docs) # number of words
        return (D, M, T)
    
    def freeze(self):
        print "Blog.freeze: %d posts added."%len(self.docs)
        self.lexicon.freeze()
        for d in self.docs: d.numericize()

class Document(object):
    def __init__(self, sentences, lexicon):
        self.sentences = sentences
        self.lexicon = lexicon
        lexicon.update(w for s in self.sentences for w in s)
        print self.sentences
    
    def __len__(self):
        return sum(len(s) for s in sentences)

    def numericize(self):
        self.i_sentences = map(self.lexicon.indices_for_words,
                               self.sentences)

