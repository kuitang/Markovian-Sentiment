from collections import Counter
from pprint import pprint
from itertools import chain
import cPickle
import array, string, itertools, os, csv
import numpy as np

from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import EnglishStemmer
import nltk.data

stemmer = EnglishStemmer()
STOPWORDS = set((stemmer.stem(w) for w in stopwords.words('english')))

def remove_stopwords(words):
#    return words
    return [ w for w in words if w not in STOPWORDS ]

def word_transform(word):
    # don't stem
    # return word.lower().rstrip(string.punctuation)
    return stemmer.stem(word.lower().strip(string.punctuation))

# stupid tokeniztion
#def word_tokenize(sent):
#    return sent.split()
#

def sent_transform(sent):
    lower = string.lower(sent)
    words = word_tokenize(lower)
#    print "WORDS    ", words
    transformed = map(word_transform, words)
    no_stopwords = remove_stopwords(transformed)
#    print "SEMIFINAL", no_stopwords
    # Get rid of empty words
    return filter(None, no_stopwords)

def get_sentences(text, min_words=10):
    ascii = unidecode(text)
    sentences = filter(None, map(sent_transform, sent_tokenize(ascii)))
    wc = sum(len(s) for s in sentences)
    if wc < min_words:
        return None
    else:
        return sentences

underscore_tr = string.maketrans('_', ' ')
SENTIMENTS = ( 'neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise' )
sentiments = None
def load_wordnetaffect_lite(dirpath=os.path.join('data', 'wordnetaffectlite')):
    global sentiments
    cache_path = os.path.join('data', 'sentiments.cache')
    if os.path.exists(cache_path):
        sentiments = cPickle.load(open(cache_path))
    else:
        sentiments = dict()
        for i, e in enumerate(SENTIMENTS[1:]):
            with open(os.path.join(dirpath, e+'.txt')) as f:
                for r in csv.reader(f, delimiter=' '):
                    # the first column is junk; add the rest
                    # treat bigrams as two unigrams
                    for gram in r[1:]:
                        words = gram.split('_')
                        sentiments.update((word_transform(w), i + 1) for w in words)
        cPickle.dump(sentiments, open(cache_path, 'w'))

#SENTIMENTS = ( 'neutral', 'positive', 'negative' )
#SENTIMENTS = ( 'neutral', 'positive', 'negative' )
#sentiments = None
#
SUBJCLUE_COLS = ( 0, 2, -1 ) # type, word, pole
def load_subjclue(path=os.path.join('data', 'subjclueslen1-HLTEMNLP05.tff')): 
    global sentiments
    cache_path = os.path.join('data', 'sentiments.cache')
    if os.path.exists(cache_path):
        sentiments = cPickle.load(open(cache_path))
    else:
        sentiments = dict()
        for r in csv.reader(open(path), delimiter=' '):
            cols = [ r[c].split('=')[1] for c in SUBJCLUE_COLS ]
            if cols[0] == 'strongsubj':
                sentiments[word_transform(cols[1])] = 1 if cols[2] == 'positive' else 2
        cPickle.dump(sentiments, open(cache_path, 'w'))

SUBJECTIVITIES = ( 'objective', 'subjective' )
subjectivities = None
SENTIWORD_POSSCORE_C = 2
SENTIWORD_NEGSCORE_C = 3
SENTIWORD_SYNSET_C = 4

def load_sentiwordnet(path=os.path.join('data', 'sentiwordnet')):
    global subjectivities
    cache_path = os.path.join('data', 'subjectivities.cache')
    if os.path.exists(cache_path):
        subjectivities = cPickle.load(open(cache_path))
    else:
        subjectivities = dict()
        for r in csv.reader(open(path), delimiter='\t'):
            try:
                ps, ns = float(r[SENTIWORD_POSSCORE_C]), float(r[SENTIWORD_NEGSCORE_C])
            except ValueError as e:
                print "Error: %s. Row was: %s"%(e, r)

            # remove _ grams
            unigrams = string.translate(r[SENTIWORD_SYNSET_C],
                                        underscore_tr)

            # Each word in the synset is wwww#n, so remove last two chars
            words = [ word_transform(w[:-2]) for w in unigrams.split() ]
            for w in words:
                if abs(ps - ns) >= 0.5:
                    if ps > ns:
                        subjectivities[w] = 1
                    else:
                        subjectivities[w] = 2
                else:
                    subjectivities[w] = 0
    cPickle.dump(subjectivities, open(cache_path, 'w'))

# Load stuff
def load():
    if not load.loaded:
        load.loaded = True
        load_sentiwordnet()
#        load_subjclue()
        load_wordnetaffect_lite()
#    print subjectivities.keys()
#    print sentiments.keys()
load.loaded = False

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
        if not self.frozen:
            words_and_counts = self.freqs.most_common()
            print "Lexicon.freeze: We have %d unique words"%len(words_and_counts)
            self.words = [ w for w, _ in words_and_counts ]
            self.worddict = dict((w, i) for i, w in enumerate(self.words))
            self.frozen = True

    def __getitem__(self, word):
        return self.worddict[word]

class Blog(object):
    def __init__(self, min_words_per_post=20):
        self.docs = []
        self.test_docs = []
        self.lexicon = Lexicon()
        self.min_words_per_post = min_words_per_post
        self.reject = 0

    # This should be the only function to see raw text.
    def add_doc(self, text, test_prop=0.2):
        sentences = get_sentences(text, self.min_words_per_post) 
        if sentences:
            self.lexicon.update(w for s in sentences for w in s)
            if np.random.binomial(1, test_prop): # test set instead
                self.test_docs.append(sentences)
            else:
                self.docs.append(sentences)
        else:
            self.reject += 1

    def avg_len(self):
        return sum(len(d) for d in self.docs) / float(len(self.docs))

    def vectorize_test_set(self):
        pass

    def vectorize_lda_doc(self, p_subjective=.1):
        """
        Prepare bag of words for normal LDA in document mode
        """
        print "Blog.freeze: %d posts added, %d rejected."%(
                len(self.docs), self.reject)
        self.lexicon.freeze()
        self.n_words = sum(len(s) for d in self.docs for s in d)
        print self.n_words
        self.words = np.empty(self.n_words)
        self.doc_belong = np.empty_like(self.words)
        self.topic_assign = np.empty_like(self.words)

        # Count variables
        K, D, V = len(SENTIMENTS), len(self.docs), self.n_words
        self.topic_counts = np.zeros((K, V))
        self.doc_counts   = np.zeros((D, K))
        self.topic_N      = np.zeros(K)
        self.doc_N        = np.zeros(D)

        # Initialize the ASYMMETRIC PRIOR!!!
        # Ratio of 0 : 1 is 1 : 10
        self.lam = np.ones((S, V)) * 0.1
        for w, i in self.lexicon.worddict.iteritems():
            k = sentiments.get(w, 0)
            self.lam[k, i] = 1
        
        sent_hits, sent_nonneut_hits, subj_hits, misses = 0, 0, 0, 0
        i = 0
        for id, doc in enumerate(self.docs):
            # Ignore sentences
            for w in chain.from_iterable(doc):
                self.words[i] = self.lexicon[w]
                self.doc_belong[i] = id
                k = None
                if w in sentiments:
                    sent_hits += 1
                    k = sentiments[w]
                    if k != 0: # nonneut
                        sent_nonneut_hits += 1
                        print "#%d: %s was %d"%(i, w, k)
                elif subjectivities.get(w, -1) == 0:  # neut
                    subj_hits += 1
                    k = 0
                else: # no prior knowledge:
                    misses += 1
                    if np.random.binomial(1, p_subjective): # give it subjective
                        k = np.random.randint(1, K)
                    else: # or not
                        k = 0

                self.topic_assign[i] = k

                self.topic_N[k] += 1
                self.topic_counts[k, self.words[i]] += 1
                self.doc_N[id] += 1
                self.doc_counts[id, k] += 1

                i += 1

        print "Initialization: sentiment hits = %d (of which %d non-neutral), subjectivity hits = %d, misses = %d"%(sent_hits, sent_nonneut_hits, subj_hits, misses)
        print "Distribution:", np.histogram(self.topic_assign, bins=range(8))

    def vectorize(self, p_subjective=0.2):
        """
        Prepare bag of sentences/words.
        """
        print "Blog.freeze: %d posts added, %d rejected."%(
                len(self.docs), self.reject)
        self.lexicon.freeze()
        self.n_sentences = sum(len(d) for d in self.docs)
        self.n_words = sum(len(s) for d in self.docs for s in d)
        print self.n_words

        # Index vectors for LDA
        # FML; cannot store these as ints because we need to do floating divison!
        self.words  = np.empty(self.n_words)
        self.doc_belong = np.empty_like(self.words)
        self.sent_belong = np.empty_like(self.words)

        self.sent_to_doc = np.empty(self.n_sentences)

        # Initial sentiment assignments from emotion lexicons
        self.sent_assign = np.empty_like(self.words)
        self.subj_assign = np.empty(self.n_sentences)

        # Count variables (see page 101 of [Lin03])
        # We use a flat index for sentences, for sentence variables omit the
        # d index.
        Nd = np.array([ len(d) for d in self.docs ])
        Nm = np.zeros(self.n_sentences)
        Ndk = np.zeros((len(self.docs), 2)) # two subjectivities
        Nmj = np.zeros((self.n_sentences, len(SENTIMENTS)))
        Njr = np.zeros((len(SENTIMENTS), len(self.lexicon)))
        Nj  = np.zeros(len(SENTIMENTS))
        self.counts = ( Nd, Nm, Ndk, Nmj, Njr, Nj )

        i = 0
        s_i = 0
        
        sent_hits, sent_nonneut_hits, subj_hits, misses = 0, 0, 0, 0
        for id, d in enumerate(self.docs):
            for s in d:
                self.sent_to_doc[s_i] = id
                Nm[s_i] = len(s)

                sent_subj = 0
#                sent_subj = np.random.randint(2)
                for w in s:
                    self.words[i] = self.lexicon[w]
                    self.doc_belong[i] = id
                    self.sent_belong[i] = s_i
                    # TODO
                    if w in sentiments:
                        sent_hits += 1
                        self.sent_assign[i] = sentiments[w]
                        if self.sent_assign[i] != 0: # not neutral
                            sent_subj = 1 
                            sent_nonneut_hits += 1
                    elif subjectivities.get(w, -1) == 0: # neutral
                        subj_hits += 1
                        self.sent_assign[i] = 0
                    else: # if no prior knowledge, random assignment
                        misses += 1
                        self.sent_assign[i] = np.random.randint(0, len(SENTIMENTS))
                        # Prior-prior knowledge is uniform... could potentially improve
                        # Should really randomly assign these proportionally, in
                        # accordance with *some* prior
                        # for some reason this makes seeking alpha diverge...
                        # I know why: you introduce some "singularity" into alpha,
                        # because then ALL objective words are neutral, which is
                        # not always the case!
                        # if self.sent_assign[i] != 0: # not neutral
                        #     sent_subj = 1 
                        #
                        # Correct assignment is randomly assign neutral/nonneutral.

                    Nmj[s_i, self.sent_assign[i]] += 1
                    Njr[self.sent_assign[i], self.words[i]] += 1
                    Nj[self.sent_assign[i]] += 1
                    i += 1

                # TODO
                self.subj_assign[s_i] = sent_subj # Assign subjective or objective
                Ndk[id, sent_subj] += 1
                s_i += 1
        print "Initialization: sentiment hits = %d (of which %d non-neutral), subjectivity hits = %d, misses = %d"%(sent_hits, sent_nonneut_hits, subj_hits, misses)
        print "%d subj, %d obj"%(np.sum(self.subj_assign == 1), np.sum(self.subj_assign == 0))
