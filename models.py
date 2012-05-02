from collections import Counter
from pprint import pprint
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

print STOPWORDS

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

def get_sentences(text, min_words=20):
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
                        sentiments.update((word_transform(w), i) for w in words)
        cPickle.dump(sentiments, open(cache_path, 'w'))

SUBJECTIVITIES = ( 'neutral', 'positive', 'negative' )
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
            self.worddict = dict(words_and_counts)
            self.words = [ w for w, _ in words_and_counts ]
            self.frozen = True

    def __getitem__(self, word):
        return self.worddict[word]

class Blog(object):
    def __init__(self, min_words_per_post=20):
        self.docs = []
        self.lexicon = Lexicon()
        self.min_words_per_post = min_words_per_post
        self.reject = 0

    # This should be the only function to see raw text.
    def add_doc(self, text):
        sentences = get_sentences(text, self.min_words_per_post) 
        if sentences:
            self.docs.append(sentences)
            self.lexicon.update(w for s in sentences for w in s)
        else:
            self.reject += 1

    def avg_len(self):
        return sum(len(d) for d in self.docs) / float(len(self.docs))

    def vectorize(self):
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
        self.words  = np.empty(self.n_words, 'int')
        self.doc_belong = np.empty_like(self.words)
        self.sent_belong = np.empty_like(self.words)

        self.sent_to_doc = np.empty(self.n_sentences, 'int')

        # Initial sentiment assignments from emotion lexicons
        self.sent_assign = np.empty_like(self.words)
        self.subj_assign = np.empty(self.n_sentences, 'int')

        # Count variables (see page 101 of [Lin03])
        # We use a flat index for sentences, for sentence variables omit the
        # d index.
        Nd = np.array([ len(d) for d in self.docs ], 'int')
        Nm = np.zeros(self.n_sentences, 'int')
        Ndk = np.zeros((len(self.docs), 2), 'int') # two subjectivities
        Nmj = np.zeros((self.n_sentences, len(SENTIMENTS)), 'int')
        Njr = np.zeros((len(SENTIMENTS), len(self.lexicon)), 'int')
        Nj  = np.zeros(len(SENTIMENTS), 'int')
        self.counts = ( Nd, Nm, Ndk, Nmj, Njr, Nj )

        i = 0
        s_i = 0
        
        sent_hits, sent_nonneut_hits, subj_hits, misses = 0, 0, 0, 0
        for id, d in enumerate(self.docs):
            for s in d:
                self.sent_to_doc[s_i] = id
                Nm[s_i] = len(s)

                sent_subj = 0
                for w in s:
                    self.words[i] = self.lexicon[w]
                    self.doc_belong[i] = id
                    self.sent_belong[i] = s_i
                    # TODO
                    if w in sentiments:
                        sent_hits += 1
                        st = sentiments[w]
                        if st != 0: # not neutral
                            sent_subj = 1 
                            sent_nonneut_hits += 1
                        self.sent_assign[i] = st
                    elif subjectivities.get(w, -1) == 0: # neutral
                        subj_hits += 1
                        self.sent_assign[i] = 0
                    else: # if no prior knowledge, random assignment
                        misses += 1
                        self.sent_assign[i] = np.random.randint(0, len(SENTIMENTS))
                        if self.sent_assign[i] != 0:
                            sent_subj = 1

                    Nmj[s_i, self.sent_assign[i]] += 1
                    Njr[self.sent_assign[i], self.words[i]] += 1
                    Nj[self.sent_assign[i]] += 1
                    i += 1

                # TODO
                self.subj_assign[s_i] = sent_subj # Assign subjective or objective
                Ndk[id, sent_subj] += 1
                s_i += 1
        print "Initialization: sentiment hits = %d (of which %d non-neutral), subjectivity hits = %d, misses = %d"%(sent_hits, sent_nonneut_hits, subj_hits, misses)
