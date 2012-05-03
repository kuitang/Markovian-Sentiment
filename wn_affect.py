from lxml.etree import parse
from collections import defaultdict
import sys, os, glob, re, cPickle, ctypes

def hash(): return defaultdict(hash)

sys.setrecursionlimit(30)

# Beware of cycles
def children(adj, node, visited=None):
    if visited is None:
        visited = set()

    if node in visited or len(adj[node]) == 0:
        return []
    
    visited.add(node)
    return adj[node] + sum((children(adj, c, visited) for c in adj[node]), [])

def load_wn():
    os.environ['WNSEARCHDIR'] = os.path.join('wordnet-1.6', 'dict')
    libname = glob.glob('libwn.*')[0]
    wn = ctypes.CDLL(libname)
    assert(wn.wninit() == 0)

    # Prepare the functions we will use
    wn.read_synset.restype = ctypes.c_void_p
    wn.FmtSynset.argtypes = [ ctypes.c_void_p, ctypes.c_int ]
    wn.FmtSynset.restype = ctypes.c_char_p

    return wn

POS = { 'n': 1, 'v': 2, 'a': 3, 'r': 4 }
def synset_words(wn, hashid):
    p, off = hashid.split('#')
    synsetptr = wn.read_synset(POS[p], int(off), 0)
    words = wn.FmtSynset(synsetptr, 0)
    return words

def load_wn_affect(dirpath=os.path.join('data', 'wn-affect-1.1')):

    # Parse the emotion hierarchy
    hierarchy_dom = parse(os.path.join(dirpath, 'a-hierarchy.xml'))
    adj = defaultdict(list)
    for cat in hierarchy_dom.findall('.//categ[@isa]'):
        adj[cat.get('isa')].append(cat.get('name'))

    edict = dict()
    emotions = sum((adj[e] for e in adj['emotion']), [])
    for e in emotions:
        edict[e] = e
        for c in children(adj, e):
            edict[c] = e

    # Load WordNet 1.6 library
    wn = load_wn()

    # Parse the synset xml
    emotion_for_word = dict()
    synsets_dom = parse(os.path.join(dirpath, 'a-synsets.xml'))
    for syn in synsets_dom.findall('//*[@categ]'):
        hashid, c = syn.get('id'), syn.get('categ')
        words = synset_words(wn, hashid)
        print c, '->', words
        try:
            e = edict[c]
            #emotion_for_word.update((w, e) for w in words)
        except KeyError:
            print "Warning: %s was not found"%e

if __name__ == "__main__":
    load_wn_affect()

