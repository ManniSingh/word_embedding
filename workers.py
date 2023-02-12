from nltk.corpus import wordnet as wn
import re
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
import numpy as np
  
lemmatizer = WordNetLemmatizer()

import sys
sys.path.append("../../imports/")
import saver as sv

syndef = sv.load('syndef')

def getTuples(sent,WINDOW=3):
    '''
    Generate tuples in a WINDOW.
    
    parameters:
    -----------
    sent: list of str 
    WINDOW: int
    
    returns:
    --------
    pres: list of sorted tuples
    '''
    pres = list() #partial result
    for i,target in enumerate(sent):
        if len(sent)>=i+WINDOW+1:
            right = sent[i+1:i+WINDOW+1]
            for con in right: 
                if con != target:
                    tup = [con,target]
                    tup.sort()
                    pres.append(tuple(tup))
    return pres

def getTriples(sent,WINDOW=3):
    '''
    Generate triples in a WINDOW.
    
    parameters:
    -----------
    sent: list of str 
    WINDOW: int
    
    returns:
    --------
    pres: list of sorted tuples
    '''
    pres = list() #partial result
    for i,target in enumerate(sent):
        if len(sent)>=i+WINDOW+1:
            right = sent[i+1:i+WINDOW+1]
            length = len(right)
            for j,con in enumerate(right): 
                if j+1<length:
                    con2 = right[j+1]
                else:
                    continue
                if con != target and con2 != target:
                    tup = [con2,con,target]
                    tup.sort()
                    pres.append(tuple(tup))               
    return pres

def getBlist(synset):
    nodes = [l.name() for l in synset.lemmas()]
    hypo = [l.name() for h in synset.hyponyms() for l in h.lemmas()]
    nodes = nodes+hypo
    #hyper = lambda s: s.hypernyms()
    #hyper = list(synset.closure(hyper, depth=3))
    #if hyper:
    #    hyper = [l.name() for h in hyper for l in h.lemmas()]
    #    nodes = nodes+hyper
    nodes = set([lemmatizer.lemmatize(node) for node in nodes])
    #text=synset.definition()
    #name = synset.name().split('.')[0]
    #if name not in text:
    #    continue
    for synset_,words in syndef.items(): 
        overlap = words & nodes
        if overlap:
            rem = words-overlap
            ss = [wn.synsets(word) for word in rem]
            ss = [_s for s in ss for _s in s]
            for s in ss:
                if s.lowest_common_hypernyms(synset):
                #if synset in ss:
                    names = set([lemma.name() for lemma in wn.synset(synset_).lemmas()])
                    nodes.update(names)
    hyper = [l.name() for h in synset.hypernyms() for l in h.lemmas()]
    nodes.update(set(hyper))
    return nodes

synset2index = dict()
synglink = dict()

vocab = list(wn.all_synsets())


def init_ss(s2i,sgl):
    global synset2index,synglink 
    synset2index  = s2i
    synglink   = sgl

def getMvector(index):
    '''
    parameters:
    -----------
    index : int 
    
    returns:
    --------
    vector: numpy vector
    '''
    synset = vocab[index]
    vector = np.zeros(len(synset2index))
    connections = list()
    connections.extend([l.synset().name() for l in synset.lemmas()])
    connections.extend([lemma.synset().name() for _target in synset.hyponyms() for lemma in _target.lemmas()])
    connections.extend([lemma.synset().name() for _target in synset.hypernyms() for lemma in _target.lemmas()])
    #connections.extend([lemma.synset().name() for _target in synset.root_hypernyms() for lemma in _target.lemmas()])
    connections.extend(s.name() for s in synset.member_holonyms())
    #connections.extend([lname.synset().name() for lemma in synset.lemmas() for lname in lemma.derivationally_related_forms()])
    #connections.extend([lname.synset().name() for lemma in synset.lemmas() for lname in lemma.pertainyms()])
    #connections.extend([lname.synset().name() for lemma in synset.lemmas() for lname in lemma.antonyms()])
    slinks = synglink[synset.name()]
    connections.extend(list(slinks))
    connections=set([synset2index[ss] for ss in connections])
    #connections.remove(index)
    #n = len(connections)
    for i in connections:
        vector[i]=1
    return vector

var_dict = {}

def init_worker(alpha,iters,P,P_0,M):
    var_dict['alpha'] = alpha
    var_dict['iters'] = iters
    var_dict['P'] = P
    var_dict['P_0'] = P_0
    var_dict['M'] = M

def getVec(index):
    '''
    Executes power equation and returns the vector.
    '''
    alpha = var_dict['alpha']
    iters = var_dict['iters']
    P = var_dict['P']
    P_0 = var_dict['P_0']
    M = var_dict['M']
    P_t = P[index]
    for _ in range(iters):
        P_t = (1-alpha)*P_0[index]+alpha*M*P_t
        P_t = P_t/P_t.sum()
    return np.float32(P_t)
    