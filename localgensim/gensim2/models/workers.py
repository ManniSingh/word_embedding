'''
Multiprocessing helper for eval notebook.
'''

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def avg_sim(w1,w2,model,verbose=False,maxsim=False):
    range_ = list()
    vocab = model.vocab
    a = list()
    for word in vocab:
        if word.startswith(w1+'#'):
            a.append(word)
    if w1 in vocab:
        a.append(w1)
    b = list()
    for word in vocab:
        if word.startswith(w2+'#'):
            b.append(word)
    if w2 in vocab:
        b.append(w2)
    div = len(a)*len(b)
    if div == 0:
        return 0, range_
    sims = list()
    for i in a:
        for j in b:
            sim = model.similarity(i,j)
            range_.append(sim)
            if verbose:
                print(i,j,sim)
            sims+=[sim]
    if maxsim:
        return max(sims), range_
    else:
        return sum(sims)/div, range_
  
def get_corrs(lines,model):
    sims = list()
    scores = list()
    for line in lines:
        line = line.split()
        if not line or len(line)>3:
            continue
        sim, _ = avg_sim(line[0],line[1],model)
        if sim:
            score = float(line[2])
            sims.append(sim)
            scores.append(score)
    assert len(sims)==len(scores) and len(sims)>0
    coverage = round((len(sims)/len(lines))*100,2)
    pcorr, _ = pearsonr(sims, scores)
    scorr, _ = spearmanr(sims, scores)
    return '%.2f' % (pcorr*100),'%.2f' % (scorr*100), '%.2f' % (coverage)
    
