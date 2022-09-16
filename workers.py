'''
Multiprocessing helper for eval notebook.
'''

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from multiprocessing import Pool
from tqdm import tqdm

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
    
def get_score(line,model,maxsim):
    line = line.split()
    if not line or len(line)>3:
        return 0
    sim, _ = avg_sim(line[0],line[1],model,maxsim=maxsim)
    return sim
    
def get_scores(lines,model,maxsim):
    with Pool(processes=20) as pool:
        results = list(pool.apply_async(get_score, args=(line,model,maxsim)) for line in lines)
        results = [r.get() for r in results]
    return results
 
def get_corrs(ds,model,maxsim=False):
    with open(ds) as fin:
        name = ds.split('/')[-1].split('.')[0]
        lines = fin.readlines()
        _sims = get_scores(lines,model,maxsim)
        sims = list()
        scores = list()
        for i,sim in enumerate(_sims):
            if sim!=0:
                score = float(lines[i].split()[2])
                sims.append(sim)
                scores.append(score)
        assert len(sims)==len(scores) and len(sims)>0
        coverage = round((len(sims)/len(lines))*100,2)
        pcorr, _ = pearsonr(sims, scores)
        scorr, _ = spearmanr(sims, scores)
    return '%.2f' % (pcorr*100),'%.2f' % (scorr*100), '%.2f' % (coverage)
    
