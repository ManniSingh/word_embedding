from gensim.models import KeyedVectors
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import re
import numpy as np
import pandas as pd
import multiprocessing

wnet = '/home/manni/embs/en_wiki_wnet_epoch5_300.txt'
wnet_model = KeyedVectors.load_word2vec_format(wnet, binary=False)
print('Wnet loaded')
dconf = '/home/manni/wn3.0_sense_vectors.bin'
dconf_model = KeyedVectors.load_word2vec_format(dconf, binary=True)
print('Wnet loaded')
models = [wnet_model,dconf_model]
model_names = ['wnet','dconf']

ws353A = '/home/manni/data/wordsim/EN-WS353.out'
ws353R = '/home/manni/data/wordsim/EN-WSR353.out'
ws353S = '/home/manni/data/wordsim/EN-WSS353.out'
rw = '/home/manni/data/wordsim/rw.out'
sim999 = '/home/manni/data/wordsim/EN-SIM999.out'
turk = '/home/manni/data/wordsim/EN_TRUK.txt'
mturk = '/home/manni/data/wordsim/MTURK-771.out'
rg = '/home/manni/data/wordsim/EN-RG-65.txt'
men = '/home/manni/data/wordsim/EN-MEN-LEM.out'

datasets = [ws353A,ws353R,ws353S,rw,sim999,turk,mturk,rg,men]

import workers

 
print('evaluation started..')


