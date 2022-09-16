from gensim.models import KeyedVectors
from tqdm import tqdm

wnet = '/home/manni/embs/en_wiki_wnet_epoch5_300.txt_trimmed'
wnet_model = KeyedVectors.load_word2vec_format(wnet, binary=False)
print('embeddings loaded...')
dconf = '/home/manni/wn3.0_sense_vectors.txt_trimmed'
dconf_model = KeyedVectors.load_word2vec_format(dconf, binary=False)
print('embeddings loaded...')

ws353A = '/home/manni/data/wordsim/EN-WS353.out'
ws353R = '/home/manni/data/wordsim/EN-WSR353.out'
ws353S = '/home/manni/data/wordsim/EN-WSS353.out'
rw = '/home/manni/data/wordsim/rw.out'
sim999 = '/home/manni/data/wordsim/EN-SIM999.out'
turk = '/home/manni/data/wordsim/EN_TRUK.txt'
mturk = '/home/manni/data/wordsim/MTURK-771.out'
rg = '/home/manni/data/wordsim/EN-RG-65.txt'
men = '/home/manni/data/wordsim/EN-MEN-LEM.out'

datasets = [ws353A,rw,sim999,turk,mturk,rg,men]

import workers

def get_eval(model):
    output = list()
    for ds in tqdm(datasets,position=0):
        out = workers.get_corrs(ds,model,maxsim=False)
        output.append(out)
    return output

output = get_eval(dconf_model)
print(output)
output = get_eval(wnet_model)
print(output)
