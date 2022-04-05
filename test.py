import logging
import os

import wiki as w # changed wiki to include '[]'

#from gensim.models import word2vec
from localgensim.gensim2.models import word2vec




#from gensim.models.fasttext import FastText
#from gensim.models.word2vec import Word2Vec # not in use
#from localgensim.gensim2.models.word2vec import Word2Vec # not in use

from tqdm import tqdm

print(w.__file__)
print(word2vec.__file__)

import sys
sys.path.append("../../imports/")
import saver as sv

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
os.makedirs('data/', exist_ok=True)

sentences = sv.load("Wiki_sentences_sp_sample")

logging.info('Training model %s', 'SPX2m100')
model = word2vec.Word2Vec(sentences, cbound=False, tbound=True, bound_type='lr', window=5, sg=1, hs=0, negative=5, size=300,
                          sample=0, workers=1, iter=1, min_count=100)
