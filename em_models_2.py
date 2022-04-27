#!/usr/bin/env python
# coding: utf-8


import logging
import os

#import wiki_old as w # old wiki
import wiki as w # changed wiki to include '[]'
 
#from gensim.models import word2vec # for orignal w2v
from localgensim.gensim2.models import word2vec #remmember to change flags in word2vec.py  161-162



#from gensim.models.fasttext import FastText
#from gensim.models.word2vec import Word2Vec # not in use
#from localgensim.gensim2.models.word2vec import Word2Vec # not in use

from tqdm import tqdm

WIKIXML = '/home/manni/data/wiki/enwiki-20211120-pages-articles-multistream.xml.bz2'
#WIKIXML = '/home/manni/data/wiki/enwiki-20211120-pages-articles-multistream1.xml-p1p41242.bz2'

print(w.__file__)
print(word2vec.__file__)

import sys
sys.path.append("../../imports/")
import saver as sv

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
os.makedirs('data/', exist_ok=True)

# loc = 'num'|'lr'|'ent'
# pos = True|False
# download latest wiki dump
#w.download_wiki_dump('en', WIKIXML)

# parse wiki dump
#wiki_sentences = w.WikiSentences(WIKIXML, 'en',lower=True) # Orignal
#wiki_sentences = w.WikiSentences(WIKIXML, 'en',tokenizer_func='EM',lower=True)
#wiki_sentences = w.WikiSentences(WIKIXML, 'en',tokenizer_func='DEP',lower=True,pos=False,loc=False)
#wiki_sentences = w.WikiSentences(WIKIXML, 'en',tokenizer_func='UNS',lower=True,pos=False,loc=False)
#wiki_sentences = w.WikiSentences(WIKIXML, 'en',tokenizer_func='UNSEM',lower=True,pos=False,loc=False)

#sv.save(wiki_sentences,"wiki_sentences") # orignal
#sv.save(wiki_sentences,"wiki_sentences_em")


# # Phrase mining

#from gensim.test.utils import datapath
#from gensim.models.phrases import Phrases

#phrases = Phrases(sentences, min_count=100, threshold=1)
#frozen_phrases = phrases.freeze()

#sv.save(phrases,"gensim_phrases")

# # Train procedure


sentences = sv.load("wiki_sentences_em") #New

#sentences = sv.load("wiki_sentences_spx") #New
#sentences = sv.load("wiki_sentences") #New

print("Sample sentence:")
for sent in sentences:
    print(sent[:150])
    break

print("Minimum length of token:",sentences.wiki.token_min_len)

logging.info('Training model %s', 'spxM100w5')
model = word2vec.Word2Vec(sentences, window=5, sg=1, hs=0, negative=5, size=300, sample=1e-3, workers=40, iter=5, min_count=100)
logging.info('Training done.')

#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_SPX2_mc1_epoch5_300_filtered_sample.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_w2v_mc1_epoch5_300_sample.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_SPX2R_mc1_epoch5_300_filtered.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2w2v_mc1_epoch5_300.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_SPX2_mc1_epoch5_300_con1.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_SPX2_mc1_epoch5_300_reversed.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_SPX2_mc100_epoch5_300_reversed.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_SPX2_mc100_epoch5_300_neg10.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_SPX2_mc100_epoch5_300_neg10_w3.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_SPX2S_mc100_epoch5_300.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_SPX2B_mc100_epoch5_300_sub3.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2TB_mc100_epoch5_300_LR.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2POS_mc100_epoch5_300.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2DEP_mc100_epoch5_300.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2LRM3_mc100_epoch5_300.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2LOC_mc100_epoch5_300.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2_mc100_epoch5_300.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2_mc100_epoch5_300_w1.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2_mc100_epoch5_300_reversed.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx_mc100_epoch5_300_loc.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_w2v_mc100_epoch5_300_w1.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2_mc100_epoch5_300_pos.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx_mc100_epoch5_300_ent_w10.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2_mc100_epoch5_300_dep2_w1.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2_mc100_epoch5_300_ent_static_w3.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2_mc100_epoch5_300_uns.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2_mc100_epoch5_300_unsem.txt'
emb_file = '/home/manni/embs/en_wiki_spx_mc100_epoch5_300_em_w5_2.txt'
#emb_file = '/mnt/nfs/resdata0/manni/wiki/en_wiki_spx2_mc100_epoch5_300_uns_w1.txt'
#emb_file = '/home/manni/embs/en_wiki_w2v_mc100_epoch5_300.txt'

vocab = model.wv.vocab

print(len(vocab))

vocab.pop('[', None)
vocab.pop(']', None)
print(len(vocab))



logging.info('Save trained word vectors')
with open(emb_file, 'w', encoding='utf-8') as f:
    f.write('%d %d\n' % (len(vocab), 300))
    for word in tqdm(vocab, position=0):
        f.write('%s %s\n' % (word, ' '.join([str(v) for v in model.wv[word]])))
logging.info('Done')

