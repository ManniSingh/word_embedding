import jieba
import logging
import math
import os
import requests
import spacy
nlp = spacy.load("en_core_web_sm")
#import nltk

from localgensim.gensim2.corpora.wikicorpus import WikiCorpus
from tqdm import tqdm


def download_file(url, file):
    r = requests.get(url, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    logging.info('Downloading %s to %s', url, file)
    with open(file, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size / block_size), unit='KB',
                         unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    logging.info('Done')


def download_wiki_dump(lang, path):
    url = 'https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles-multistream.xml.bz2'
    if not os.path.exists(path):
        download_file(url.format(lang=lang), path)
    else:
        logging.info('%s exists, skip download', path)


class WikiSentences:
    # reference: https://github.com/LasseRegin/gensim-word2vec-model/blob/master/train.py
    def __init__(self, wiki_dump_path, lang, lower=False, pos=False, loc=False):
        logging.info('Parsing wiki corpus Altered...')
        logging.info('utils.py line 55, Pattern altered to include [ and ] ...')
        self.wiki = WikiCorpus(wiki_dump_path,lower=lower)
        self.lang = lang
        self.pos = pos
        self.loc = loc

    def __iter__(self):
        for sentence in self.wiki.get_texts():
            if self.pos:
                #sentence = [tup[0]+'#'+tup[1] if tup[0] not in '[]' else tup[0] for tup in nltk.pos_tag(sentence[:10000])]
                _sentence = nlp(" ".join(sentence[:10000]))
                sentence = [word.text+'#'+word.dep_ if word.text not in '[]' else word.text for word in _sentence]
            if self.loc:
                _sentence = list()
                i = 0
                try:
                    while i<len(sentence):
                        if sentence[i] == '[':
                            _sentence.append(sentence[i])
                            i+=1
                            j = 1
                            while sentence[i] != ']':
                                _sentence.append(sentence[i]+'#'+str(j))
                                i+=1
                                j+=1
                        else:
                            _sentence.append(sentence[i])
                            i+=1
                    sentence = _sentence
                except:
                    sentence = _sentence
                    
            if self.lang == 'zh':
                yield list(jieba.cut(''.join(sentence), cut_all=False))
            else:
                yield list(sentence)
