import jieba
import logging
import math
import os
import requests
import re

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
    def __init__(self, wiki_dump_path, lang, lower=False):
        logging.info('Parsing wiki corpus Altered...')
        logging.info('utils.py line 55, Pattern altered to include [ and ] ...')
        self.wiki = WikiCorpus(wiki_dump_path,lower=lower)
        self.lang = lang

    def __iter__(self):
        for sentence in self.wiki.get_texts():
            sentence = ' '.join(sentence)
            phrases = set(re.findall(r"\[\s[\w\s]+\]",sentence))
            sentence = sentence.replace('[','')
            sentence = sentence.replace(']','')
            for phrase in phrases:
                phrase = phrase.lstrip('[').rstrip(']')
                sentence = re.sub(phrase, ' ['+phrase+'] ', sentence)
            sentence = sentence.split()
            assert ' ' not in sentence
            if self.lang == 'zh':
                yield list(jieba.cut(''.join(sentence), cut_all=False))
            else:
                yield list(sentence)
