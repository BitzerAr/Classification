import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import codecs
import math

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
from pylab import rcParams
from gensim import corpora, models, similarities
from collections import Counter
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

import seaborn as sns
import base64
import string
import re
from collections import defaultdict


from gensim.models import KeyedVectors

# Display plots in this notebook, instead of externally. 
from pylab import rcParams
rcParams['figure.figsize'] = 16, 8
#%matplotlib inline
nlp = spacy.load('es')
filepath = 'sample_esp.txt'  
a= []
with open(filepath) as fp:  
    line = fp.readline()
    cnt = 1
    while line:
        #print("Line {}: {}".format(cnt, line.strip()))
        line = fp.readline()
        cnt += 1
        a.append(line)
a = [item.split('?')[0] for item in a]
corpus = nlp('\n'.join(a))

stoplist = set('el un la los una las Cada tu mi y en que de al'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in a]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 0] for text in texts]
dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict') 
print(dictionary)
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)
tfidf = models.TfidfModel(corpus)
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)

vec = "patas arana"
new_doc = dictionary.doc2bow(vec.lower().split())
sims = index[tfidf[new_doc]]
print(list(enumerate(sims)))


