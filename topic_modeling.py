#!/usr/bin/env python
import keras
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import re
import csv
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.layers import Input, LSTM,merge, RepeatVector,TimeDistributed,Dense,Dropout,Embedding,Masking,Reshape,Lambda
from keras.models import Model
from collections import defaultdict
from itertools import count
from functools import partial
from collections import defaultdict
from keras.models import Sequential
from collections import defaultdict
from keras.preprocessing import text
import nltk
import sys
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import numpy as np
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from sklearn.preprocessing import Normalizer
from keras.optimizers import Adam, RMSprop, SGD
from Get_Amazon_reviews_vectors import write_vec_to_csv
from collections import deque
from keras import optimizers
import tensorflow as tf
from keras import backend as K
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora
nltk.download('punkt')
nltk.download('stopwords')
nltk.download("wordnet")
stop_words = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
reload(sys)
sys.setdefaultencoding('utf8')
def get_data():
    targets = ['cameras','laptops','mobilephone','tablets','TVs','video_surveillance']
    path =['/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_test/cameras',
           ]
    data = []
    target=[]
    filename = []

    for j in  xrange(len(path)):

        print path[j]
        #for i in path[j]:
        for file in os.listdir(path[j]):

            with open(os.path.join(path[j], file), 'r') as f:
                 document = f.read().lower()
                 target.append(targets[j])
                 filename.append(file)
                 data.append(document)

    return data,target,filename
def preprocess(text):

    # doc = []
    # for i in text:
    #     doc.append(' '.join(re.findall(r"[\w']+|[,]", i)))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    doc = text
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    print normalized
    doc = word_tokenize(normalized)
    #doc = keras.preprocessing.text.Tokenizer(num_words=None,lower=True, split=" ").fit_on_texts(doc)
    #print doc
    #doc = [word for word in doc if word.isalpha()]
    # doc = [word for word in doc if word not in stop_words]
    #print doc
    return doc
def get_corpus():
    #stemmer = PorterStemmer()
    data,target,filename = get_data()
    corpus_train_tmp = [preprocess(text) for text in data]
    corpus, data, target, filenames = filter_docs(corpus_train_tmp,data,target, filename, lambda doc: (len(doc) != 0))
    # for i in corpus_train:
    #     print i
    return corpus, target, filenames
def filter_docs(corpus, texts, labels, filenames, condition_on_doc):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """

    number_of_docs = len(corpus)

    for i in range(4):
        tmp = texts
        if i == 0:
            corpus = [corpus for (corpus,tmp) in zip(corpus,tmp) if len(tmp) > 0]
        if i == 1:
            labels = [labels for (labels, tmp) in zip(labels, tmp) if len(tmp) > 0]
        if i == 2:
            filenames = [filenames for (filenames, tmp) in zip(filenames, tmp) if len(tmp) > 0]
        if i == 3:
            texts = [texts for (texts, tmp) in zip(texts, tmp) if len(tmp) > 0]


    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels, filenames)
def document_matrix():
    corpus, target, filenames = get_corpus()
    # Creating the term dictionary of our courpus,
    dictionary = corpora.Dictionary(corpus)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus]

    return doc_term_matrix,dictionary,target,filenames
def LDA_model():
    doc_term_matrix, dictionary, target, filenames = document_matrix()
    Lda = gensim.models.LdaModel
    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=2, id2word=dictionary, passes=50)
    # for i in ldamodel.print_topics(num_words=10):
    #     print i
    print (ldamodel.print_topics(num_words=25))

def main():
    LDA_model()


if __name__ == "__main__":
    main()
