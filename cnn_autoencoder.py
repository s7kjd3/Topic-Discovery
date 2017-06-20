#!/usr/bin/env python
import keras
import os
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import re
import csv
from keras.layers import Input, LSTM, RepeatVector,TimeDistributed,Dense,Dropout,Embedding,GRU,Conv1D,MaxPooling1D,UpSampling1D
from keras.models import Model
from collections import defaultdict
from itertools import count
from functools import partial
from collections import defaultdict
from keras.models import Sequential
from keras.layers import LSTM
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
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import Normalizer
from keras.optimizers import Adam, RMSprop, SGD
from Get_reviews_vectors import write_vec_to_csv
reload(sys)
sys.setdefaultencoding('utf8')
def get_data():
    targets = ['cameras','laptops','mobilephone','tablets','TVs','video_surveillance']
    path =['/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/cameras',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/laptops',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/mobilephone',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/tablets',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/TVs',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/video_surveillance']
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

    #print len(target)
    return data,target,filename
def preprocess(text):
    text = text.lower()
    doc = ' '.join(re.findall(r"[\w']+|[.,!?;/-]", text))
    #print doc
    doc = word_tokenize(doc)
    #doc = keras.preprocessing.text.Tokenizer(num_words=None,lower=True, split=" ").fit_on_texts(doc)
    #print doc
    #doc = [word for word in doc if word.isalpha()]
    #doc = [word for word in doc if word not in stop_words]
    #print doc
    return doc
def get_corpus():
    #stemmer = PorterStemmer()
    data,target,filename = get_data()
    # corpus_train_tmp = [preprocess(text) for text in data]
    corpus_train_tmp = data
    #filter empty docs
    corpus_train, data, target, filenames = filter_docs(corpus_train_tmp,data,target, filename, lambda doc: (len(doc) != 0))
    # for i in corpus_train:
    #     print i
    return corpus_train, target, filenames
def filter_docs(corpus, texts, labels, filenames, condition_on_doc):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """

    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]
    filenames = [filename for (filename,doc) in zip(filenames, corpus) if condition_on_doc(doc)]
    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels, filenames)
def string_to_integer(strings):
    string_to_number = {string: i for i, string in enumerate(set(strings), 1)}
    test = [(string_to_number[string], string) for string in strings]
    # string_to_number = defaultdict(partial(next, count(1)))
    # test = [(string_to_number[string], string) for string in strings]
    return test
def preprocess_embedding():
    corpus_train, target, filenames = get_corpus()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus_train)
    sequences = tokenizer.texts_to_sequences(corpus_train)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    MAX_SEQUENCE_LENGTH = 50
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    #loading google vectors
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/home/flippped/Desktop/xiangmu/baseline/GoogleNews-vectors-negative300.bin.gz', binary=True)
    word2vec_model.init_sims(replace=True)

    # create one matrix for documents words
    EMBEDDING_DIM = 300
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    print embedding_matrix.shape
    for word, i in word_index.items():
            try:
                embedding_vector = word2vec_model[str(word)]
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

            except:
                continue


    return data,target,filenames,embedding_matrix, word_index

def lstm():
    data,  targets, filenames, embedding_matrix, word_index = preprocess_embedding()

    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 50
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable= False,
                                name='layer_embedding') #mask_zero=True,


    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = Conv1D(150, 5, activation='relu',name='conv_1')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(75, 5, activation='relu',name='conv_2')(x)
    x = MaxPooling1D(5)(x)
    encoded = Conv1D(30, 5, activation='relu', name='conv_3')(x)
    #encoded = MaxPooling1D(2)(x)
    #decoded = RepeatVector(MAX_SEQUENCE_LENGTH,name='layer_repeat')(encoded)
    x = Conv1D(30, 5, activation='relu', name='conv_3')(encoded)
    x = UpSampling1D(5)(x)
    x = Conv1D(75, 5, activation='relu',name='conv_4')(x)
    x = UpSampling1D(5)(x)
    x = Conv1D(150, 5, activation='relu',name='conv_5')(x)
    x = UpSampling1D(5)(x)
    x = Conv1D(300, 5, activation='relu',name='conv_6')(x)
    x = UpSampling1D(5)(x)
    decoded = Dense(300,activation='linear',name='linear_layer')(x)
    autoencoder = Model(sequence_input, decoded)
    #print sequence_autoencoder.get_layer('lstm_6').output
    encoder = Model(sequence_input, encoded)
    autoencoder.compile(loss='cosine_proximity',
                  optimizer='sgd')#, metrics=['acc'])
    intermediate_layer_model = Model(inputs=autoencoder.input,
                                     outputs=autoencoder.get_layer('layer_embedding').output)
    print autoencoder.get_layer('layer_embedding').output
    #sequence_autoencoder.fit(a, sequence_autoencoder.get_layer('layer_embedding').output,epochs=2)
    autoencoder.fit(data, intermediate_layer_model.predict(data), epochs=20)
    autoencoder.save_weights('embedded_weights_conv.h5')



def main():
 lstm()
 #preprocess_embedding()
 #embedding()


if __name__ == "__main__":
 main()
