#!/usr/bin/env python
import keras
import os
import ast
import json
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import re
import csv
from ast import literal_eval
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.layers import Input, LSTM, RepeatVector,TimeDistributed,Dense,Dropout,Embedding
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
from sklearn.feature_extraction.text import TfidfVectordizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import numpy as np
from nltk import word_tokenize,PorterStemmer
from nltk import download
from nltk.corpus import stopwords
from sklearn.preprocessing import Normalizer
from keras.optimizers import Adam, RMSprop, SGD
from Get_Amazon_reviews_vectors import write_vec_to_csv
reload(sys)
stop_words = stopwords.words('english')
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
    text = re.sub("[^a-zA-Z]", " ", text)

    #doc = ' '.join(re.findall(r"[\w']+|[.,!?;/-]", text))
    #print doc
    doc = word_tokenize(text)
    # print doc
    #doc = keras.preprocessing.text.Tokenizer(num_words=None,lower=True, split=" ").fit_on_texts(doc)
    #print doc
    #doc = [word for word in text if word.isalpha()]

    #doc = [word for word in doc if word not in stop_words]
    #print doc
    return doc
def get_corpus():
    stemmer = PorterStemmer()
    data,target,filename = get_data()
    corpus_tmp = []
    for i in data:
        tmp = []
        corpus_tmp.append(preprocess(i))
    corpus_tt = []
    for j in corpus_tmp:

        corpus_tt.append([stemmer.stem(plural) for plural in j])


    #corpus_train_tmp = data
    #filter empty docs
    corpus_train, data, target, filenames = filter_docs(corpus_tt,data,target, filename, lambda doc: (len(doc) != 0))
    print corpus_train
    # stemming
    # corpus_xxx = []
    # for i in range(len(corpus_train)):
    #     # print corpus_train[i]
    #     corpus_xxx.append([stemmer.stem(plural) for plural in corpus_train[i]])

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

    corpus = []
    for i in range(len(corpus_train)):
        tmp = ' '.join([str(item) for item in corpus_train[i]])
        #tmp = str(' '.join(corpus_train[i]))
        corpus.append(tmp)
    #     print corpus
    #print
    # test = []
    # for i in corpus:
    #     test.append(' '.join([word for word in i.split()]))

    # print test
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)

    word_index = tokenizer.word_index
    key_list = []
    for i, key in word_index.items():

        key_list.append(key)
    max_key = max(key_list)
    min_key = min(key_list)
    print max_key, min_key
    print('Found %s unique tokens.' % len(word_index))

    MAX_SEQUENCE_LENGTH = max_key
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    test = []
    for i in range(len(data)):
        #data[i] = [float(x) / float(max_key) for x in data[i]]
        test.append([float(x) / float(max_key) for x in data[i]])


    return test,target,filenames,word_index,max_key

def lstm():
    data_tmp,  targets, filenames, word_index,max_key = preprocess_embedding()
    data= np.reshape(data_tmp,(1,len(data_tmp),max_key))
    print data.shape
    keras.callbacks.TensorBoard(log_dir='./Graph_lstm_no_embedding', histogram_freq=0,
                                write_graph=True, write_images=True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph_lstm_no_embedding', histogram_freq=10,
                                             # embeddings_layer_names='layer_embedding',
                                             # embeddings_freq=100,
                                             write_graph=True, write_images=True)
    model = Sequential()
    timesteps =len(data_tmp)
    print timesteps
    data_dim = max_key
    model.add(LSTM(int(data_dim/2), return_sequences=True,
                   input_shape=(timesteps, data_dim)))
    model.add(LSTM(int(data_dim/4),return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(int(data_dim/8),return_sequences=True))
    model.add(LSTM(int(data_dim / 16),return_sequences=True))
    model.add(LSTM(int(data_dim / 32),return_sequences=True, name='layer_5'))
    model.add(LSTM(int(data_dim / 32)))
    model.add(RepeatVector(len(data_tmp)))
    model.add(LSTM(int(data_dim / 16), return_sequences=True))
    model.add(LSTM(int(data_dim / 8), return_sequences=True))
    model.add(LSTM(int(data_dim / 4), return_sequences=True))
    model.add(LSTM(int(data_dim / 2), return_sequences=True))
    model.add(LSTM(data_dim, return_sequences=True, activation='linear'))
    print model.summary()

    model.compile(loss='cosine_proximity',
                  optimizer='sgd')

    model.fit(data, data, epochs=20,callbacks=[tbCallBack])

    hidden_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer('layer_5').output)
    print hidden_layer_model.predict(data).shape
    print hidden_layer_model.predict(data)
   #  EMBEDDING_DIM = 300
   #  MAX_SEQUENCE_LENGTH = 50
   #  keras.callbacks.TensorBoard(log_dir='./Graph_lstm_embedding', histogram_freq=0,
   #                              write_graph=True, write_images=True)
   #  tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph_lstm_embedding', histogram_freq=10,
   #                                           embeddings_layer_names='layer_embedding',
   #                                           embeddings_freq=100,
   #                                           write_graph=True, write_images=True)
   #  embedding_layer = Embedding(len(word_index) + 1,
   #                              EMBEDDING_DIM,
   #                              weights=[embedding_matrix],
   #                              input_length=MAX_SEQUENCE_LENGTH,
   #                              trainable= False,
   #                              name='layer_embedding') #mask_zero=True,
   #
   #
   #  sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
   #  embedded_sequences = embedding_layer(sequence_input)
   #
   #  x1 = LSTM(150, return_sequences=True,name='lstm_1')(embedded_sequences)
   #
   #  #x2 = LSTM(75, return_sequences=True,name='lstm_2')(x1)
   #  encoded = LSTM(30,name='lstm_3')(x1)
   #  x3 = RepeatVector(MAX_SEQUENCE_LENGTH,name='layer_repeat')(encoded)
   # # x4 = LSTM(75, return_sequences=True,name='lstm_4')(x3)
   #  x5 = LSTM(150, return_sequences=True,name='lstm_5')(x3)
   #  decoded = LSTM(300, return_sequences=True,activation='linear',name='lstm_6')(x5)
   #
   #  sequence_autoencoder = Model(sequence_input, decoded)
   #  #print sequence_autoencoder.get_layer('lstm_6').output
   #  encoder = Model(sequence_input, encoded)
   #  sequence_autoencoder.compile(loss='cosine_proximity',
   #                optimizer='sgd')#, metrics=['acc'])
   #  embedding_layer = Model(inputs=sequence_autoencoder.input,
   #                                   outputs=sequence_autoencoder.get_layer('layer_embedding').output)
   #
   #
   #  sequence_autoencoder.fit(data, embedding_layer.predict(data), epochs=1, callbacks=[tbCallBack])

    data_final = np.reshape(hidden_layer_model.predict(data),(len(data_tmp),list(hidden_layer_model.predict(data).shape)[2]))
    csvname = 'lstm_autoencoder_representation'
    write_vec_to_csv(data_final,targets,filenames,csvname)
    model.save('lstm_autoencoder_representation.h5')

def write_vec_to_csv(doc_vector_train,targets,filenames, csvname):
    # target_name_train = []
    # for i in xrange(len(targets)):
    #     target_name_train.append(newsgroups_train.target_names[newsgroups_train.target[i]])
    # print len(target_name_train)
    # print doc_vector_train_tsne.shape
    # print len(newsgroups_train.filenames)
    output_train = np.column_stack((targets,filenames, doc_vector_train))
    output_train = np.array(output_train)

    with open('reviews_500_' + csvname + '_.csv', 'w') as f:
        fieldnames = ['target_names', 'filenames']
        for i in xrange(len(doc_vector_train[1])):
            fieldnames.append('x'+ str(i))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        #writer = csv.DictWriter(f)
        writer.writeheader()
        writer = csv.writer(f)

        writer.writerows(output_train)

def main():
    lstm()
 #preprocess_embedding()
 #preprocess_embedding()
 #embedding()


if __name__ == "__main__":
 main()
