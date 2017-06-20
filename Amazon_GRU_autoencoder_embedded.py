#!/usr/bin/env python
import keras
import os
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import re
import csv
from keras import backend as K
from keras.layers import Input, LSTM, RepeatVector,TimeDistributed,Dense,Dropout,Embedding,GRU
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
from Get_Amazon_reviews_vectors import write_vec_to_csv
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
    for i in corpus_train:
        print i
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


    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/home/flippped/Windows/Linux_Project/xiangmu/baseline/GoogleNews-vectors-negative300.bin.gz', binary=True)
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

def integer_vectors():
    os.chdir(r'/home/flippped/Windows/Linux_Project/xiangmu/baseline/')
    filename = 'reviews_50_integer_2_original_.csv'
    with open(filename, 'rU') as csvf:
        data = pd.read_csv(filename, skiprows=[0], header=None)
        # data = csv.reader(csvf)
        dl = data.values.tolist()

    return dl
def GRU():
    data,  targets, filenames, embedding_matrix, word_index = preprocess_embedding()
    #print embedding_matrix.shape
    # dl = integer_vectors()
    # labels = [x[0] for x in dl]
    # filenames = [x[1] for x in dl]
    # a = np.array([x[2:] for x in dl])
    # tmp_1 = []
    # for i in a:
    #     for j in i:
    #         tmp_1.append(j)
    # max_integer = max(tmp_1)
    # a_reshaped = a.reshape(len(a), 50, 1)
    # #data_reshape = data.reshape(len(data), 50, 1)
    # # print a_reshape.shape

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

    encoded = GRU(150, return_sequences=True,name='lstm_1')(embedded_sequences)
    encoded = GRU(75, return_sequences=True,name='lstm_2')(encoded)
    encoded = GRU(10,return_sequences=True,name='lstm_3')(encoded)
    encoded = GRU(1, return_sequences=True, name='lstm_4')(encoded)
    decoded = RepeatVector(MAX_SEQUENCE_LENGTH,name='layer_repeat')(encoded)
    encoded = GRU(10,return_sequences=True,name='lstm_5')(encoded)
    decoded = GRU(75, return_sequences=True,name='lstm_6')(decoded)
    decoded = GRU(150, return_sequences=True,name='lstm_7')(decoded)
    decoded = GRU(300, return_sequences=True,activation='linear',name='lstm_8')(decoded)

    sequence_autoencoder = Model(sequence_input, decoded)
    #print sequence_autoencoder.get_layer('lstm_6').output
    encoder = Model(sequence_input, encoded)
    sequence_autoencoder.compile(loss='cosine_proximity',
                  optimizer='sgd')#, metrics=['acc'])
    embedding_layer_model = Model(inputs=sequence_autoencoder.input,
                                     outputs=sequence_autoencoder.get_layer('layer_embedding').output)

    print embedding_layer_model.predict(data).shape

    # for i in intermediate_layer_model.predict(data):
    #     print i


    #print sequence_autoencoder.get_layer('layer_embedding').output
    #sequence_autoencoder.fit(a, sequence_autoencoder.get_layer('layer_embedding').output,epochs=2)
    sequence_autoencoder.fit(data, embedding_layer_model.predict(data), epochs=1)
    sequence_autoencoder.save_weights('embedded_weights.h5')


    csvname = 'GRU_autoencoder_weight'



    print "******************************************************"
    print sequence_autoencoder.layers[3].get_weights()[0].shape
    print "******************************************************"
    print sequence_autoencoder.layers[3].get_weights()[0][1]
    print "******************************************************"

    #write_vec_to_csv(sequence_autoencoder.layers[3].get_weights()[0], targets, filenames, csvname)
    hidden_layer_model = Model(inputs=sequence_autoencoder.input,
                               outputs=sequence_autoencoder.get_layer('layer_4').output)

    print hidden_layer_model.predict(data).shape
    print hidden_layer_model.predict(data)


def write_vec_to_csv(doc_vector_train, targets, filenames, csvname):
    # target_name_train = []
    # for i in xrange(len(targets)):
    #     target_name_train.append(newsgroups_train.target_names[newsgroups_train.target[i]])
    # print len(target_name_train)
    # print doc_vector_train_tsne.shape
    # print len(newsgroups_train.filenames)
    output_train = np.column_stack((targets, filenames, doc_vector_train))
    output_train = np.array(output_train)

    with open('reviews_50_' + csvname + '_.csv', 'w') as f:
        fieldnames = ['target_names', 'filenames']
        for i in xrange(len(doc_vector_train[1])):
            fieldnames.append('x' + str(i))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # writer = csv.DictWriter(f)
        writer.writeheader()
        writer = csv.writer(f)

        writer.writerows(output_train)





def main():
 GRU()
 #preprocess_embedding()
 #embedding()


if __name__ == "__main__":
 main()

