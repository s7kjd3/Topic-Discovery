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
nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
reload(sys)
sys.setdefaultencoding('utf8')
def get_data():
    targets = ['cameras','laptops','mobilephone','tablets','TVs','video_surveillance']
    path =['/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_test/cameras',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_test/laptops',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_test/mobilephone',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_test/tablets',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_test/TVs',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_test/video_surveillance']
    data = []
    target=[]
    filename = []
    data_to_sentences = []
    for j in  xrange(len(path)):

        print path[j]
        #for i in path[j]:
        for file in os.listdir(path[j]):

            with open(os.path.join(path[j], file), 'r') as f:
                 document = f.read().lower()
                 target.append(targets[j])
                 filename.append(file)
                 data.append(document)
    for i in range(len(data)):
        data_to_sentences.append((tokenizer.tokenize(data[i])))

    print len(data_to_sentences)
    return data_to_sentences,target,filename
def preprocess(text):

    # doc = []
    # for i in text:
    #     doc.append(' '.join(re.findall(r"[\w']+|[,]", i)))

    doc = text
    # doc = word_tokenize(doc)
    #doc = keras.preprocessing.text.Tokenizer(num_words=None,lower=True, split=" ").fit_on_texts(doc)
    #print doc
    #doc = [word for word in doc if word.isalpha()]
    doc = [word for word in doc if word not in stop_words]
    #print doc
    return doc
def get_corpus():
    #stemmer = PorterStemmer()
    data,target,filename = get_data()
    corpus_train_tmp = [preprocess(text) for text in data]
    corpus_sentences, data, target, filenames = filter_docs(corpus_train_tmp,data,target, filename, lambda doc: (len(doc) != 0))
    # for i in corpus_train:
    #     print i
    return corpus_sentences, target, filenames
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

def preprocess_embedding():
    corpus_train, target, filenames = get_corpus()
    sentences = []
    sent_counter = []
    word_counter = []

    # transform document into sentences
    for i in range(len(corpus_train)):

        for j in corpus_train[i]:
            j_tokenized = word_tokenize(j)
            sentences.append(j)
            word_counter.append(len(j_tokenized))
        # count sentences number in one document
        sent_counter.append(len(corpus_train[i]))

    max_sentences_len = max(sent_counter)
    max_words_len = max(word_counter)


    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    MAX_SEQUENCE_LENGTH = max_words_len
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    mask_zero = np.zeros(MAX_SEQUENCE_LENGTH)
    data_new = []
    tmp_mask = -(sent_counter[0])
    for i in range(len(sent_counter)):
        if i < len(sent_counter) - 1:
            tmp_mask += sent_counter[i]
        for iii in range(max_sentences_len - sent_counter[i]):
            data_new.append(mask_zero)

        for ii in range(sent_counter[i]):
            data_new.append(data[tmp_mask + ii])




    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/home/flippped/Windows/Linux_Project/xiangmu/baseline/GoogleNews-vectors-negative300.bin.gz', binary=True)
    word2vec_model.init_sims(replace=True)

    # create one matrix for documents words
    EMBEDDING_DIM = 300
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
            try:
                embedding_vector = word2vec_model[str(word)]
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

            except:
                continue


    return data,target,filenames,embedding_matrix, word_index, max_words_len, max_sentences_len

def lstm_words():
    data,  targets, filenames, embedding_matrix, word_index, max_words_len, max_sentences_len = preprocess_embedding()




    max_sentences_no = max_sentences_len
    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = max_words_len

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable= False,
                                name='layer_embedding',mask_zero=True)





    # Encode each timestep, encode sentence into one vector
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedded_sequences = embedding_layer(sequence_input)
    print embedded_sequences

    lstm_sentence = LSTM(300,return_sequences=True)(embedded_sequences)
    print lstm_sentence

    lstm_sentence = LSTM(300)(lstm_sentence)
    print lstm_sentence

    lstm_sentence = Reshape((max_sentences_no,300))(lstm_sentence)
    print lstm_sentence

    ######################################
    word_encoded_model = Model(sequence_input, lstm_sentence)
    # print word_encoded_model.summary()
    #
    #
    # #Wrap vectors every 8 steps, encode document into one vector
    # sentence_input = Input(shape=(max_sentences_no, 300))
    # seq_encoded = TimeDistributed(word_encoded_model)(sentence_input)
    # print seq_encoded
    #
    # seq_encoded = Dropout(0.2)(seq_encoded)


    seq_encoded = LSTM(300)(lstm_sentence)
    print seq_encoded

    # sent_encoded_model = Model(sentence_input,seq_encoded)
    # print sent_encoded_model.summary()
    ###########################################

    seq_repeatvector = RepeatVector(max_sentences_no)(seq_encoded)
    print seq_repeatvector



    seq_decoded = LSTM(300, return_sequences=True)(seq_repeatvector)
    print (seq_decoded)

    # seq_decoded = Dense(300)(seq_decoded)
    # print (seq_decoded)

    seq_decoded_model = Model(sequence_input, seq_decoded)
    # seq_decoded_model = Model(sentence_input,seq_decoded)
    #
    # print seq_decoded_model.summary()


    seq_decoded = Reshape(())
    seq_decoded = tf.reshape(seq_decoded, [-1, 300])
    seq_decoded = Lambda(new_shape)(seq_decoded)
    print seq_decoded

    # seq_decoded = tf.unstack(seq_decoded,axis=1)
    # print seq_decoded
    #
    #
    # seq_decoded = merge(seq_decoded)
    # # seq_decoded = tf.concat(seq_decoded,axis=0)
    # print seq_decoded

    # seq_decoded = Dense(300)(seq_decoded)
    # print (seq_decoded)
    seq_decoded_model_2 = Model(sequence_input,seq_decoded)

    word_repeatvector = RepeatVector(MAX_SEQUENCE_LENGTH)(seq_decoded)
    # word_repeatvector = RepeatVector(MAX_SEQUENCE_LENGTH, name='layer_repeat_2')(word_decoded)
    print word_repeatvector
    word_decoded = LSTM(300,return_sequences=True)(word_repeatvector)
    print word_decoded
    # word_repeatvector = RepeatVector(MAX_SEQUENCE_LENGTH,name='layer_repeat_2')(word_decoded_model)
    # word_decoded = LSTM(300, return_sequences=True)(word_repeatvector)


    words_decoded_model = Model(sequence_input, word_decoded)
    print words_decoded_model.summary()
    # seq_decoded = Reshape((1, 300),input_shape=())(seq_decoded)
    # seq_decoded = LSTM(300, return_sequences=True, activation='softmax', name='lstm_6')(seq_decoded)
    # sent_decoded_model = Model(sequence_input,seq_decoded)

    words_decoded_model.compile(loss='cosine_proximity',
                                 optimizer='sgd')

    embedding_layer = Model(inputs=words_decoded_model.input,
                            outputs=words_decoded_model.get_layer('layer_embedding').output)

    words_decoded_model.fit(data, embedding_layer.predict(data), epochs=1)



def new_shape(x):
    # with tf.Session():
    #     a = x.eval()
    # aa = np.reshape(a,(-1,8,300))

    return x





def main():
    lstm_words()


if __name__ == "__main__":
    main()
