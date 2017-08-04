import keras
import os
from sklearn.manifold import TSNE
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import re
import csv
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.layers import Input, LSTM,merge, RepeatVector,TimeDistributed,Dense,Dropout,Embedding,Masking,Reshape,Lambda,SimpleRNN
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
from collections import deque
from keras import optimizers
import tensorflow as tf
from keras import backend as K
import codecs
from nltk.stem.porter import *

reload(sys)
sys.setdefaultencoding('utf8')
stop_words = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def get_data():
    title_file = (r'/home/flippped/thesis_project/stackoverflow/title_StackOverflow.txt')
    label_file = (r'/home/flippped/thesis_project/stackoverflow/label_StackOverflow.txt')
    # path = [r'/home/flippped/thesis_project/stackoverflow']
    # for j in  xrange(len(path)):
    #
    #     print path[j]
    #     #for i in path[j]:
    #     for file in os.listdir(path[j]):
    #
    #         with open(os.path.join(path[j], file), 'r') as f:
    #             data = f.readlines()

    with open(title_file) as f:
        data = f.readlines()
    with open(label_file)as f:
        label_tmp = f.readlines()
    label = []
    for i in label_tmp:
        label.append(int(i))
    data_list = []
    for i in data:
        data_list.append(i)
    print data_list[1]
    sum = 0
    count = []
    for i in range(len(data_list)):
        sum += len(data_list[i])
        count.append(len(data_list[i]))
    print ("total size: " + str(len(data_list)))
    print ("average size: " + str(int(sum / len(data_list))))
    print ("longest document" + str(max(count)))
    print ("shortest document" + str(min(count)))
    for i in data_list:
        print i
    return data_list,label
def preprocess(text):
    stemmer = PorterStemmer()
    text = text.lower()
    doc = ' '.join(re.findall(r"[\w']+|[,.?!]", text))
    #print doc
    doc = word_tokenize(doc)
    # doc = keras.preprocessing.text.Tokenizer(num_words=None,lower=True, split=" ").fit_on_texts(doc)
    #print doc

    # doc = [word for word in text if word.isalpha()]
    # print doc
    doc = [word for word in doc if word not in stop_words]
    # doc = [stemmer.stem(plural) for plural in doc]

    return doc
def get_corpus():
    #stemmer = PorterStemmer()
    data, labels = get_data()
    corpus_train_tmp = [preprocess(text) for text in data]

    #corpus_train_tmp = data
    #filter empty docs
    corpus_train, data, labels = filter_docs(corpus_train_tmp,data,labels)
    corpus_train = list(corpus_train)
    data = list(data)
    labels = list(labels)


    return corpus_train, data, labels

def filter_docs(corpus,  text, labels):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """

    number_of_docs = len(corpus)

    corpus, text,labels = zip(*((x, y,z) for x, y,z in zip(corpus, text,labels) if len(x) > 0))
    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, text, labels)


def preprocess_embedding():
    corpus_train, data_original, labels = get_corpus()
    tmp_length = []
    for i in corpus_train:
        tmp_length.append(len(i))
    max_tmp_length = max(tmp_length)

    for i in range(len(corpus_train)):

        corpus_train[i] = ' '.join(corpus_train[i])


    print corpus_train
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus_train)
    sequences = tokenizer.texts_to_sequences(corpus_train)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    MAX_SEQUENCE_LENGTH = max_tmp_length
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

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


    return data,data_original, labels, embedding_matrix, word_index,MAX_SEQUENCE_LENGTH

def lstm():
    data,  data_original, labels, embedding_matrix, word_index, MAX_SEQUENCE_LENGTH = preprocess_embedding()
    EMBEDDING_DIM = 300
    keras.callbacks.TensorBoard(log_dir='./Graph_lstm_embedding', histogram_freq=0,
                                write_graph=True, write_images=True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph_lstm_embedding', histogram_freq=10,
                                             embeddings_layer_names='layer_embedding',
                                             embeddings_freq=100,
                                             write_graph=True, write_images=True)
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable= False,
                                name='layer_embedding', mask_zero=True,)

    # STep 1 Training
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    #x1 = LSTM(150, return_sequences=True,name='lstm_1')(embedded_sequences) # [ batchsize, timesteps, input_dimension ]
    dropout = Dropout(0.2)(embedded_sequences)
    # [batchsize, timesteps, output_dimension ]

    #x2 = LSTM(75, return_sequences=True,name='lstm_2')(x1)
    encoded = LSTM(300,name='lstm_3')(dropout)

    x3 = RepeatVector(MAX_SEQUENCE_LENGTH,name='layer_repeat')(encoded)
   # x4 = LSTM(75, return_sequences=True,name='lstm_4')(x3)
    #x5 = LSTM(150, return_sequences=True,name='lstm_5')(x3)
    decoded = LSTM(300, return_sequences=True,activation='softmax',name='lstm_6')(x3)
    print decoded
    sequence_autoencoder = Model(sequence_input, decoded)
    #print sequence_autoencoder.get_layer('lstm_6').output
    encoder = Model(sequence_input, encoded) # two functions that you learn
    adam = optimizers.Adam(lr=0.0005)
    RMSprop = optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    sequence_autoencoder.compile(loss='cosine_proximity',
                  optimizer=adam)#, metrics=['acc'])
    embedding_layer = Model(inputs=sequence_autoencoder.input,
                                     outputs=sequence_autoencoder.get_layer('layer_embedding').output)


    sequence_autoencoder.fit(data, embedding_layer.predict(data), epochs=50, callbacks=[tbCallBack])
    # Training is done

    # define the encoding function using the trained encoded weights

    #encoder = Model(sequence_input, encoded) # a function that you learn
    # print '**************************************************'
    encoded_data = encoder.predict(data)
    # print encoded_data
    # print '**************************************************'
    # print encoded_data.shape
    #
    #
    csvname = 'LSTM_autoencoder'
    write_vec_to_csv(encoded_data,labels,csvname)

    # model.compile(optimizer='rmsprop',
    #               loss='mse', )
    # model.fit(a, model.get_layer('lstm_6').output, nb_epoch=5)
    # model.save('test.h5')



def write_vec_to_csv(doc_vector_train,labels, csvname):
    # target_name_train = []
    # for i in xrange(len(targets)):
    #     target_name_train.append(newsgroups_train.target_names[newsgroups_train.target[i]])
    # print len(target_name_train)
    # print doc_vector_train_tsne.shape
    # print len(newsgroups_train.filenames)
    output_train = np.column_stack((labels, doc_vector_train))
    output_train = np.array(output_train)

    with open('stackoverflow_0.2_' + csvname + '_.csv', 'w') as f:
        fieldnames = ['content']
        for i in xrange(len(doc_vector_train[1])):
            fieldnames.append('x'+ str(i))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        #writer = csv.DictWriter(f)
        writer.writeheader()
        writer = csv.writer(f)

        writer.writerows(output_train)
def main():
    # lstm()

    get_data()

if __name__ == "__main__":
    main()
