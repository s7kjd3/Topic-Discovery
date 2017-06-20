#!/usr/bin/env python
import keras
import os
import scipy
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import re
import csv
from collections import Counter
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
from sklearn.cluster import KMeans,AgglomerativeClustering,hierarchical,DBSCAN,SpectralClustering,SpectralCoclustering, spectral_clustering,AffinityPropagation
# from spherecluster import SphericalKMeans
stop_words = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
reload(sys)
sys.setdefaultencoding('utf8')
def get_data():
    targets = ['cameras','laptops','mobilephone','tablets','TVs','video_surveillance']
    path =['/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_500/cameras',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_500/laptops',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_500/mobilephone',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_500/tablets',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_500/TVs',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews_500/video_surveillance']
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
    doc = word_tokenize(text)
    doc = [word for word in doc if word.isalpha()]
    doc = [word for word in doc if word not in stop_words]

    return doc

def get_corpus():
    #stemmer = PorterStemmer()
    data,target,filename = get_data()
    corpus_train_tmp = [preprocess(text) for text in data]
    print corpus_train_tmp
    print len(corpus_train_tmp)
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

    # if texts is not None:
    #     texts = [text for (text, doc) in zip(texts, corpus)
    #              if condition_on_doc(doc)]
    #
    # labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
    # corpus = [doc for doc in corpus if condition_on_doc(doc)]
    # filenames = [filename for (filename,doc) in zip(filenames, corpus) if condition_on_doc(doc)]
    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels, filenames)
def string_to_integer(strings):
    string_to_number = {string: i for i, string in enumerate(set(strings), 1)}
    test = [(string_to_number[string], string) for string in strings]
    # string_to_number = defaultdict(partial(next, count(1)))
    # test = [(string_to_number[string], string) for string in strings]
    return test

def clustering():
    corpus_train, target, filenames = get_corpus()

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/home/flippped/Windows/Linux_Project/xiangmu/baseline/GoogleNews-vectors-negative300.bin.gz', binary=True)
    word2vec_model.init_sims(replace=True)
    # create one matrix for documents words
    corpus_embeddings = []
    corpus_words = []
    for i in range(len(corpus_train)):
        tmp_vec = []
        tmp_word = []
        for word in corpus_train[i]:
            try:
                embedding_vector = word2vec_model[str(word)]
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    tmp_word.append(str(word))
                    tmp_vec.append(embedding_vector)
            except:
                continue

        corpus_embeddings.append(tmp_vec)
        corpus_words.append(tmp_word)


    # for i in corpus_embeddings:
    #     pre_computed = scipy.spatial.distance.pdist(i,met)
    #y = input("Please choose clustering method: 1: Kmeans 2: spherical k-means 3: Agglomerative Clustering 4: Spectral Clustering ")
    y = 1
    corpus_cluster_embeddings = []
    corpus_cluster_words = []
    for a in range(len(corpus_embeddings)):


        if y == 1:
            model = KMeans(n_clusters=10, max_iter=10000000).fit(corpus_embeddings[a])
        # if y == 2:
        #     model = SphericalKMeans(n_clusters=10, max_iter=10000000).fit(corpus_embeddings[a])
        if y == 3:
            x = Input("Please choose matrix method: 1: Cosine 2: Euclidean ")
            if x ==1:
                model = AgglomerativeClustering(n_clusters=10, affinity='cosine', linkage='complete').fit(corpus_embeddings[a])
            if x ==2:
                model = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='complete').fit(corpus_embeddings[a])
        if y == 4:
            model = SpectralClustering(n_clusters=10).fit(corpus_embeddings[corpus_embeddings[a]])
        if y == 5:
            model = AffinityPropagation(max_iter=10000000).fit(corpus_embeddings[corpus_embeddings[a]])
        # for i in model.cluster_centers_:
        #     print i
        tmp = []
        for i in model.cluster_centers_:

            for j in range(len(corpus_embeddings[a])):
                if str(corpus_embeddings[a][j]) == str(i):
                    tmp.append(corpus_words[a][j])
                    break
                else:
                    continue
        corpus_cluster_words.append(tmp)
        corpus_cluster_embeddings.append(model.cluster_centers_)


    print len(corpus_cluster_words), len(corpus_cluster_embeddings)
    for i in range(len(corpus_cluster_words)):
        corpus_cluster_words[i] = ' '.join(corpus_cluster_words[i])
    for i in corpus_cluster_words:
        print i
    return corpus_cluster_words, target, filenames, word2vec_model
    #return corpus_cluster_embeddings, corpus_cluster_words, target, filenames

def lstm():
    data,  targets, filenames, word2vec_model = clustering()
    tmp_len = []
    tmp_word = []
    for i in data:
        tmp_len.append(len(i))
        for j in i:
            tmp_word.append(j)
    max_sen_len = max(tmp_len)

    EMBEDDING_DIM = 300

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    MAX_SEQUENCE_LENGTH = max_sen_len
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


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


    keras.callbacks.TensorBoard(log_dir='./Graph_lstm_embedding', histogram_freq=0,
                                write_graph=True, write_images=True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph_lstm_embedding', histogram_freq=10,
                                             embeddings_layer_names='layer_embedding',
                                             embeddings_freq=100,
                                             write_graph=True, write_images=True)

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=max_sen_len,
                                trainable= False,
                                name='layer_embedding', mask_zero=True,)

    # STep 1 Training
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    #x1 = LSTM(150, return_sequences=True,name='lstm_1')(embedded_sequences) # [ batchsize, timesteps, input_dimension ]

    # [batchsize, timesteps, output_dimension ]

    #x2 = LSTM(75, return_sequences=True,name='lstm_2')(x1)
    encoded = LSTM(300,name='lstm_3')(embedded_sequences)

    x3 = RepeatVector(MAX_SEQUENCE_LENGTH,name='layer_repeat')(encoded)
   # x4 = LSTM(75, return_sequences=True,name='lstm_4')(x3)
    #x5 = LSTM(150, return_sequences=True,name='lstm_5')(x3)
    decoded = LSTM(300, return_sequences=True,activation='softmax',name='lstm_6')(x3)

    sequence_autoencoder = Model(sequence_input, decoded)
    #print sequence_autoencoder.get_layer('lstm_6').output
    encoder = Model(sequence_input, encoded) # two functions that you learn

    sequence_autoencoder.compile(loss='cosine_proximity',
                  optimizer='sgd')#, metrics=['acc'])
    embedding_layer = Model(inputs=sequence_autoencoder.input,
                                     outputs=sequence_autoencoder.get_layer('layer_embedding').output)


    sequence_autoencoder.fit(data, embedding_layer.predict(data), epochs=10, callbacks=[tbCallBack])
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
    #
    #
    #
    csvname = 'lstm_autoencoder_clustering_representation'
    write_vec_to_csv(encoded_data,targets,filenames,csvname)

    # model.compile(optimizer='rmsprop',
    #               loss='mse', )
    # model.fit(a, model.get_layer('lstm_6').output, nb_epoch=5)
    # model.save('test.h5')
    print encoder.predict(data)


def write_vec_to_csv(doc_vector_train,targets,filenames, csvname):
    # target_name_train = []
    # for i in xrange(len(targets)):
    #     target_name_train.append(newsgroups_train.target_names[newsgroups_train.target[i]])
    # print len(target_name_train)
    # print doc_vector_train_tsne.shape
    # print len(newsgroups_train.filenames)
    output_train = np.column_stack((targets,filenames, doc_vector_train))
    output_train = np.array(output_train)

    with open('reviews_500_final_' + csvname + '_.csv', 'w') as f:
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
 # clustering()
 # get_corpus()
 #preprocess_embedding()
 #embedding()


if __name__ == "__main__":
 main()
