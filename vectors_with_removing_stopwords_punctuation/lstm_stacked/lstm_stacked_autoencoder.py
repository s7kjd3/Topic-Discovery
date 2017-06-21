
#!/usr/bin/env python
import keras
import os
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import re
import csv
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.layers import Input, LSTM, RepeatVector,TimeDistributed,Dense,Dropout,Embedding,Flatten
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
from keras import optimizers
reload(sys)
sys.setdefaultencoding('utf8')
stop_words = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def get_data():
    targets = ['cameras','laptops','mobilephone','tablets','TVs','video_surveillance']
    path =['/home/flippped/thesis_project/Reviews/Amazon_300_500/camera_no',
           '/home/flippped/thesis_project/Reviews/Amazon_300_500/laptop_no',
           '/home/flippped/thesis_project/Reviews/Amazon_300_500/mobilephone_no',
           '/home/flippped/thesis_project/Reviews/Amazon_300_500/tablets_no',
           '/home/flippped/thesis_project/Reviews/Amazon_300_500/TVs_no',
           '/home/flippped/thesis_project/Reviews/Amazon_300_500/video_surveillance_no']
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
    #doc = ' '.join(re.findall(r"[\w']+|[,]", text))
    #print doc
    doc = word_tokenize(text)
    #doc = keras.preprocessing.text.Tokenizer(num_words=None,lower=True, split=" ").fit_on_texts(doc)
    #print doc
    doc = [word for word in doc if word.isalpha()]
    doc = [word for word in doc if word not in stop_words]

    return doc
def get_corpus():
    #stemmer = PorterStemmer()
    data,target,filename = get_data()
    corpus_train_tmp = [preprocess(text) for text in data]
    #corpus_train_tmp = data
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
    corpus, texts,labels, filenames = zip(*((x, y,z,w) for x, y,z,w in zip(corpus, texts,labels, filenames) if len(x) > 0))

    corpus = list(corpus)
    labels = list(labels)
    filenames = list(filenames)
    texts = list(texts)
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
def preprocess_embedding():
    corpus_train, target, filenames = get_corpus()
    tmp_length = []
    for i in corpus_train:
        tmp_length.append(len(i))
    max_tmp_length = max(tmp_length)

    for i in range(len(corpus_train)):
        corpus_train[i] = ' '.join(corpus_train[i])


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


    return data,target,filenames,embedding_matrix, word_index,MAX_SEQUENCE_LENGTH

def lstm():
    data,  targets, filenames, embedding_matrix, word_index, MAX_SEQUENCE_LENGTH = preprocess_embedding()
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

    # [batchsize, timesteps, output_dimension ]

    x2 = LSTM(300, return_sequences=True,name='lstm_2')(embedded_sequences)
    encoded = LSTM(300,name='lstm_3')(x2)

    x3 = RepeatVector(MAX_SEQUENCE_LENGTH,name='layer_repeat')(encoded)
   # x4 = LSTM(75, return_sequences=True,name='lstm_4')(x3)
   #  x5 = LSTM(300, return_sequences=True,name='lstm_5')(x3)
    decoded = LSTM(300, return_sequences=True,activation='linear',name='lstm_6')(x3)

    sequence_autoencoder = Model(sequence_input, decoded)
    #print sequence_autoencoder.get_layer('lstm_6').output
    encoder = Model(sequence_input, encoded) # two functions that you learn

    RMSprop = optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    sgd = optimizers.sgd(lr=0.001,momentum=0.9)

    sequence_autoencoder.compile(loss='cosine_proximity',
                  optimizer=sgd)#, metrics=['acc'])
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
    csvname = 'Amazon_lstm_autoencoder'
    write_vec_to_csv(encoded_data,targets,filenames,csvname)

    # model.compile(optimizer='rmsprop',
    #               loss='mse', )
    # model.fit(a, model.get_layer('lstm_6').output, nb_epoch=5)
    # model.save('test.h5')


# def enceonding(input):
#     # this is what the learning learns
#     encoded = weights * inputs
#
# def decoding(encoded):
#     reconscructed_iput = weights * encoded
#     return reconstructed_input


def write_vec_to_csv(doc_vector_train,targets,filenames, csvname):
    # target_name_train = []
    # for i in xrange(len(targets)):
    #     target_name_train.append(newsgroups_train.target_names[newsgroups_train.target[i]])
    # print len(target_name_train)
    # print doc_vector_train_tsne.shape
    # print len(newsgroups_train.filenames)
    output_train = np.column_stack((targets,filenames, doc_vector_train))
    output_train = np.array(output_train)

    with open('reviews_300_500_' + csvname + '_.csv', 'w') as f:
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
 #embedding()


if __name__ == "__main__":
 main()
