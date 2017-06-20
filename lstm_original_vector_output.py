import keras
import numpy
import os
import re
import csv
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from collections import defaultdict
from itertools import count
from functools import partial
from collections import defaultdict
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import defaultdict
from keras.preprocessing import text
import nltk
import sys
import gensim
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

stop_words = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#reload(sys)
#sys.setdefaultencoding('utf8')
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
    for j in  range(len(path)):

        #print path[j]
        #for i in path[j]:
        for file in os.listdir(path[j]):

            with open(os.path.join(path[j], file), 'r') as f:
                 document = f.read().lower()
                 #document = re.sub(r'^https?:\/\/.*[\r\n]*\*', '', document, flags=re.MULTILINE)
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
    corpus_train_tmp = [preprocess(text) for text in data]

    corpus_train_xxx = []


    #filter empty docs
    corpus_train, data, target, filenames = filter_docs(corpus_train_tmp,data,target, filename, lambda doc: (len(doc) != 0))
    #stemming
    #print corpus_train
    # for i in range(len(corpus_train)):
    #     #print corpus_train[i]
    #     corpus_train_xxx.append([stemmer.stem(plural) for plural in corpus_train[i]])
    #dictionary
    for i in corpus_train:
        for j in i:
            corpus_train_xxx.append(j)
    test = dict(string_to_integer(corpus_train_xxx))
    max_integer = float(max(k for k,v in test.iteritems()))
    print max_integer
    #print test
    corpus_integer = []
    for i in corpus_train:
        tmp=[]
        for j in i:
            # ddd = (k for k, v in test.iteritems() if v == j)
            # tmp.append(ddd / max_integer)
            for key, value in test.iteritems():
                if j == value:
                    tmp.append(key)
                    #tmp.append(key / max_integer)
        corpus_integer.append(tmp)

    print corpus_integer
    corpus_integer_padded = keras.preprocessing.sequence.pad_sequences(corpus_integer, maxlen=50, dtype='int32',
                                               padding='pre', truncating='pre', value=0.)
    write_vec_to_csv(corpus_integer_padded, target, filenames, csvname='original')
    return corpus_integer, data, target, filenames
def string_to_integer(strings):
    string_to_number = {string: i for i, string in enumerate(set(strings), 1)}
    test = [(string_to_number[string], string) for string in strings]
    # string_to_number = defaultdict(partial(next, count(1)))
    # test = [(string_to_number[string], string) for string in strings]
    return test
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

def write_vec_to_csv(doc_vector_train,targets,filenames, csvname):
    # target_name_train = []
    # for i in xrange(len(targets)):
    #     target_name_train.append(newsgroups_train.target_names[newsgroups_train.target[i]])
    # print len(target_name_train)
    # print doc_vector_train_tsne.shape
    # print len(newsgroups_train.filenames)
    output_train = np.column_stack((targets,filenames, doc_vector_train))
    output_train = np.array(output_train)

    with open('reviews_500_embedding_lstm' + csvname + '_.csv', 'w') as f:
        fieldnames = ['target_names', 'filenames']
        for i in xrange(len(doc_vector_train[1])):
            fieldnames.append('x'+ str(i))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        #writer = csv.DictWriter(f)
        writer.writeheader()
        writer = csv.writer(f)

        writer.writerows(output_train)

# def lstm_autoencoder(corpus):
#     #padding document
#     max_review_length = 50
#     padded_corpus= sequence.pad_sequences(corpus, maxlen=max_review_length)
#     # create the model
#
#     inputs = Input(shape=(timesteps, input_dim))
#     encoded = LSTM(latent_dim)(inputs)
#
#     decoded = RepeatVector(timesteps)(encoded)
#     decoded = LSTM(input_dim, return_sequences=True)(decoded)
#
#     sequence_autoencoder = Model(inputs, decoded)
#     encoder = Model(inputs, encoded)
#
#
#     embedding_vecor_length = 32
#     model = Sequential()
#     model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
#     model.add(LSTM(100))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     print(model.summary())
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)



def main():
 get_corpus()



if __name__ == "__main__":
 main()