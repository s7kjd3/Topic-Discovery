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
from word2vec import *
import pandas as pd

def get_vector():
    os.chdir('/home/flippped/Windows/Linux_Project/xiangmu/baseline/')

    files = ['reviews_500_doc2vec_.csv',
            'reviews_500_LSI_.csv',
            'reviews_500_lstm_autoencoder_representation_.csv',
             'reviews_500_word2vec_google_vectors_averaging_.csv',
            'reviews_500_word2vec_google_vectors_tfidf_.csv']
    for file in files:
        with open(file, 'rU') as csvf:
            data = pd.read_csv(file, skiprows=[0], header=None)
            # data = csv.reader(csvf)
            dl = data.values.tolist()
            labels = [x[0] for x in dl]
            filenames = [x[1] for x in dl]
            a = np.array([x[2:] for x in dl])
            a_a = vector_dimension_reduction(a)
            csvname = file
            write_vec_to_csv(a_a,labels,filenames,csvname)


def vector_dimension_reduction(doc_vectors_train):
    tsne = TSNE(n_components=2, init='pca', random_state=None, method='barnes_hut', n_iter=1000)

    doc_vectors_train = np.array(doc_vectors_train)
    #print doc_vectors_train
    doc_vector_train_tsne = tsne.fit_transform(doc_vectors_train)
    #print new_doc_vec_tsne

    return doc_vector_train_tsne

#output vectors of which dimension is reduced into csv
def write_vec_to_csv(doc_vector_train,targets,filenames, csvname):
    # target_name_train = []
    # for i in xrange(len(targets)):
    #     target_name_train.append(newsgroups_train.target_names[newsgroups_train.target[i]])
    # print len(target_name_train)
    # print doc_vector_train_tsne.shape
    # print len(newsgroups_train.filenames)
    output_train = np.column_stack((targets,filenames, doc_vector_train))
    output_train = np.array(output_train)

    with open('Reduced_' + csvname + '_.csv', 'w') as f:
        fieldnames = ['target_names', 'filenames']
        for i in xrange(len(doc_vector_train[1])):
            fieldnames.append('x'+ str(i))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer = csv.writer(f)

        writer.writerows(output_train)


def main():
 get_vector()



if __name__ == "__main__":
 main()