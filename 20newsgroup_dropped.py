import nltk
import sys
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
download('punkt')
download('stopwords')
stop_words = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
reload(sys)
sys.setdefaultencoding('utf8')
def get_data():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories= ['alt.atheism'])
 # 'comp.graphics',
 # 'comp.os.ms-windows.misc',
 # 'comp.sys.ibm.pc.hardware',
 # 'comp.sys.mac.hardware',
 # 'comp.windows.x',
 # 'misc.forsale',
 # 'rec.autos',
 # 'rec.motorcycles',
 # 'rec.sport.baseball',
 # 'rec.sport.hockey',
 # 'sci.crypt',
 # 'sci.electronics',
 # 'sci.med',
 # 'sci.space',
 # 'soc.religion.christian',
 # 'talk.politics.guns',
 # 'talk.politics.mideast',
 # 'talk.politics.misc',
 # 'talk.religion.misc'])
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),categories= ['alt.atheism'])
 # 'comp.graphics',
 # 'comp.os.ms-windows.misc',
 # 'comp.sys.ibm.pc.hardware',
 # 'comp.sys.mac.hardware',
 # 'comp.windows.x',
 # 'misc.forsale',
 # 'rec.autos',
 # 'rec.motorcycles',
 # 'rec.sport.baseball',
 # 'rec.sport.hockey',
 # 'sci.crypt',
 # 'sci.electronics',
 # 'sci.med',
 # 'sci.space',
 # 'soc.religion.christian',
 # 'talk.politics.guns',
 # 'talk.politics.mideast',
 # 'talk.politics.misc',
 # 'talk.religion.misc'])
    # texts_train, target_train = newsgroups_train.data, newsgroups_train.target
    # texts_test, target_test = newsgroups_test.data,newsgroups_test.target
    #return texts_train, target_train, newsgroups_train.filenames, texts_test, target_test, newsgroups_test.filenames


    return newsgroups_train, newsgroups_test

class MyTaggedDocument(object):
    def __init__(self, corpus_train, newsgroups_train, corpus_test, newsgroups_test):
        self.corpus_train = corpus_train
        self.newsgroups_train = newsgroups_train
        self.corpus_test = corpus_test
        self.newsgroups_test = newsgroups_test
        self.corpus = self.corpus_train + self.corpus_test

        self.docList = self.newsgroups_train.filenames + self.newsgroups_test.filenames
        self.target = self.newsgroups_train.target + self.newsgroups_test.target

    def __iter__(self):
        for idx, doc in enumerate(self.corpus):
            # yield TaggedDocument(doc, [self.newsgroups_train.target_names[self.target[idx]] + str[idx]])
            yield TaggedDocument(doc, [idx])



def preprocess(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    doc = word_tokenize(text)
    doc = [word for word in doc if word.isalpha()]
    doc = [word for word in doc if word not in stop_words]

    return doc


def get_corpus():
    stemmer = PorterStemmer()
    newsgroups_train, newsgroups_test = get_data()
    corpus_train_tmp = [preprocess(text) for text in newsgroups_train.data]
    corpus_test_tmp = [preprocess(text) for text in newsgroups_test.data]

    corpus_train_xxx = []
    corpus_test_xxx = []


    #filter empty docs
    corpus_train, newsgroups_train.data, newsgroups_train.target, newsgroups_train.filenames = filter_docs(corpus_train_tmp,newsgroups_train.data,newsgroups_train.target, newsgroups_train.filenames, lambda doc: (len(doc) != 0))
    corpus_test, newsgroups_test.data, newsgroups_test.target, newsgroups_test.filenames = filter_docs(corpus_test_tmp,newsgroups_test.data, newsgroups_test.target, newsgroups_test.filenames, lambda doc: (len(doc) != 0))
    # print len(newsgroups_train.data)
    # print len(corpus_train)
    # print len(newsgroups_train.target)
    # print len(newsgroups_train.filenames)

    #stemming
    #print corpus_train
    for i in range(len(corpus_train)):
        #print corpus_train[i]
        corpus_train_xxx.append([stemmer.stem(plural) for plural in corpus_train[i]])
    # for i in xrange(len(singles)):
    #     new.append([stemmer.stem(plural) for plural in singles[i]])
    for i in range(len(corpus_test)):
        corpus_test_xxx.append([stemmer.stem(plural) for plural in corpus_test[i]])
    return corpus_train_xxx, newsgroups_train, corpus_test_xxx, newsgroups_test

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


def get_vectors():
    y = input("Choose your way to get vectors: 1: Tfidf, 2: LSI, 3: word2vec, 4: doc2vec: ")
    if y == 1:
        corpus_train, newsgroups_train, corpus_test, newsgroups_test = get_corpus()
        words_train = []
        words_test = []
        for i in xrange(len(newsgroups_train.target)):
            words_train.append(' '.join(corpus_train[i]))
        for i in xrange(len(newsgroups_test.target)):
            words_test.append(' '.join(corpus_test[i]))
        vectorizer = TfidfVectorizer(sublinear_tf=True)
        doc_vectors_train = vectorizer.fit_transform(words_train)
        doc_vectors_test = vectorizer.transform(words_test)
        doc_vectors_train = doc_vectors_train.toarray()
        doc_vectors_test = doc_vectors_test.toarray()
        print doc_vectors_test[1]
        #doc_vectors_train, doc_vectors_test = vector_dimension_reduction(doc_vectors_train,doc_vectors_test)
        csvname = "tfidf"
        write_vec_to_csv(doc_vectors_train, doc_vectors_test, newsgroups_train, newsgroups_test, csvname)

    elif y ==2:
        corpus_train, newsgroups_train, corpus_test, newsgroups_test = get_corpus()
        words_train = []
        words_test = []
        for i in xrange(len(newsgroups_train.target)):
            words_train.append(' '.join(corpus_train[i]))
        for i in xrange(len(newsgroups_test.target)):
            words_test.append(' '.join(corpus_test[i]))
        vectorizer = TfidfVectorizer()
        vectors_train = vectorizer.fit_transform(words_train)
        vectors_test = vectorizer.transform(words_test)
        svd = TruncatedSVD(n_components=300)
        vectorizer = make_pipeline(svd, Normalizer(copy=False))
        doc_vectors_train = vectorizer.fit_transform(vectors_train)
        doc_vectors_test = vectorizer.transform(vectors_test)
        #doc_vectors_train, doc_vectors_test = vector_dimension_reduction(doc_vectors_train, doc_vectors_test)
        csvname = "LSI"
        write_vec_to_csv(doc_vectors_train, doc_vectors_test, newsgroups_train, newsgroups_test,csvname)

    elif y ==3:
        corpus_train, newsgroups_train, corpus_test, newsgroups_test = get_corpus()

        vector_source = input("Vector source: 1: GoogleNews-vectors-negative300.bin.gz:, 2: Build vectors on 20newsgroup: ")
        if vector_source ==1:
            doc_vectors_train = []
            doc_vectors_test = []
            filename = 'GoogleNews-vectors-negative300.bin.gz'
            word2vec_model = Word2Vec.load_word2vec_format(filename, binary=True)


            word2vec_model.init_sims(replace=True)
            corpus_train, newsgroups_train.data, newsgroups_train.target, newsgroups_train.filenames = filter_docs(corpus_train, newsgroups_train.data, newsgroups_train.target, newsgroups_train.filenames, lambda doc: vector_representation_filter(word2vec_model, doc))
            corpus_test, newsgroups_test.data, newsgroups_test.target, newsgroups_test.filenames = filter_docs(corpus_test, newsgroups_test.data, newsgroups_test.target, newsgroups_test.filenames, lambda doc: vector_representation_filter(word2vec_model, doc))


            doc_vector_method = input("Please choose method to get doc vectors from word vectors, 1: averaging, 2: tfidf weighting: ")
            #averaging word vectors
            if doc_vector_method ==1:
                doc_vectors_train, doc_vectors_test = averaging_word2vec(word2vec_model, corpus_train, corpus_test)

                #doc_vectors_train, doc_vectors_test = vector_dimension_reduction(doc_vectors_train, doc_vectors_test)
                csvname = "word2vec_google_vectors_averaging"
                write_vec_to_csv(doc_vectors_train, doc_vectors_test, newsgroups_train, newsgroups_test, csvname)
            #tfidf averaging word vectors
            elif doc_vector_method ==2:
                doc_vectors_train, doc_vectors_test = tfidf_word2vec(word2vec_model, corpus_train, corpus_test)
                #doc_vectors_train, doc_vectors_test = vector_dimension_reduction(doc_vectors_train, doc_vectors_test)
                csvname = "word2vec_google_vectors_tfidf"
                write_vec_to_csv(doc_vectors_train, doc_vectors_test, newsgroups_train, newsgroups_test,csvname)

        elif vector_source ==2:
            sentences = []
            doc_vectors_train=[]
            doc_vectors_test=[]
            for w in newsgroups_train.data:
                sentences += data_to_sentences(w,tokenizer)
            for  w in newsgroups_test.data:
                sentences += data_to_sentences(w,tokenizer)
            # Set values for various parameters
            num_features = 300  # Word vector dimensionality
            min_word_count = 0  # Minimum word count
            num_workers = 4  # Number of threads to run in parallel
            context = 10  # Context window size
            downsampling = 1e-3  # Downsample setting for frequent words

            word2vec_model = Word2Vec(sentences, workers=num_workers, \
                                                size=num_features, min_count = min_word_count, \
                                                 window = context, sample = downsampling)

            word2vec_model.init_sims(replace=True)
            corpus_train, newsgroups_train.data, newsgroups_train.target, newsgroups_train.filenames = filter_docs(corpus_train,newsgroups_train.data,newsgroups_train.target, newsgroups_train.filenames, lambda doc: vector_representation_filter(word2vec_model, doc))
            corpus_test, newsgroups_test.data, newsgroups_test.target, newsgroups_test.filenames = filter_docs(corpus_test, newsgroups_test.data, newsgroups_test.target, newsgroups_test.filenames, lambda doc: vector_representation_filter(word2vec_model, doc))

            doc_vectors_method = input("Please choose method to get doc vectors from word vectors, 1: averaging, 2: tfidf weighting: ")

            if doc_vectors_method ==1:
                doc_vectors_train, doc_vectors_test = averaging_word2vec(word2vec_model, corpus_train, corpus_test)
                # print doc_vectors_train
                #
                #doc_vectors_train, doc_vectors_test = vector_dimension_reduction(doc_vectors_train,doc_vectors_test)
                csvname = "word2vec_averaging"
                write_vec_to_csv(doc_vectors_train, doc_vectors_test, newsgroups_train, newsgroups_test, csvname)
            elif doc_vectors_method ==2:
                doc_vectors_train, doc_vectors_test = tfidf_word2vec(word2vec_model,corpus_train,corpus_test)
                #print doc_vectors_test
                #doc_vectors_train, doc_vectors_test = vector_dimension_reduction(doc_vectors_train, doc_vectors_test)
                csvname = "word2vec_tfidf"
                write_vec_to_csv(doc_vectors_train, doc_vectors_test, newsgroups_train, newsgroups_test, csvname)

    elif y ==4:
        corpus_train, newsgroups_train, corpus_test, newsgroups_test = get_corpus()
        doc_vectors_train = []
        doc_vectors_test = []
        documents = MyTaggedDocument(corpus_train, newsgroups_train, corpus_test, newsgroups_test)
        doc2vec_model = Doc2Vec(size=300, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025)
        doc2vec_model.build_vocab(documents)
        for epoch in range(10):
            doc2vec_model.train(documents)
            doc2vec_model.alpha -= 0.002  # decrease the learning rate
            doc2vec_model.min_alpha = doc2vec_model.alpha  # fix the learning rate, no deca
            doc2vec_model.train(documents)
        for i in xrange(len(newsgroups_train.filenames)):
            #doc_vector_train.append(doc2vec_model.docvecs[newsgroups_train.target_names[newsgroups_train.target[i]] + str(i)])
            doc_vectors_train.append(doc2vec_model.docvecs[i])
        for i in xrange(len(newsgroups_test.filenames)):
            #doc_vector_test.append(doc2vec_model.docvecs[newsgroups_test.target_names[newsgroups_test.target[i]] + str(i + len(newsgroups_train.filenames) - 1)])
            doc_vectors_test.append(doc2vec_model.docvecs[i + len(newsgroups_train.filenames) - 1])
        print doc_vectors_train[1]
        #doc_vectors_train, doc_vectors_test = vector_dimension_reduction(doc_vectors_train, doc_vectors_test)
        csvname = "doc2vec"
        write_vec_to_csv(doc_vectors_train, doc_vectors_test, newsgroups_train, newsgroups_test, csvname)

    return doc_vectors_train, doc_vectors_test, newsgroups_train, newsgroups_test

def vector_dimension_reduction(doc_vectors_train, doc_vectors_test):
    tsne = TSNE(n_components=2, init='pca', random_state=None, method='barnes_hut', n_iter=1000)

    doc_vectors_train = np.array(doc_vectors_train)
    doc_vectors_test = np.array(doc_vectors_test)
    print doc_vectors_train
    doc_vector_train_tsne = tsne.fit_transform(doc_vectors_train)
    doc_vector_test_tsne = tsne.fit_transform(doc_vectors_test)
    #print new_doc_vec_tsne

    return doc_vector_train_tsne, doc_vector_test_tsne

#output vectors of which dimension is reduced into csv
def write_vec_to_csv(doc_vector_train, doc_vector_test,newsgroups_train, newsgroups_test,csvname):
    target_name_train = []
    target_name_test = []
    for i in xrange(len(newsgroups_train.target)):
        target_name_train.append(newsgroups_train.target_names[newsgroups_train.target[i]])
    for i in xrange(len(newsgroups_test.target)):
        target_name_test.append(newsgroups_test.target_names[newsgroups_test.target[i]])
    # print len(target_name_train)
    # print doc_vector_train_tsne.shape
    # print len(newsgroups_train.filenames)
    output_train = np.column_stack((target_name_train, newsgroups_train.target, doc_vector_train, newsgroups_train.filenames))
    output_test = np.column_stack((target_name_test, newsgroups_test.target, doc_vector_test,newsgroups_test.filenames))
    output_train = np.array(output_train)
    output_test = np.array(output_test)

    with open(csvname + 'train.csv', 'w') as f:
        fieldnames = ['target_names', 'target', 'X', 'Y', 'filenames']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer = csv.writer(f)

        writer.writerows(output_train)

    with open(csvname + 'test.csv', 'w') as f:
        fieldnames = ['target_names', 'target', 'X', 'Y', 'filenames']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer = csv.writer(f)

        writer.writerows(output_test)

        #
        # print doc2vec_model.docvecs.most_similar(['talk.politics.guns5'])
        #
        # print newsgroups_train.target_names[newsgroups_train.target['talk.politics.guns5']]
        #
        # test_target = newsgroups_train.target + newsgroups_test.target
        # for item in doc2vec_model.docvecs.most_similar('talk.politics.guns5'):
        #
        #     print newsgroups_train.target_names[test_target[item[0]]]
        #print doc2vec_model.docvecs['alt.atheism50'].shape

#output vectors into csv
# def write_vec_to_csv(doc_vector_train, doc_vector_test,newsgroups_train, newsgroups_test,csvname):
#     target_name_train = []
#     target_name_test = []
#     for i in xrange(len(newsgroups_train.target)):
#         target_name_train.append(newsgroups_train.target_names[newsgroups_train.target[i]] + str(i))
#     for i in xrange(len(newsgroups_test.target)):
#         target_name_test.append(newsgroups_test.target_names[newsgroups_test.target[i]] + str(i))
#     # print len(target_name_train)
#     # print doc_vector_train_tsne.shape
#     # print len(newsgroups_train.filenames)
#     output_train = np.column_stack((target_name_train, doc_vector_train))
#     output_test = np.column_stack((target_name_test, doc_vector_test))
#     output_train = np.array(output_train)
#     output_test = np.array(output_test)
#
#     with open(csvname + 'train.csv', 'w') as f:
#         fieldnames = []
#         for i in xrange(len(doc_vector_train[1])):
#             fieldnames.append('axis'+str(i))
#         #fieldnames = ['target_names', 'target', 'X', 'Y', 'filenames']
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer = csv.writer(f)
#
#         writer.writerows(output_train)
#
#     with open(csvname + 'test.csv', 'w') as f:
#         fieldnames = []
#         for i in xrange(len(doc_vector_test[1])):
#             fieldnames.append('axis'+str(i))
#         #fieldnames = ['target_names', 'target', 'X', 'Y', 'filenames']
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer = csv.writer(f)
#
#         writer.writerows(output_test)
#
#         #
#         # print doc2vec_model.docvecs.most_similar(['talk.politics.guns5'])
#         #
#         # print newsgroups_train.target_names[newsgroups_train.target['talk.politics.guns5']]
#         #
#         # test_target = newsgroups_train.target + newsgroups_test.target
#         # for item in doc2vec_model.docvecs.most_similar('talk.politics.guns5'):
#         #
#         #     print newsgroups_train.target_names[test_target[item[0]]]
#         #print doc2vec_model.docvecs['alt.atheism50'].shape

# def averaging_word2vec(word2vec_model, doc):
#     # remove out-of-vocabulary words
#     doc = [word for word in doc if word in word2vec_model.vocab]
#     # print "*********************************************"
#     # # print word2vec_model[doc]
#     print np.mean(word2vec_model[doc], axis=0)
#     return np.mean(word2vec_model[doc], axis=0)

def vector_representation_filter(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)

def averaging_word2vec(word2vec_model, corpus_train, corpus_test):
    #print old_file_list
    doc_vectors_train = np.array([[0]*300])
    doc_vectors_test = np.array([[0]*300])
    print "Calculating vectors for documents/sentences using averaging score"
    for corpus in corpus_train:
        temp = []
        #nwords = 0
        corpus_filtered = [word for word in corpus if word in word2vec_model.vocab]
        for j in xrange(len(corpus_filtered)):
            new_style = corpus_filtered[j]
            temp.append(word2vec_model[str(new_style)])
        # print doc_vectors_train.shape
        # print np.mean(temp,axis=0).shape
        doc_vectors_train = np.append(doc_vectors_train,[np.mean(temp,axis=0)], axis=0)

    doc_vectors_train = np.delete(doc_vectors_train,0,axis=0)
        #doc_vectors_train.append(np.mean(temp,axis=0))

    for corpus in corpus_test:
        temp = []
        #nwords = 0
        corpus_filtered = [word for word in corpus if word in word2vec_model.vocab]
        for j in xrange(len(corpus_filtered)):
            new_style = corpus_filtered[j]
            #print model[str(new_style)]
            temp.append(word2vec_model[str(new_style)])
        doc_vectors_test = np.append(doc_vectors_test, [np.mean(temp, axis=0)], axis=0)
        #doc_vectors_test.append(np.mean(temp, axis=0))
    doc_vectors_test = np.delete(doc_vectors_test, 0, axis=0)
    # print np.asarray(doc_vectors_train).shape
    # print np.asarray(doc_vectors_test).shape
    # print np.asarray(doc_vectors_test[1]).shape
    # print doc_vectors_train[1]
    return doc_vectors_train, doc_vectors_test

def tfidf_word2vec(word2vec_model, corpus_train, corpus_test):
    print "Calculating vectors for documents/sentences using averaging tfidf score"
    idf = inverse_document_frequencies(corpus_train)
    min_idf = 10000000.0
    doc_vector_train=[]
    doc_vector_test=[]
    y = input("Choose mothod to calculate tf score: 1: term_frequency, 2: sublinear_term_frequency, 3: augmented_term_frequency: ")
    for i in xrange(len(corpus_train)):
        # temp = [0] * 300
        temp = np.zeros((300,), dtype="float32")
        nwords = 0
        doc_train = corpus_train[i]
        #filter words which not exist in vocabulary
        doc_train = [word for word in doc_train if word in word2vec_model.vocab]

        for w in corpus_train[i]:
            if min_idf > idf[w]:
                min_idf = idf[w]
        for j in xrange(len(doc_train)):
            nwords = nwords + 1
            if y == 1:
                tf = term_frequency(doc_train[j], doc_train)
            elif y == 2:
                tf = sublinear_term_frequency(doc_train[j], doc_train)
            elif y == 3:
                tf = augmented_term_frequency(doc_train[j], doc_train)
             #calculate tfidf
            tfidf = tf * idf[doc_train[j]]
            temp = np.add(temp, map(lambda x: x * tfidf, word2vec_model[str(doc_train[j])]))
        doc_vector_train.append(np.divide(temp, nwords))

    for i in xrange(len(corpus_test)):
        # temp = [0] * 300
        temp = np.zeros((300,), dtype="float32")
        nwords = 0
        doc_test = corpus_test[i]
        doc_test = [word for word in doc_test if word in word2vec_model.vocab]
        for j in xrange(len(doc_test)):
            nwords = nwords + 1
            if y == 1:
                tf = term_frequency(doc_test[j], doc_test)
            elif y == 2:
                tf = sublinear_term_frequency(doc_test[j], doc_test)
            elif y == 3:
                tf = augmented_term_frequency(doc_test[j], doc_test)
            if doc_test[j] in idf.keys():
                tfidf = tf * idf[doc_test[j]]

            elif doc_test[j] not in idf.keys():
                tfidf = tf * min_idf
            temp = np.add(temp, map(lambda x: x * tfidf, word2vec_model[str(doc_test[j])]))

        doc_vector_test.append(np.divide(temp, nwords))
    return doc_vector_train, doc_vector_test

#def doc2vec():







def main():
 get_vectors()



if __name__ == "__main__":
 main()



