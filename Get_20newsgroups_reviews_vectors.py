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

stop_words = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
reload(sys)
sys.setdefaultencoding('utf8')
def get_data():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories= ['alt.atheism',
 'comp.sys.ibm.pc.hardware',
 'rec.motorcycles',
 'rec.sport.baseball',
 'sci.electronics',
 'sci.med',
 'talk.politics.guns'])

    for i in range(20):
        print newsgroups_train.target[i]
        print newsgroups_train.filenames[i]
    return newsgroups_train.data, newsgroups_train.target, newsgroups_train.filenames


class MyTaggedDocument(object):
    def __init__(self, corpus, targets, filenames):
        self.corpus = corpus
        self.docList = filenames
        self.target = targets

    def __iter__(self):
        for idx, doc in enumerate(self.corpus):
            # yield TaggedDocument(doc, [self.newsgroups_train.target_names[self.target[idx]] + str[idx]])
            yield TaggedDocument(doc, [idx])


# def preprocess(text):
#
#     r = re.compile(r'[{}]+'.format(re.escape(punctuation)))
#     doc = []
#     text = r.sub('', text)
#     # for i in text:
#     #     doc.append(i.translate(None,punctuation))
#
#     #text = text.translate(' ',punctuation)
#     #text = re.sub('@#$%&*_~?![]{};:.*[\r\n]*-,<>', '', text, flags=re.MULTILINE)
#     #doc = " ".join(doc)
#     print text
#     #text = re.sub("[^a-zA-Z]", " ", text)
#     doc = word_tokenize(text)
#     #doc = [word for word in doc if word.isalpha()]
#     doc = [word for word in doc if word not in stop_words]
#
#     return doc


def get_corpus():
    stemmer = PorterStemmer()
    data,target,filename = get_data()

    data_to_sent = []
    # doc = ' '.join(re.findall(r"[\w']+|[.,!?;/-]", data))

    for w in data:
        data_to_sent.append(data_to_sentences(w,tokenizer))

    print data_to_sent[1]


    sentences = []

    for i in data_to_sent:
        tmp = []
        list2 = [x for x in i if x != []]
        for j in list2:
            remove_stopwords = ([word for word in j if word not in stop_words])
            stemmings = ([stemmer.stem(plural) for plural in remove_stopwords])
            tmp.append(stemmings)
            # tmp.append(remove_stopwords)
        sentences.append(tmp)

    sentences, target, filenames = filter_docs(sentences, target, filename)
    return sentences, target, filenames

def filter_docs(corpus, labels, filenames):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """

    number_of_docs = len(corpus)
    corpus, labels,filenames = zip(*((x, y,z) for x, y,z in zip(corpus,labels,filenames) if len(x) > 0))

    corpus = list(corpus)
    labels = list(labels)
    filenames = list(filenames)
    print("{} docs removed".format(number_of_docs - len(corpus)))

    return corpus,labels, filenames

def get_vectors():
    y = input("Choose your way to get vectors: 1: Tfidf, 2: LSA, 3: word2vec, 4: doc2vec: ")
    if y == 1:
        sentences, targets, filenames = get_corpus()
        words_train = []
        words_train_tmp = []
        for i in sentences:
            tmp = []
            for j in i:
                for k in j:
                    tmp.append(k)
            words_train_tmp.append(tmp)
        for i in xrange(len(targets)):
            #
            words_train.append(' '.join(words_train_tmp[i]))
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=5000)
        doc_vectors_train = vectorizer.fit_transform(words_train)
        doc_vectors_train = doc_vectors_train.toarray()
        print doc_vectors_train.shape

        #doc_vectors_train= vector_dimension_reduction(doc_vectors_train)
        csvname = "tfidf"
        write_vec_to_csv(doc_vectors_train,  targets, filenames,  csvname)

    elif y ==2:
        corpus, targets, filenames = get_corpus()
        words_train = []
        words_train_tmp = []
        for i in corpus:
            tmp = []
            for j in i:
                for k in j:
                    tmp.append(k)
            words_train_tmp.append(tmp)
        for i in xrange(len(targets)):
            #
            words_train.append(' '.join(words_train_tmp[i]))
        vectorizer = TfidfVectorizer()
        vectors_train = vectorizer.fit_transform(words_train)
        svd = TruncatedSVD(n_components=300)
        vectorizer = make_pipeline(svd, Normalizer(copy=False))
        doc_vectors_train = vectorizer.fit_transform(vectors_train)
        #doc_vectors_train = vector_dimension_reduction(doc_vectors_train)
        csvname = "LSI"
        write_vec_to_csv(doc_vectors_train, targets, filenames, csvname)

    elif y ==3:
        corpus, targets, filenames = get_corpus()
        vector_source = input("Vector source: 1: GoogleNews-vectors-negative300.bin.gz:, 2: Build vectors on 20newsgroup: ")
        if vector_source ==1:
            doc_vectors_train = []
            filename = 'GoogleNews-vectors-negative300.bin.gz'
            #word2vec_model = Word2Vec.load_word2vec_format(filename, binary=True)
            word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)


            word2vec_model.init_sims(replace=True)

            corpus_train = []
            for i in range(len(corpus)):
                tmp = []
                for j in corpus[i]:
                    for k in j:
                        if str(k) in word2vec_model.vocab:
                            tmp.append(k)
                        else:
                            continue
                corpus_train.append(tmp)

            corpus_train,  targets, filenames = filter_docs(corpus_train, targets, filenames)
            # print corpus_train[0]
            # print corpus_train[1]
            # words_train_tmp = []
            # for i in corpus:
            #     tmp = []
            #     for j in i:
            #         for k in j:
            #             tmp.append(k)
            #     words_train_tmp.append(tmp)
            # corpus_train = words_train_tmp

            doc_vector_method = input("Please choose method to get doc vectors from word vectors, 1: averaging, 2: tfidf weighting: ")
            #averaging word vectors
            if doc_vector_method ==1:
                doc_vectors_train = averaging_word2vec(word2vec_model, corpus_train)
                #doc_vectors_train = vector_dimension_reduction(doc_vectors_train)
                csvname = "word2vec_google_vectors_averaging"
                write_vec_to_csv(doc_vectors_train, targets, filenames, csvname)

            #tfidf averaging word vectors
            elif doc_vector_method ==2:
                doc_vectors_train = tfidf_word2vec(word2vec_model, corpus_train)
                #doc_vectors_train = vector_dimension_reduction(doc_vectors_train)
                csvname = "word2vec_google_vectors_tfidf"
                write_vec_to_csv(doc_vectors_train, targets, filenames, csvname)

        elif vector_source ==2:
            sentences = []
            doc_vectors_train=[]
            for w in corpus:
                for j in w:
                    sentences.append(j)

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
            # corpus_train, targets, filenames = filter_docs(corpus, targets, filenames,
            #                                                      lambda doc: vector_representation_filter(
            #                                                          word2vec_model, doc))
            words_train_tmp = []
            for i in corpus:
                tmp = []
                for j in i:
                    for k in j:
                        tmp.append(k)
                words_train_tmp.append(tmp)
            corpus_train = words_train_tmp
            corpus_train, targets, filenames = filter_docs(corpus_train, targets, filenames)

            doc_vectors_method = input("Please choose method to get doc vectors from word vectors, 1: averaging, 2: tfidf weighting: ")

            if doc_vectors_method ==1:
                doc_vectors_train= averaging_word2vec(word2vec_model, corpus_train)
                # print doc_vectors_train
                #
                #doc_vectors_train = vector_dimension_reduction(doc_vectors_train)
                csvname = "word2vec_averaging"
                write_vec_to_csv(doc_vectors_train, targets, filenames, csvname)
            elif doc_vectors_method ==2:
                doc_vectors_train= tfidf_word2vec(word2vec_model,corpus_train)
                #doc_vectors_train = vector_dimension_reduction(doc_vectors_train)
                csvname = "word2vec_tfidf"
                write_vec_to_csv(doc_vectors_train, targets, filenames, csvname)

    elif y ==4:
        corpus, targets, filenames = get_corpus()
        words_train_tmp = []
        for i in corpus:
            tmp = []
            for j in i:
                for k in j:
                    tmp.append(k)
            words_train_tmp.append(tmp)
        doc_vectors_train = []
        corpus = words_train_tmp
        documents = MyTaggedDocument(corpus, targets, filenames)
        doc2vec_model = Doc2Vec(size=300, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025)
        doc2vec_model.build_vocab(documents)
        for epoch in range(10):
            doc2vec_model.train(documents)
            doc2vec_model.alpha -= 0.002  # decrease the learning rate
            doc2vec_model.min_alpha = doc2vec_model.alpha  # fix the learning rate, no deca
            doc2vec_model.train(documents)
        for i in xrange(len(filenames)):
            #doc_vector_train.append(doc2vec_model.docvecs[newsgroups_train.target_names[newsgroups_train.target[i]] + str(i)])
            doc_vectors_train.append(doc2vec_model.docvecs[i])

        #doc_vectors_train = vector_dimension_reduction(doc_vectors_train)
        csvname = "doc2vec"
        write_vec_to_csv(doc_vectors_train, targets, filenames, csvname)

    return doc_vectors_train, targets, filenames

def vector_dimension_reduction(doc_vectors_train):
    tsne = TSNE(n_components=2, init='pca', random_state=None, method='barnes_hut', n_iter=1000)

    doc_vectors_train = np.array(doc_vectors_train)
    #print doc_vectors_train
    doc_vector_train_tsne = tsne.fit_transform(doc_vectors_train)
    #print new_doc_vec_tsne

    return doc_vector_train_tsne

#output vectors of which dimension is reduced into csv
def write_vec_to_csv(doc_vector_train,targets,filenames, csvname):

    output_train = np.column_stack((targets,filenames, doc_vector_train))
    #output_train = np.array(output_train)

    with open('20newsgroups_' + csvname + '_.csv', 'w') as f:
        fieldnames = ['target_names', 'filenames']
        for i in xrange(len(doc_vector_train[1])):
            fieldnames.append('x'+ str(i))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer = csv.writer(f)

        writer.writerows(output_train)





def averaging_word2vec(word2vec_model, corpus_train):
    #print old_file_list
    doc_vectors_train = np.array([[0]*300])
    #doc_vectors_train = []
    print "Calculating vectors for documents/sentences using averaging score"
    for corpus in corpus_train:
        temp = []
        #nwords = 0
        # # if use google vectors
        # corpus_filtered = [word for word in corpus if word in word2vec_model.vocab]
        # if not use google vectors
        corpus_filtered = [word for word in corpus if word in word2vec_model.wv.vocab]
        for j in xrange(len(corpus_filtered)):
            new_style = corpus_filtered[j]
            temp.append(word2vec_model[str(new_style)].tolist())
        # print doc_vectors_train.shape
        # print np.mean(temp,axis=0).shape
        doc_vectors_train = np.append(doc_vectors_train,[np.mean(temp,axis=0)], axis=0)
        #doc_vectors_train.append(temp)


    doc_vectors_train = np.delete(doc_vectors_train,0,axis=0)
        #doc_vectors_train.append(np.mean(temp,axis=0))

    return doc_vectors_train

def tfidf_word2vec(word2vec_model, corpus_train):
    print "Calculating vectors for documents/sentences using averaging tfidf score"
    idf = inverse_document_frequencies(corpus_train)
    min_idf = 10000000.0
    doc_vector_train=[]

    y = input("Choose mothod to calculate tf score: 1: term_frequency, 2: sublinear_term_frequency, 3: augmented_term_frequency: ")
    for i in xrange(len(corpus_train)):
        # temp = [0] * 300
        temp = np.zeros((300,), dtype="float32")
        nwords = 0
        doc_train = corpus_train[i]
        #filter words which not exist in vocabulary

        # # if use google vectors
        # doc_train = [word for word in doc_train if word in word2vec_model.vocab]
        # if not use google vectors
        doc_train = [word for word in doc_train if word in word2vec_model.wv.vocab]

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


    return doc_vector_train

def main():
 get_vectors()

    # get_data()

if __name__ == "__main__":
 main()
