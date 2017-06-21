import nltk
import sys
import gensim
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import numpy as np
from nltk import word_tokenize
from string import punctuation
from nltk import download
from nltk.corpus import stopwords
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import Normalizer
from word2vec import *
import keras
from keras.preprocessing import text
nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
reload(sys)
sys.setdefaultencoding('utf8')
def get_data():
    targets = ['cameras','laptops','mobilephone','tablets','TVs','video_surveillance']
    path =['/home/flippped/thesis_project/Reviews/Amazon_30_50/camera_no',
           '/home/flippped/thesis_project/Reviews/Amazon_30_50/laptop_no',
           '/home/flippped/thesis_project/Reviews/Amazon_30_50/mobilephone_no',
           '/home/flippped/thesis_project/Reviews/Amazon_30_50/tablets_no',
           '/home/flippped/thesis_project/Reviews/Amazon_30_50/TVs_no',
           '/home/flippped/thesis_project/Reviews/Amazon_30_50/video_surveillance_no']
    data = []
    target=[]
    filename = []
    for j in  xrange(len(path)):

        print path[j]
        #for i in path[j]:
        for file in sorted(os.listdir(path[j])):

            with open(os.path.join(path[j], file), 'r') as f:
                 document = f.read().lower()
                 target.append(targets[j])
                 filename.append(file)
                 data.append(document)

    #print len(target)
    print len(filename)
    print len(target)
    return data,target,filename

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
    for i in range(10):
        print data_to_sent[i]


    sentences = []

    for i in data_to_sent:
        tmp = []
        list2 = [x for x in i if x != []]
        for j in list2:
            remove_stopwords = ([word for word in j if word not in stop_words])
            # stemmings = ([stemmer.stem(plural) for plural in remove_stopwords])
            # tmp.append(stemmings)
            tmp.append(remove_stopwords)
        sentences.append(tmp)

    sentences, data, target, filenames = filter_docs(sentences, data, target, filename, lambda doc: (len(doc) != 0))
    return sentences, target, filenames

def filter_docs(corpus,  texts, labels, filenames, condition_on_doc):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """

    number_of_docs = len(corpus)
    corpus, texts,labels, filenames = zip(*((x, y,z,w) for x, y,z,w in zip(corpus, texts,labels, filenames) if len(x) > 0))

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return corpus, texts, labels, filenames

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
            #corpus_train,  targets, filenames = filter_docs(corpus, targets, filenames, lambda doc: vector_representation_filter(word2vec_model, doc))


            words_train_tmp = []
            for i in corpus:
                tmp = []
                for j in i:
                    for k in j:
                        tmp.append(k)
                words_train_tmp.append(tmp)
            corpus_train = words_train_tmp

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

    with open('reviews_30_50_' + csvname + '_.csv', 'w') as f:
        fieldnames = ['target_names', 'filenames']
        for i in xrange(len(doc_vector_train[1])):
            fieldnames.append('x'+ str(i))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer = csv.writer(f)

        writer.writerows(output_train)



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

# def vector_representation_filter(word2vec_model, doc):
#     """check if at least one word of the document is in the
#     word2vec dictionary"""
#
#     return not all(word not in word2vec_model.wv.vocab for word in doc)

def averaging_word2vec(word2vec_model, corpus_train):
    #print old_file_list
    doc_vectors_train = np.array([[0]*300])
    #doc_vectors_train = []
    print "Calculating vectors for documents/sentences using averaging score"
    for corpus in corpus_train:
        temp = []
        #nwords = 0
        # if use google vectors
        corpus_filtered = [word for word in corpus if word in word2vec_model.vocab]
        # if not use google vectors
        # corpus_filtered = [word for word in corpus if word in word2vec_model.wv.vocab]
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

        # if use google vectors
        #doc_train = [word for word in doc_train if word in word2vec_model.vocab]
        # if not use google vectors
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


    return doc_vector_train

#def doc2vec():







def main():
 get_vectors()
 #get_corpus()
 #get_data()



if __name__ == "__main__":
 main()



