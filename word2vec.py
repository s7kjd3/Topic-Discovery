import os
import re
import sys
import math
import magic
import imghdr
import PyPDF2
import time
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import numpy as np
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import nltk.data
from nltk.stem.porter import *
from scipy.cluster.hierarchy import ward, dendrogram,linkage
import logging
from gensim.models import word2vec
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
import csv
reload(sys)
sys.setdefaultencoding('utf8')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def pdftotext():
#transform pdf into text file
    #print path
    path = [[],[]]
    path[0] = input('Please input the old documents directory(style is like [\' \',\' \']) : ')
    path[1] = input('Please input the new document directory(style is like [\' \',\' \']) : ')
    for j in  xrange(len(path)):
        for i in path[j]:
            for filename in os.listdir(i):
                check_path = os.path.join(i, filename)
                if os.path.isdir(check_path):
                    continue
                else:
                    with open(os.path.join(i, filename), 'r') as f:
                        #determine file type
                        file_type = magic.from_buffer(f.read(1024), mime=True)
                        #print file_type
                        if file_type == 'application/pdf':
                            if not os.path.exists('pdfTOtext' + str(j)):
                                os.makedirs('pdfTOtext' + str(j))
                                path[j].append('pdfTOtext' + str(j))
                            if os.path.exists('pdfTOtext' + str(j)):
                                path[j].append('pdfTOtext' + str(j))

                            pdfReader = PyPDF2.PdfFileReader(f)
                        #create file and extract pdf content into it
                            file = open(os.path.join('pdfTOtext' + str(j), filename + ".txt"), 'w+')
                            #file = open(filename + ".txt",'w+')
                            for i in xrange(pdfReader.getNumPages()):
                                page = pdfReader.getPage(i)
                                page_content = page.extractText().encode('utf-8')
                                file.write(page_content)

                            file.close()
                        else:
                            continue
        #print path
    return path


def get_text_content():
    #SYMBOLS = '{}()[]@#$%?!.,:;+-*/&|<>=~$1234567890"'
    stops = set(stopwords.words("english"))
    path = pdftotext()
    sentences = []
    old_file_list = [[],[]]
    new_file_list = [[],[]]
    print "Extracting sentences..."
    #get contect from text file
    for j in xrange(len(path)):
        #data = [[], []]
        for i in path[j]:

            for filename in os.listdir(i):
                check_path = os.path.join(i, filename)
                if os.path.isdir(check_path):
                    continue
                else:
                    file_name, file_extension = os.path.splitext(filename)
                    #print file_name
                    if file_extension != '.pdf':
                        with open(os.path.join(i, filename), 'r') as f:

                              document = f.read()

                              document = document.replace("\n", " ")
                              document = document.replace("\t", " ")

                              sentences += data_to_sentences(document, tokenizer)
                              sentences_solo = data_to_sentences(document, tokenizer)
                              # print tmp
                              # print "****************"
                        #store words of document in list
                        if j == 0:
                             tmp = []
                             old_file_list[0].append(filename)
                             #old_file_list[1].append(' '.join(list(itertools.chain.from_iterable(t))))
                             for k in xrange(len(sentences_solo)):
                                 tmp.extend(sentences_solo[k])
                             old_file_list[1].append(tmp)
                        if j == 1:
                             tmp = []
                             new_file_list[0].append(filename)
                             #new_file_list[1].append(' '.join(list(itertools.chain.from_iterable(sentences))))
                             for k in xrange(len(sentences_solo)):
                                tmp.extend(sentences_solo[k])
                             new_file_list[1].append(tmp)
                             #print list(itertools.chain.from_iterable(sentences))
                    else:
                             continue
    print sentences

    return sentences, old_file_list, new_file_list

def data_to_wordlist(data, remove_stopwords=True ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(data).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #print review_text
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    stemmer = PorterStemmer()
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        words = [stemmer.stem(plural) for plural in words]
    #print words
    # 5. Return a list of words
    return(words)



def data_to_sentences(data, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    #print data
    raw_sentences = tokenizer.tokenize(data.strip())

    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( data_to_wordlist( raw_sentence, \
              remove_stopwords ))

    #print sentences
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def word_to_vector():
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    sentences, old_file_list, new_file_list = get_text_content()
    #print old_file_list[1]
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 0  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)

    print "Training model..."
    #`sg` defines the training algorithm. By default (`sg=0`), CBOW is used. Otherwise (`sg=1`), skip-gram is employed.

    #`min_count` = ignore all words with total frequency lower than this

    #`max_vocab_size` = limit RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Every 10 million word types
    # need about 1GB of RAM. Set to `None` for no limit (default).

    #`workers` = use this many worker threads to train the model (=faster training with multicore machines).

    #`hs` = if 1, hierarchical softmax will be used for model training.If set to 0 (default), and `negative` is non-zero, negative sampling will be used.

    #`negative` = if > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20).
    #Default is 5. If set to 0, no negative samping is used.

    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                              hs = 1, negative= 0, size=num_features, min_count=min_word_count, \
                              window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    # print model.most_similar("dog")
    print model['dog']
    #print old_file_list[0][1]
    # for i in xrange(len(old_file_list[1][1])):
    #     print old_file_list[1][1][i]
    return model, old_file_list, new_file_list


def average_sentence_vec():
    model, old_file_list, new_file_list = word_to_vector()
    #print old_file_list
    print "Calculating vectors for documents/sentences using averaging score"
    old_doc_vec = [[],[]]
    old_doc_vec[0] = old_file_list[0]
    new_doc_vec = [[],[]]
    new_doc_vec[0] = new_file_list[0]
    for i in xrange(len(new_file_list[0])):
        temp = np.zeros((300,),dtype="float32")
        nwords = 0
        for j in xrange(len(new_file_list[1][i])):
            #print str(new_file_list[1][i][j])
            nwords = nwords + 1
            new_style = new_file_list[1][i][j]
            #print model[str(new_style)]
            temp = np.add(temp,model[str(new_style)])
            #temp = map(sum, izip(model[str(new_style)],temp))
            #print np.asarray(temp).shape
            #temp = temp + model[str(new_style)]
        print temp
        new_doc_vec[1].append(np.divide(temp,nwords))
    for i in xrange(len(old_file_list[0])):
        #temp = [0] * 300
        temp = np.zeros((300,), dtype="float32")
        nwords = 0
        for j in xrange(len(old_file_list[1][i])):
            nwords = nwords + 1
            new_style = old_file_list[1][i][j]
            temp = np.add(temp, model[str(new_style)])
            #temp = map(sum, izip(model[str(new_style)],temp))
        old_doc_vec[1].append(np.divide(temp,nwords))

    return new_doc_vec, old_doc_vec


def term_frequency(term, document):
    return document.count(term)

def sublinear_term_frequency(term, document):
    count = document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def augmented_term_frequency(term, document):
    max_count = max([term_frequency(t, document) for t in document])
    return (0.5 + ((0.5 * term_frequency(term, document)) / max_count))

def inverse_document_frequencies(documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, documents)
        idf_values[tkn] = 1 + math.log(len(documents) / (sum(contains_token)))
    return idf_values


def tfidf_sentence_vec(model, old_file_list, new_file_list):
    print old_file_list[1]

    print "Calculating vectors for documents/sentences using averaging tfidf score"
    old_doc_vec = [[], []]
    old_doc_vec[0] = old_file_list[0]
    new_doc_vec = [[], []]
    new_doc_vec[0] = new_file_list[0]
    total_nwords = 0
    #vectorizer = TfidfVectorizer()
    # print old_file_list[1]
    # old_doc_vec[1] = vectorizer.fit_transform(old_file_list[1])
    # new_doc_vec[1] = vectorizer.transform(new_file_list[1])
    idf = inverse_document_frequencies(old_file_list[1])
    # if words of new document cannot be found in old docuemnt, take the smallest value of tfidf score

    min_idf = 10000000.0
    for i in old_file_list[1]:
        for w in i:
            if min_idf > idf[w]:
                min_idf = idf[w]

    # word2weight_idf = defaultdict(
    #     lambda: min_idf,
    #     [(w, vectorizer.idf_[i]) for w, i in vectorizer.vocabulary_.items()])

    # for i in xrange(len(old_file_list[0])):
    #     for term in xrange(len(old_file_list[1][i])):
    #         total_nwords = total_nwords + 1

    y = input("Choose mothod to calculate tf score: 1: term_frequency, 2: sublinear_term_frequency, 3: augmented_term_frequency: ")


    for i in xrange(len(old_file_list[0])):
        # temp = [0] * 300
        temp = np.zeros((300,), dtype="float32")
        nwords = 0
        for j in xrange(len(old_file_list[1][i])):
            nwords = nwords + 1
            if y == 1:
                tf = term_frequency(old_file_list[1][i][j], old_file_list[1][i])
            elif y == 2:
                tf = sublinear_term_frequency(old_file_list[1][i][j], old_file_list[1][i])

            elif y == 3:
                tf = augmented_term_frequency(old_file_list[1][i][j], old_file_list[1][i])
            tfidf = tf * idf[old_file_list[1][i][j]]
            temp = np.add(temp, map(lambda x: x * tfidf, model[str(old_file_list[1][i][j])]))
            # temp = map(sum, izip(model[str(new_style)],temp))

        old_doc_vec[1].append(np.divide(temp, nwords))


    for i in xrange(len(new_file_list[0])):
        temp = np.zeros((300,), dtype="float32")

        for j in xrange(len(new_file_list[1][i])):

            nwords = nwords + 1
            if y == 1:
                tf = term_frequency(new_file_list[1][i][j], new_file_list[1][i])
            elif y == 2:
                tf = sublinear_term_frequency(new_file_list[1][i][j], new_file_list[1][i])
            elif y == 3:
                tf = augmented_term_frequency(new_file_list[1][i][j], new_file_list[1][i])

            if new_file_list[1][i][j] in idf.keys():
                tfidf = tf * idf[new_file_list[1][i][j]]

            elif new_file_list[1][i][j] not in idf.keys():
                tfidf = tf * min_idf

            temp = np.add(temp, map(lambda x: x * tfidf, model[str(new_file_list[1][i][j])]))


            #temp = np.add(temp, model[str(new_style)])

        new_doc_vec[1].append(np.divide(temp, nwords))

    return new_doc_vec, old_doc_vec


def dimension_reduction():
    model, old_file_list, new_file_list = word_to_vector()
    new_doc_vec_tsne = [[],[]]
    old_doc_vec_tsne = [[],[]]
    ave_or_tfidf = input(
        "Choose the way to represent documents through embedded vectors: 1:average embedded vectors, 2: average tfidf embedded vectors: ")
    if ave_or_tfidf == 1:
        new_doc_vec, old_doc_vec = average_sentence_vec()
    elif ave_or_tfidf == 2:
        new_doc_vec, old_doc_vec = tfidf_sentence_vec(model, old_file_list, new_file_list)
    new_doc_vec_tsne[0] = new_doc_vec[0]
    old_doc_vec_tsne[0] = old_doc_vec[0]
    tsne = TSNE(n_components=2, init='pca', random_state=None, method='barnes_hut', n_iter=1000)
    new_doc_vec_tsne[1] = tsne.fit_transform(new_doc_vec[1])
    old_doc_vec_tsne[1] = tsne.fit_transform(old_doc_vec[1])
    #print new_doc_vec_tsne
    print new_doc_vec_tsne[1]
    return new_doc_vec_tsne, old_doc_vec_tsne, ave_or_tfidf


def write_vec_to_csv():
    new_doc_vec_tsne, old_doc_vec_tsne, ave_or_tfidf = dimension_reduction()
    output = np.column_stack((new_doc_vec_tsne[0],new_doc_vec_tsne[1]))
    output = np.array(output)

    if ave_or_tfidf == 1:
        with open('word2vec_average.csv', 'w') as f:
            fieldnames = ['Name', 'X', 'Y']
            writer = csv.DictWriter(f,fieldnames=fieldnames)
            writer.writeheader()
            writer = csv.writer(f)

            writer.writerows(output)
    elif ave_or_tfidf == 2:
        with open('word2vec_tfidf.csv', 'w') as f:
            fieldnames = ['Name', 'X', 'Y']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer = csv.writer(f)

            writer.writerows(output)


def document_distance():
    model, old_file_list, new_file_list = word_to_vector()
    similarity_information = [[],[[],[]]]


    ave_or_tfidf = input("Choose the way to represent documents through embedded vectors: 1:average embedded vectors, 2: average tfidf embedded vectors: ")
    if ave_or_tfidf == 1:
        new_doc_vec, old_doc_vec = average_sentence_vec()
    elif ave_or_tfidf == 2:
        new_doc_vec, old_doc_vec = tfidf_sentence_vec(model, old_file_list, new_file_list)
    similarity_information[0] = new_doc_vec[0]


    measure = input("Input your distance calculation choice: 1: Cosine distance, 2: Jaccard distance, 3: Euclidean distance: ")
    if measure == 1:
       for i in xrange(len(new_doc_vec[1])):
           distance_matrix = []
           similarity_information[1][0].append(old_doc_vec[0])
           for j in xrange(len(old_doc_vec[1])):
               #distance_matrix[1][0].append(old_doc_vec[0])
               distance_matrix.append(scipy.spatial.distance.cosine(new_doc_vec[1][i], old_doc_vec[1][j]))
           similarity_information[1][1].append(distance_matrix)
    if measure == 2:
       for i in xrange(len(new_doc_vec[1])):
           distance_matrix = []
           similarity_information[1][0].append(old_doc_vec[0])
           for j in xrange(len(old_doc_vec[1])):
               distance_matrix.append(scipy.spatial.distance.jaccard(new_doc_vec[1][i], old_doc_vec[1][j]))
           similarity_information[1][1].append(distance_matrix)


    if measure == 3:
        for i in xrange(len(new_doc_vec[1])):
            distance_matrix = []
            similarity_information[1][0].append(old_doc_vec[0])
            for j in xrange(len(old_doc_vec[1])):
                distance_matrix.append(scipy.spatial.distance.euclidean(new_doc_vec[1][i], old_doc_vec[1][j]))
            similarity_information[1][1].append(distance_matrix)
    # print similarity_information[0]
    # print similarity_information[1][0]
    # print similarity_information[1][1]
    #print similarity_information
    return similarity_information


#select top k similar documents in the old data for every new document
def top_K():
    top_k_index = []
    similarity_information = document_distance()
    k = input("Input the number of similar document with the shortest distance: ")
    for i in xrange(len(similarity_information[0])):
        #create a list with k indexes which have smallest value
        tem = sorted(range(len(similarity_information[1][1][i])), key=lambda j: similarity_information[1][1][i][j])[:k]
        top_k_index.append(tem)
    return top_k_index, similarity_information

#Show the top k similar documents
def similar_documents():
    top_k_index, similarity_information = top_K()
    for i in xrange(len(top_k_index)):
        print ('\nBelow shows the top k similar documents of: {}'.format(similarity_information[0][i]))
        for j in top_k_index[i]:
            print similarity_information[1][0][i][j]
            print similarity_information[1][1][i][j]

def visualization():
    model, old_file_list, new_file_list = word_to_vector()
    ave_or_tfidf = input("Choose the way to represent documents through embedded vectors: 1:average embedded vectors, 2: average tfidf embedded vectors: ")
    if ave_or_tfidf == 1:
        new_doc_vec, old_doc_vec = average_sentence_vec()
    elif ave_or_tfidf == 2:
        new_doc_vec, old_doc_vec = tfidf_sentence_vec(model, old_file_list, new_file_list)


    data_list = pdist(old_doc_vec[1])
    data_link = linkage(data_list)
    dendrogram(data_link, truncate_mode='lastp',show_contracted=True, orientation= 'right', labels=old_doc_vec[0])
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.suptitle('Samples clustering', fontweight='bold', fontsize=14)
    plt.show()


def main():
    #word_to_vector()
    #average_sentence_vec()
    #tfidf_sentence_vec()
    #similar_documents()
    #visualization()
    #dimension_reduction()
    write_vec_to_csv()
if __name__ == "__main__":
    main()