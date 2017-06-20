import csv
import os
import keras
import re
import magic
import imghdr
import PyPDF2
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import numpy as np
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scipy.cluster.hierarchy import ward, dendrogram,linkage
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE



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
    SYMBOLS = '{}()[]@#$%?!.,:;+-*/&|<>=~$1234567890"'
    stops = set(stopwords.words("english"))
    path = pdftotext()

    #get contect from text file
    for j in xrange(len(path)):
        data = [[], []]
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
                              document = re.sub(r'^https?:\/\/.*[\r\n]*', '', document, flags=re.MULTILINE)

                              document = " ".join(filter(lambda x: x[0] != '@', document.split()))

                              document = document.replace("\n", " ")
                              document = document.replace("\t", " ")
                              #print document
                              #print document

                              letters_only = re.sub("[^a-zA-Z]", " ", document)
                              letters_only = re.sub(" +", " ", letters_only)
                              print letters_only
                              print 'letters'
                              words = ' '.join([word for word in letters_only.split() if word not in stops])

                        data[0].append(filename)
                        data[1].append(words)
                    else:
                        continue
        if j==0:
            old_data = data
        if j==1:
            new_data = data
                    #print data[1]
                    #if file_type == 'application/pdf':
        print old_data
    return old_data, new_data


def method():
    old_data, new_data = get_text_content()

    x = input("Input your feture extraction choice, 1: TfidfVectorizer, 2: CounterVectorizer, 3: LSI : ")
    old_data_matrix = [[],[]]
    new_data_matrix = [[],[]]

    old_data_matrix[0] = np.array(old_data[0])
    new_data_matrix[0] = np.array(new_data[0])
    print("\nExtracting features...")
    if  x == 1:
        y = input("Input feature numbers, if not, please input None : ")

        t0 = time.time()
        vectorizer = TfidfVectorizer(max_features=y )
        #print vectorizer
        # get vector matrix
        old_data_matrix[1] = vectorizer.fit_transform(old_data[1])
        new_data_matrix[1] = vectorizer.transform(new_data[1])
        old_data_matrix[1] = old_data_matrix[1].toarray()
        new_data_matrix[1] = new_data_matrix[1].toarray()
        #print data_matrix
        # for w,i in vectorizer.vocabulary_.items():
        #     print w, vectorizer.idf_[i]

        #testing if new_docuemnts use same parameter (like total number of documents) as old_documents
        # for i in xrange(len(old_data_matrix[1])):
        #     print old_data_matrix[0][i]
        #     print old_data_matrix[1][i]
        #     print "**********************"
        # print "new"
        # for i in xrange(len(new_data_matrix[1])):
        #     print new_data_matrix[0][i]
        #     print new_data_matrix[1][i]
        #print vectorizer.vocabulary_.get('dog')
        print("  done in %.3fsec" % (time.time() - t0))


    elif x == 2:
        t0 = time.time()
        y = input("Input feature numbers, if not, please input None : ")
        vectorizer = CountVectorizer(max_features=y)
        # get vector matrix
        old_data_matrix[1] = vectorizer.fit_transform(old_data[1])
        new_data_matrix[1] = vectorizer.transform(new_data[1])
        old_data_matrix[1] = old_data_matrix[1].toarray()
        new_data_matrix[1] = new_data_matrix[1].toarray()
        print("  done in %.3fsec" % (time.time() - t0))
    elif x == 3:
        t0 = time.time()
        y = input("Input feature numbers > 100: ")
        vectorizer = TfidfVectorizer(max_features=y)
        # get vector matrix
        old_data_matrix[1] = vectorizer.fit_transform(old_data[1])
        new_data_matrix[1] = vectorizer.transform(new_data[1])
        print("\nPerforming dimensionality reduction using LSA...")
        svd = TruncatedSVD(n_components=300)
        vectorizer = make_pipeline(svd, Normalizer(copy=False))
        old_data_matrix[1] = vectorizer.fit_transform(old_data_matrix[1])
        new_data_matrix[1] = vectorizer.transform(new_data_matrix[1])
        #data_matrix[1] = data_matrix[1].toarray()
        #print data_matrix[1]
        print("  done in %.3fsec" % (time.time() - t0))
        explained_variance = svd.explained_variance_ratio_.sum()
        print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

        print ("\nNumber of documents and features: \n" )


    #print old_data_matrix[0]
    # print new_data_matrix
    print old_data_matrix[1][1]
    print "****************************"
    return old_data_matrix, new_data_matrix, vectorizer,x


def dimension_reduction():
    old_data_matrix, new_data_matrix, vectorizer, x = method()
    new_doc_vec_tsne = [[],[]]
    old_doc_vec_tsne = [[],[]]
    # ave_or_tfidf = input(
    #     "Choose the way to represent documents through embedded vectors: 1:average embedded vectors, 2: average tfidf embedded vectors: ")
    # if ave_or_tfidf == 1:
    #     new_doc_vec, old_doc_vec = average_sentence_vec()
    # elif ave_or_tfidf == 2:
    #     new_doc_vec, old_doc_vec = tfidf_sentence_vec()
    new_doc_vec_tsne[0] = new_data_matrix[0]
    old_doc_vec_tsne[0] = old_data_matrix[0]
    tsne = TSNE(n_components=2, init='pca', random_state=None, method='barnes_hut', n_iter=1000)
    new_doc_vec_tsne[1] = tsne.fit_transform(new_data_matrix[1])
    old_doc_vec_tsne[1] = tsne.fit_transform(old_data_matrix[1])
    #print new_doc_vec_tsne

    return new_doc_vec_tsne, old_doc_vec_tsne, x

def write_vec_to_csv():
    new_doc_vec, old_doc_vec, x = dimension_reduction()

    output = np.column_stack((new_doc_vec[0], new_doc_vec[1]))
    output = np.array(output)

    if x == 1:
        with open('TfidfVectorizer.csv', 'w') as f:
            fieldnames = ['Name', 'X', 'Y']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer = csv.writer(f)

            writer.writerows(output)
    elif x == 2:
        with open('CounterVectorizer.csv', 'w') as f:
            fieldnames = ['Name', 'X', 'Y']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer = csv.writer(f)

            writer.writerows(output)
    elif x == 3:
        with open('LSI.csv', 'w') as f:
            fieldnames = ['Name', 'X', 'Y']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer = csv.writer(f)

            writer.writerows(output)
#calculate document distance
def document_distance():
    old_data_matrix, new_data_matrix, vectorizer,x = method()
    similarity_information = [[],[[],[]]]

    similarity_information[0] = new_data_matrix[0]

    measure = input("Input your distance calculation choice: 1: Cosine distance, 2: Jaccard distance, 3: Euclidean distance: ")
    if measure == 1:
       for i in xrange(len(new_data_matrix[1])):
           distance_matrix = []
           similarity_information[1][0].append(old_data_matrix[0])
           for j in xrange(len(old_data_matrix[1])):
               #distance_matrix[1][0].append(old_data_matrix[0])
               distance_matrix.append(scipy.spatial.distance.cosine(new_data_matrix[1][i], old_data_matrix[1][j]))
           similarity_information[1][1].append(distance_matrix)
    if measure == 2:
       for i in xrange(len(new_data_matrix[1])):
           distance_matrix = []
           similarity_information[1][0].append(old_data_matrix[0])
           for j in xrange(len(old_data_matrix[1])):
               distance_matrix.append(scipy.spatial.distance.jaccard(new_data_matrix[1][i], old_data_matrix[1][j]))
           similarity_information[1][1].append(distance_matrix)


    if measure == 3:
        for i in xrange(len(new_data_matrix[1])):
            distance_matrix = []
            similarity_information[1][0].append(old_data_matrix[0])
            for j in xrange(len(old_data_matrix[1])):
                distance_matrix.append(scipy.spatial.distance.euclidean(new_data_matrix[1][i], old_data_matrix[1][j]))
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
    old_data_matrix, new_data_matrix, vectorizer,x = method()
    print old_data_matrix
    data_list = pdist(old_data_matrix[1])
    data_link = linkage(data_list)
    dendrogram(data_link, truncate_mode='lastp',show_contracted=True, orientation= 'right', labels=old_data_matrix[0])
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.suptitle('Samples clustering', fontweight='bold', fontsize=14)
    plt.show()


# def scatter_graph():
#     top_k_index, similarity_information = top_K()
#     distance = np.asarray(similarity_information[1][1])
#     mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
#     pos = mds.fit_transform(distance)
#     xs, ys = pos[:, 0], pos[:, 1]
#     for x, y, name in zip(xs, ys, similarity_information[0]):
#         color = 'orange' if "Austen" in name else 'skyblue'
#         plt.scatter(x, y, c=color)
#         plt.text(x, y, name)
#     plt.show()


# def hierarchical_document_clustering():
#     old_data_matrix, new_data_matrix, vectorizer = method()
#     dist = 1 - cosine_similarity(old_data_matrix[1])
#     linkage_matrix = ward(dist)  # define the linkage_matrix using ward clustering pre-computed distances
#
#     fig, ax = plt.subplots(figsize=(15, 20))  # set size
#     ax = dendrogram(linkage_matrix, orientation="right");
#
#     plt.tick_params( \
#         axis='x',  # changes apply to the x-axis
#         which='both',  # both major and minor ticks are affected
#         bottom='off',  # ticks along the bottom edge are off
#         top='off',  # ticks along the top edge are off
#         labelbottom='off')
#
#     plt.tight_layout()  # show plot with tight layout
#
#     # uncomment below to save figure
#     plt.savefig('ward_clusters.png', dpi=200)  # save figure as ward_clusters


def kmeans():
    old_data_matrix, new_data_matrix, vectorizer,x = method()
    #Compute k-means
    print ("\nInput the number of cluster centers, the number should not be larger than %d" % old_data_matrix[1].shape[0] )
    num_cluster_centers = input('Input the number of cluster centers: ')
    km = KMeans(n_clusters=num_cluster_centers, init='k-means++', max_iter=100, n_init=1)
    km.fit(old_data_matrix[1])
    clusters = km.labels_.tolist()

    #print top words per cluster
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num_cluster_centers):
        print "Cluster %d:" % i,
        for ind in order_centroids[i, :20]:
            print ' %s' % terms[ind],
        print
    #print vectorizer.vocabulary_.get('dog')
# def get_new_document_content():
#
# def get_data_matrix ():
#     data_matrix, vectorizer = method()



def main():
    visualization()
    #write_vec_to_csv()
    #scatter_graph()
    #similar_documents()
    #hierarchical_document_clustering()
    #kmeans()


if __name__ == "__main__":
    main()