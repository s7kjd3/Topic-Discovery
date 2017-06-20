import csv
import os
import math
import sklearn
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import tree, linear_model
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC, NuSVC
def data_preprocess():
    os.chdir(r'/home/flippped/thesis_project/vectors_with_removing_stopwords_punctuation/GRU')
    filename = 'reviews_30_50_Amazon_GRU_autoencoder_.csv'
    with open(filename, 'rU') as csvf:
        data = pd.read_csv(filename,skiprows=[0], header=None)
        #data = csv.reader(csvf)
        dl = data.values.tolist()
        print len(dl)
        np.random.shuffle(dl)
        labels = [x[0] for x in dl]
        for i in range(len(labels)):
            if labels[i] == 'cameras':
                labels[i]=0
            elif labels[i] == 'laptops':
                labels[i]=1
            elif labels[i] == 'mobilephone':
                labels[i]=2
            elif labels[i] == 'tablets':
                labels[i]=3
            elif labels[i] == 'TVs':
                labels[i]=4
            elif labels[i] == 'video_surveillance':
                labels[i]=5
        filename = np.array([x[1] for x in dl])
        vectors = np.array([x[2:] for x in dl])
        train_vectors = np.array(vectors[:(len(dl) / 2)])
        test_vectors = np.array(vectors[(len(dl) - len(dl) / 2):])
        train_labels = np.array(labels[:(len(dl) / 2)])
        test_labels = np.array(labels[(len(dl) - len(dl) / 2):])
        train_file = np.array(filename[:(len(dl) / 2)])
        test_file = np.array(filename[(len(dl) - len(dl) / 2):])
    print len(train_vectors)
    print len(test_vectors)
    return train_vectors, train_labels,train_file, test_vectors, test_labels, test_file
def evaluation():
    train_vectors, train_labels, train_file, test_vectors, test_labels, test_file = data_preprocess()
    print train_labels
    print train_file
    print test_labels
    print test_file


    print "*********************************SVM evaluation:"
    clf = NuSVC(kernel='rbf',probability=True)
    clf.fit(train_vectors, train_labels)
    print clf.score(test_vectors,test_labels)
    print "*********************************nearest neighbors evaluation:"
    neigh = KNeighborsClassifier(n_neighbors=20,algorithm='ball_tree')
    neigh.fit(train_vectors,train_labels)
    print neigh.score(test_vectors,test_labels)
    print "*********************************Nearest Centroid evaluation:"
    clf = NearestCentroid(metric='manhattan')
    clf.fit(train_vectors,train_labels)
    print clf.score(test_vectors,test_labels)
    print "*********************************Decision Tree evaluation:"
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_vectors, train_labels)
    print clf.score(test_vectors, test_labels)
    print "*********************************linear_model evaluation:"
    clf = linear_model.SGDClassifier()
    clf.fit(train_vectors, train_labels)
    print clf.score(test_vectors, test_labels)
    print "*********************************neural netowrk evaluation:"
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50, 50), random_state=1)
    clf.fit(train_vectors, train_labels)
    print clf.score(test_vectors, test_labels)


def main():
    evaluation()

if __name__ == "__main__":
 main()
