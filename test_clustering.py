import csv
import os
import math
import sklearn
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import numpy
from scipy.cluster.hierarchy import *
from sklearn.cluster import AgglomerativeClustering,hierarchical,DBSCAN,SpectralClustering,SpectralCoclustering, spectral_clustering
# from spherecluster import SphericalKMeans
# from spherecluster import VonMisesFisherMixture
from scipy import spatial



def cluster():
    os.chdir(r'/home/flippped/Windows/Linux_Project/xiangmu/baseline/vectors_reviews_10_50/')
    filename = 'reviews_50_word2vec_doc2vec_.csv'
    with open(filename, 'rU') as csvf:
        data = pd.read_csv(filename,skiprows=[0], header=None)
        #data = csv.reader(csvf)
        dl = data.values.tolist()

        camera_no = 0
        laptop_no = 0
        mobilephone_no = 0
        tablets_no = 0
        TVs_no = 0
        video_surveillance_no = 0
        labels = [x[0] for x in dl]
        for i in labels:
            if i == 'cameras':
                camera_no +=1
            elif i == 'laptops':
                laptop_no +=1
            elif i == 'mobilephone':
                mobilephone_no +=1
            elif i == 'tablets':
                tablets_no +=1
            elif i == 'TVs':
                TVs_no +=1
            elif i == 'video_surveillance':
                video_surveillance_no +=1

        labels_no = [camera_no,laptop_no,mobilephone_no,tablets_no,TVs_no,video_surveillance_no]
        print (labels_no)
        labels_name = ['cameras', 'laptops', 'mobilephone', 'tablets', 'TVs', 'video_surveillance']
        labels_target = []
        cluster_accuracy = []
        filenames = [x[1] for x in dl]
        a = np.array([x[2:] for x in dl])
        clust_centers = 6
        #Z = linkage(a, 'ward')
        for mm in range(2):
            # if mm == 0:
            #     dist = Pycluster.kl
            if mm == 1:
                model = DBSCAN(eps=1,min_samples=10,algorithm='brute',metric='cosine').fit(a)
            if mm == 0:
                model = DBSCAN(eps=0.5,min_samples=10,algorithm='kd_tree',metric='minkowski',p=2).fit(a)
            labels_xxx = [0,0,0,0,0,0]
            label_count = np.zeros((len(model.core_sample_indices_),6))
            cluster_name = ['DBSCAN_cosine','DBSCAN_SSN']
            print (model.core_sample_indices_)
            print (model.labels_)
            for j in range(len(model.core_sample_indices_)):
                for i in range(len(model.labels_)):
                    if model.labels_[i] == j:
                        if labels[i] == 'cameras':
                            label_count[j][0] +=1
                        if labels[i] == 'laptops':
                            label_count[j][1] +=1
                        if labels[i] == 'mobilephone':
                            label_count[j][2] +=1
                        if labels[i] == 'tablets':
                            label_count[j][3] +=1
                        if labels[i] == 'TVs':
                            label_count[j][4] +=1
                        if labels[i] == 'video_surveillance':
                            label_count[j][5] +=1
            print ('Below is the ' + cluster_name[mm] + 'clusters purity: ')
            summ = 0
            for i in label_count:
                summ += max(i)
            purity = summ / float(len(labels))
            print (purity)
            print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            print ('Below is the ' + cluster_name[mm] + 'clusters entropy: ')
            single_cluster_entropy = []
            for i in label_count:
                tmp = 0
                for j in i:
                    tmp += j/float(sum(i)) * np.log2(j/float(sum(i)))
                single_cluster_entropy.append(tmp)
            total_entropy = 0
            for i in range(len(model.core_sample_indices_)):
                total_entropy += (single_cluster_entropy[i] * (sum(label_count[i]) / float(len(labels))))
            print (total_entropy)
            print ('*************************************************************************************************')







def main():
 cluster()

if __name__ == "__main__":
 main()


