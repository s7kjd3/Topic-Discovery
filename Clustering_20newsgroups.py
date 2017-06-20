import csv
import os
import sklearn
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import numpy
from scipy import spatial

def cluster():
    os.chdir(r'/home/flippped/Windows/Linux_Project/xiangmu/baseline/vectors_20groupsnews/')
    filename = '20newsgroups_doc2vec.csv'
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
        print labels_no
        labels_name = ['cameras', 'laptops', 'mobilephone', 'tablets', 'TVs', 'video_surveillance']
        labels_target = []
        cluster_accuracy = []
        filenames = [x[1] for x in dl]
        a = np.array([x[2:] for x in dl])
        clust_centers = 6
        file = open('20newsgroups_doc2vec_' + "_" + 'cluster_accuracy' + ".txt", 'w+')
        for mm in xrange(2):
        #y = input('Please clustering method: 1:k-means, 2:spherical k-means, 3:movMF-soft, 4:movMF-hard: ')
            if mm == 0:
                model = KMeans(n_clusters=clust_centers,max_iter=10000000).fit(a)
            # if mm == 1:
            #     model = SphericalKMeans(n_clusters=clust_centers, max_iter=10000000).fit(a)
            # if i == 2:
            #     model = VonMisesFisherMixture(n_clusters=clust_centers,  posterior_type='soft',max_iter=10000000).fit(a)
            # if i == 3:
            #     model = VonMisesFisherMixture(n_clusters=clust_centers,  posterior_type='hard',max_iter=10000000).fit(a)

            #print model.cluster_centers_
            #To determine corresponding lables for model.labels_
            # s = [0,camera_no,laptop_no,mobilephone_no,tablets_no,TVs_no]
            # ss = [s[0],s[0]+s[1],s[0]+s[1]+s[2],s[0]+s[1]+s[2]+s[3],s[0]+s[1]+s[2]+s[3]+s[4],s[0]+s[1]+s[2]+s[3]+s[4]+s[5]]
            # print ss
            labels_xxx = [0,0,0,0,0,0]
            for i in model.labels_:
                if i == 0:
                    labels_xxx[0]+=1
                if i == 1:
                    labels_xxx[1]+=1
                if i == 2:
                    labels_xxx[2]+=1
                if i == 3:
                    labels_xxx[3]+=1
                if i == 4:
                    labels_xxx[4]+=1
                if i == 5:
                    labels_xxx[5]+=1

            print "labels' number:"
            print labels_xxx
            for i in xrange(len(labels_name)):
                label_tmp = [0, 0, 0, 0, 0, 0]
                for j in xrange(len(model.labels_)):
                    if model.labels_[j] == i:
                        if labels[j] == labels_name[0]:
                            label_tmp[0] += 1
                        if labels[j] == labels_name[1]:
                            label_tmp[1] += 1
                        if labels[j] == labels_name[2]:
                            label_tmp[2] += 1
                        if labels[j] == labels_name[3]:
                            label_tmp[3] += 1
                        if labels[j] == labels_name[4]:
                            label_tmp[4] += 1
                        if labels[j] == labels_name[5]:
                            label_tmp[5] += 1

                labels_target.append(labels_name[label_tmp.index(max(label_tmp))])

                print max(label_tmp)
                print ('The cluster ', i ,' is consist of ', label_tmp, ' with label names are ', labels_name)
                file.write('The cluster ' + str(i) + ' is consist of ' +  str(label_tmp) + ' with label names are ' + str(labels_name) + str('\n'))

                print ('Most topics contained are about: ',labels_name[label_tmp.index(max(label_tmp))])
                file.write('Most topics contained are about: ' + labels_name[label_tmp.index(max(label_tmp))]+ str('\n'))

                #To calculate the accuracy for every cluster
                cluster_accuracy.append(float(max(label_tmp))/labels_xxx[i])
                print ('The accuracy for cluster ', i , ' is ', (float(max(label_tmp))/labels_xxx[i]))
                file.write('The accuracy for cluster ' + str(i) + ' is ' + str((float(max(label_tmp))/labels_xxx[i])) + str('\n'))

            file.write('The average accuracy is: ' +  str(numpy.mean(cluster_accuracy))+ str('\n'))

        print labels_target
        # print model.labels_
        # print model
        print model.inertia_
       # points_distance(a,labels)


def points_distance(a,labels):
    rate_of_same_label = []
    k = input('Input number of top k: ')
    for i in xrange(len(labels)):

        distance_matrix = []
        for j in xrange(len(a)):
            distance_matrix.append(spatial.distance.cosine(a[i], a[j]))

        top_k_distance_index = numpy.argsort(distance_matrix)[::1][:k]

        for m in top_k_distance_index:
            c=0
            if labels[m] == labels[i]:
                c+=1
            else:
                continue
        rate_of_same_label.append(float(c-1)/(k-1))
    print ('Rate of nearest ',k,' data point: ',rate_of_same_label, '\n')
    file.write('Rate of nearest ' + str(k) +' data point: ' + (rate_of_same_label)+ str('\n'))

    average = numpy.mean(rate_of_same_label)
    print ('The average is: ', average, '\n')
    file.write('The average is: ' + str(average)+ str('\n'))



def main():
 cluster()



if __name__ == "__main__":
 main()
