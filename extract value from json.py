import json
import os
path1 =['/home/flippped/Windows/Linux_Project/xiangmu/baseline/AmazonReviews/cameras',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/AmazonReviews/laptops',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/AmazonReviews/mobilephone',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/AmazonReviews/tablets',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/AmazonReviews/TVs',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/AmazonReviews/video_surveillance']

path2 =['/home/flippped/thesis_project/lstm_stacked_autoencoder/Amazon_30_50/camera_no',
           '/home/flippped/thesis_project/lstm_stacked_autoencoder/Amazon_30_50/laptop_no',
           '/home/flippped/thesis_project/lstm_stacked_autoencoder/Amazon_30_50/mobilephone_no',
           '/home/flippped/thesis_project/lstm_stacked_autoencoder/Amazon_30_50/tablets_no',
           '/home/flippped/thesis_project/lstm_stacked_autoencoder/Amazon_30_50/TVs_no',
           '/home/flippped/thesis_project/lstm_stacked_autoencoder/Amazon_30_50/video_surveillance_no']
labels_name = ['camera_no', 'laptop_no', 'mobilephone_no', 'tablets_no', 'TVs_no', 'video_surveillance_no']

for j in  xrange(len(path1)):
    no = 0
    count = 0
    #for i in path[j]:
    for file in os.listdir(path1[j]):



        with open(os.path.join(path1[j], file), 'r') as f:


            data = json.load(f)

            try:
                for field in data[u'Reviews']:
                    if no > 999:
                        break
                    #print len(field[u'Content'].split())
                    if (len(field[u'Content'].split()) < 50 and len(field[u'Content'].split()) > 30):
                        file = open(os.path.join(path2[j], labels_name[j] + "_" + str(no) + ".txt"), 'w+')

                        no+=1
                        count+=1
                        file.write(field[u'Content'])



                #print data[u'Reviews'][0][u'Title']
            except Exception:
                pass

    print count

# path =['/home/flippped/Desktop/xiangmu/baseline/Reviews/cameras',
#            '/home/flippped/Desktop/xiangmu/baseline/Reviews/laptops',
#            '/home/flippped/Desktop/xiangmu/baseline/Reviews/mobilephone',
#            '/home/flippped/Desktop/xiangmu/baseline/Reviews/tablets',
#            '/home/flippped/Desktop/xiangmu/baseline/Reviews/TVs',
#            '/home/flippped/Desktop/xiangmu/baseline/Reviews/video_surveillance']
# for i in xrange(len(path)):
#     t = 0
#     for j in path[i]:
#         t = t + 1
#     print t
# #
# file = 'B00A8RKTWO.json'
# with open(file,'r') as f:
#     data = json.load(f)
#     json_obj = json.dumps(data)
#     #print json_obj
#
#     try:
#         for i in data[u'Reviews']:
#             print i[u'Content']
#
#         print data[u'Reviews'][i][u'Content']
#     except Exception:
#         pass