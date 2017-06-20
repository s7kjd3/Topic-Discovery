import nltk
import sys
from nltk.corpus import stopwords
import os
# from keras.preprocessing.text import Count
import sklearn.feature_extraction.text as text
from keras.preprocessing import text
stop_words = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
reload(sys)
sys.setdefaultencoding('utf8')
def get_data():
    targets = ['cameras','laptops','mobilephone','tablets','TVs','video_surveillance']
    path =['/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/cameras',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/laptops',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/mobilephone',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/tablets',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/TVs',
           '/home/flippped/Windows/Linux_Project/xiangmu/baseline/Reviews/video_surveillance']
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

def tt():
    data, target, filename = get_data()
    vectorizer = text.CountVectorizer(input=data, stop_words='english', min_df=20)
    print vectorizer
def main():
 tt()



if __name__ == "__main__":
 main()