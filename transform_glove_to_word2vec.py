import gensim
filename = 'glove.840B.300d.txt'
gensim.scripts.glove2word2vec.glove2word2vec(filename, 'word2vec.840B.300d.txt')