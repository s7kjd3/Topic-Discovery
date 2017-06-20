from baseline import *
y = input("Choose your way to load the data: 1: load your data form local, 2: load 20newsgroup 3:")
if y ==1:
    x = input("Choose your way to extract vectors: 1: baseline models, 2: word2vec mdoels")
    if x ==1:
        visualization()
    elif x ==2:
        write_vec_to_csv()
