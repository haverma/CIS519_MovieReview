from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import time
from sklearn import svm
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

class PreProcess:

    def __init__(self, positive_dir, negative_dir):
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.training_data = []
        self.target = None

    def read_train_data(self):
        positive_files = os.listdir(self.positive_dir)
        for filename in positive_files:
            f = open(os.path.join(self.positive_dir, filename), 'r')
            content = f.read()
            self.training_data.append(content)
        negative_files = os.listdir(self.negative_dir)
        for filename in negative_files:
            f = open(os.path.join(self.negative_dir, filename), 'r')
            content = f.read()
            self.training_data.append(content)
        positive_labels_list = np.ones(len(positive_files))
        negative_labels_list = np.zeros(len(negative_files))
        self.target = np.concatenate((positive_labels_list, negative_labels_list), axis=0)

    def getTrainTfIdf(self):
        count_vect = CountVectorizer(stop_words='english')
        X_train_fit = count_vect.fit(self.training_data)
        X_train_counts = X_train_fit.transform(self.training_data)
        tfIdfFit = TfidfTransformer(use_idf=True, norm='l2', sublinear_tf=True).fit(X_train_counts)
        self.traintfIdf = tfIdfFit.transform(X_train_counts)


if __name__=="__main__":
    preprocess = PreProcess("data/pos", "data/neg")
    preprocess.read_train_data()
    preprocess.getTrainTestTfIdf()
    print "preprocessing done"
