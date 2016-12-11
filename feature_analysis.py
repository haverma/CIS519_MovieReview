from pre_processing import PreProcess
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import tree
from sklearn.metrics import accuracy_score
import math
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
max_features = 1000
feat_list = []
acc_list = []
while(max_features < 41000):
    count_vect = CountVectorizer(stop_words='english', max_features=max_features)
    X_train_fit = count_vect.fit(preprocess.training_data)
    X_train_counts = X_train_fit.transform(preprocess.training_data)
    tfIdfFit = TfidfTransformer(use_idf=True, norm='l2', sublinear_tf=True).fit(X_train_counts)
    preprocess.traintfIdf = tfIdfFit.transform(X_train_counts)
    X_test_counts = X_train_fit.transform(preprocess.test_data)
    preprocess.testtfIdf = tfIdfFit.transform(X_test_counts)

    nb_clf = MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)
    nb_clf.fit(preprocess.traintfIdf, preprocess.train_target)

    test_pred_nb = nb_clf.predict(preprocess.testtfIdf)

    nb_test_accuracy = metrics.accuracy_score(preprocess.test_target, test_pred_nb)
    feat_list.append(max_features)
    acc_list.append(nb_test_accuracy)
    max_features += 500
    print max_features
plt.plot(feat_list, acc_list)
plt.show()

