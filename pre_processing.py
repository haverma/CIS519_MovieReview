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
from nltk.tokenize import  sent_tokenize
from nltk.tokenize import word_tokenize
from scipy import sparse
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import subprocess

class PreProcess:

    def __init__(self, train_loc, test_loc):
        self.test_loc = test_loc
        self.train_loc = train_loc
        self.training_data = []
        self.train_target = None
        self.test_data = []
        self.test_target = None


    def read_train_test_data(self):
        f_train = open(os.path.join(self.train_loc, 'reviews.txt'), 'r')
        f_test = open(os.path.join(self.test_loc, 'reviews.txt'), 'r')
        for line in f_train:
            self.training_data.append(line)
        for line in f_test:
            self.test_data.append(line)
        f_train = open(os.path.join(self.train_loc, 'labels.txt'), 'r')
        f_test = open(os.path.join(self.test_loc, 'labels.txt'), 'r')
        temp_labels_test = []
        temp_labels_train = []
        for line in f_train:
            temp_labels_train.append(int(line))
        for line in f_test:
            temp_labels_test.append(int(line))
        self.train_target = np.array(temp_labels_train)
        self.test_target = np.array(temp_labels_test)


    def getTfIdf(self):
        #count_vect = CountVectorizer(stop_words='english', max_features=11000)
        count_vect = CountVectorizer(stop_words=None, max_features=11000, ngram_range=(1,2))
        #count_vect = CountVectorizer(stop_words=None, max_features=4500, ngram_range=(1, 2))
        X_train_fit = count_vect.fit(self.training_data)
        X_train_counts = X_train_fit.transform(self.training_data)
        self.X_counts = X_train_counts
        tfIdfFit = TfidfTransformer(use_idf=True, norm='l2', sublinear_tf=True).fit(X_train_counts)
        self.traintfIdf = tfIdfFit.transform(X_train_counts)
        X_test_counts = X_train_fit.transform(self.test_data)
        self.testtfIdf = tfIdfFit.transform(X_test_counts)

    def add_pos_neg_feature(self):
        pos_set = set()
        neg_set = set()
        neg_words = open('data/neg_words.txt', 'r')
        pos_words = open('data/pos_words.txt', 'r')
        for line in pos_words:
            pos_set.add(line.strip())
        for line in neg_words:
            neg_set.add(line.strip())
        n, d = self.traintfIdf.shape
        train_sent = np.zeros((n,1))
        for idx, train_instance in enumerate(self.training_data):
            sentences = sent_tokenize(train_instance)
            positive_count = 0
            negative_count = 0
            for sent in sentences:
                words = word_tokenize(sent)
                for word in words:
                    if word in pos_set:
                        positive_count+=1
                    if word in neg_set:
                        negative_count+=1
            train_sent[idx, :] = [positive_count]
            print "training instance" + str(idx)
        #train_sent = (train_sent - np.mean(train_sent, axis=0))/np.std(train_sent, axis=0)
        #train_sent = 2*train_sent;
        #self.traintfIdf = np.concatenate((self.traintfIdf, train_sent), axis=1)
        self.traintfIdf = sparse.hstack((self.traintfIdf, train_sent))

        #To write the file as weka arff
        #train_sent = np.concatenate((train_sent, np.reshape(self.train_target,[4506,  -1])), axis=1)
        #np.savetxt("data/pos_neg_words.csv", train_sent, delimiter=',', fmt="%.2f %.2f %d")

        ##################################
        ntest, dtest = self.testtfIdf.shape
        test_sent = np.zeros((ntest, 1))
        for idx, test_instances in enumerate(self.test_data):
            sentences = sent_tokenize(test_instances)
            positive_count = 0
            negative_count = 0
            for sent in sentences:
                words = word_tokenize(sent)
                for word in words:
                    if word in pos_set:
                        positive_count += 1
                    if word in neg_set:
                        negative_count += 1
            test_sent[idx, :] = [positive_count]
            print "test instance" + str(idx)
        #test_sent = (test_sent - np.mean(test_sent, axis=0)) / np.std(test_sent, axis=0)
        #test_sent = 2*test_sent;
        #self.testtfIdf = np.concatenate((self.testtfIdf, test_sent), axis=1)
        self.testtfIdf = sparse.hstack((self.testtfIdf, test_sent))
        # np.savetxt("data/X_train.txt", self.traintfIdf.toarray(), delimiter=',')
        # np.savetxt("data/X_test.txt", self.testtfIdf.toarray(), delimiter=',')



    def add_pol_feature(self):
        words_file = open('data/polarity.txt', 'r')
        words_score = {}
        for line in words_file:
            tokens = line.split("\t")
            words_score[tokens[0].strip()] = int(tokens[1].strip())
        n, d = self.traintfIdf.shape
        train_sent = np.zeros((n,1))
        for idx, train_instance in enumerate(self.training_data):
            sentences = sent_tokenize(train_instance)
            score = 0
            for sent in sentences:
                words = word_tokenize(sent)
                for word in words:
                    if word in words_score:
                        score+=words_score[word]
            train_sent[idx, :] = [score]
        self.traintfIdf = sparse.hstack((self.traintfIdf, train_sent))
        train_sent = np.concatenate((train_sent, np.reshape(self.train_target,[4506,  -1])), axis=1)
        np.savetxt("data/polarity.csv", train_sent, delimiter=',', fmt="%.d %d")
        n, d = self.testtfIdf.shape
        test_sent = np.zeros((n, 1))
        for idx, test_instance in enumerate(self.test_data):
            sentences = sent_tokenize(test_instance)
            score = 0
            for sent in sentences:
                words = word_tokenize(sent)
                for word in words:
                    if word in words_score:
                        if idx==26:
                            print word
                        score+=words_score[word]
            test_sent[idx, :] = [score]
        self.testtfIdf = sparse.hstack((self.testtfIdf, test_sent))
        test_sent = np.concatenate((test_sent, np.reshape(self.test_target, [500, -1])), axis=1)
        np.savetxt("data/test_polarity.csv", test_sent, delimiter=',', fmt="%.d %d")

    def add_polarity_pos(self):
        pos_set = set()
        neg_set = set()
        neg_words = open('data/neg_words.txt', 'r')
        pos_words = open('data/pos_words.txt', 'r')
        for line in pos_words:
            pos_set.add(line.strip())
        for line in neg_words:
            neg_set.add(line.strip())
        # n, d = self.traintfIdf.shape
        # train_sent = np.zeros((n, 2))
        # for i, train_instance in enumerate(self.training_data):
        #     neg_words = []
        #     pos_words = []
        #     f = open('stanford-postagger-2015-12-09/temp.txt', 'w')
        #     f.write(train_instance)
        #     f.close()
        #     try:
        #         s = subprocess.check_output(['java', '-mx300m', '-classpath', 'stanford-postagger-2015-12-09/stanford-postagger.jar;stanford-postagger-2015-12-09/lib/*', 'edu.stanford.nlp.tagger.maxent.MaxentTagger', '-model', 'stanford-postagger-2015-12-09/models/english-left3words-distsim.tagger', '-textFile', 'stanford-postagger-2015-12-09/temp.txt'])
        #     except subprocess.CalledProcessError as e:
        #         print e.output
        #     s_list = s.split('\n')
        #     for line in s_list:
        #         words = line.split()
        #         for idx,w in enumerate(words):
        #             tokens = w.split('_')
        #             if tokens[1] == 'JJ' or tokens[1] == 'VBZ':
        #                 if "n't_RB" in words[:idx] or "not_RB" in words[:idx]:
        #                     if tokens[0] in pos_set or tokens[0] in neg_set:
        #                         neg_words.append(tokens[0])
        #                         continue
        #                 if tokens[0] in pos_set:
        #                     pos_words.append(tokens[0])
        #                 if tokens[0] in neg_set:
        #                     neg_words.append(tokens[0])
        #     train_sent[i,:] = [len(pos_words), len(neg_words)]
        #     print str(i) + " Training data"
        # train_sent = np.concatenate((train_sent,np.reshape(self.train_target,[4506,  -1]) ), axis=1)
        # np.savetxt("data/checking.csv", train_sent, delimiter=',', fmt="%d %d %d")

        n, d = self.testtfIdf.shape
        test_sent = np.zeros((n, 2))
        for i, test_instance in enumerate(self.test_data):
            neg_words = []
            pos_words = []
            f = open('stanford-postagger-2015-12-09/temp.txt', 'w')
            f.write(test_instance)
            f.close()
            try:
                s = subprocess.check_output(['java', '-mx300m', '-classpath',
                                             'stanford-postagger-2015-12-09/stanford-postagger.jar;stanford-postagger-2015-12-09/lib/*',
                                             'edu.stanford.nlp.tagger.maxent.MaxentTagger', '-model',
                                             'stanford-postagger-2015-12-09/models/english-left3words-distsim.tagger',
                                             '-textFile', 'stanford-postagger-2015-12-09/temp.txt'])
            except subprocess.CalledProcessError as e:
                print e.output
            s_list = s.split('\n')
            for line in s_list:
                words = line.split()
                for idx, w in enumerate(words):
                    tokens = w.split('_')
                    if tokens[1] == 'JJ' or tokens[1] == 'VBZ':
                        if "n't_RB" in words[:idx] or "not_RB" in words[:idx]:
                            if tokens[0] in pos_set or tokens[0] in neg_set:
                                neg_words.append(tokens[0])
                                continue
                        if tokens[0] in pos_set:
                            pos_words.append(tokens[0])
                        if tokens[0] in neg_set:
                            neg_words.append(tokens[0])
            test_sent[i, :] = [len(pos_words), len(neg_words)]
            print str(i) + " Test data"
        #train_sent = np.concatenate((train_sent, np.reshape(self.train_target, [4506, -1])), axis=1)
        np.savetxt("data/test_checking.csv", test_sent, delimiter=',', fmt="%d %d")


    def polarity_POS_features(self):
        filePathX = "data/checking.csv"
        file = open(filePathX, 'r')
        data = np.loadtxt(file, delimiter=' ')
        self.traintfIdf = sparse.hstack((self.traintfIdf, data[:,:-1]))
        filePathX = "data/test_checking.csv"
        filetest = open(filePathX, 'r')
        data = np.loadtxt(filetest, delimiter=' ')
        self.testtfIdf = sparse.hstack((self.testtfIdf, data))









if __name__=="__main__":
    preprocess = PreProcess("data/train", "data/test")
    preprocess.read_train_test_data()
    preprocess.getTfIdf()
    preprocess.training_data.extend(preprocess.test_data)
    combined_labels = np.concatenate((preprocess.train_target, preprocess.test_target), axis=0)
    combined_data = np.array(preprocess.training_data)
    idx = np.arange(np.size(combined_data))
    np.random.seed(13)
    np.random.shuffle(idx)
    combined_data = combined_data[idx]
    combined_labels = combined_labels[idx]
    combined_labels = combined_labels.astype(int)
    np.savetxt("train_data.txt", combined_data[:4506], fmt="%s", newline="")
    np.savetxt("test_data.txt", combined_data[4506:], fmt="%s", newline="")
    np.savetxt("train_label.txt", combined_labels[:4506], fmt="%d")
    np.savetxt("test_label.txt", combined_labels[4506:], fmt="%d")
