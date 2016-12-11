import matplotlib.pyplot as plt
from pre_processing import PreProcess
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib_venn import venn3
from sklearn import svm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from pre_processing import PreProcess
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# ftest = open("data/svm_wrong.dat", 'r')
# svm = np.loadtxt(ftest, delimiter=',')
#
# ftest = open("data/softmax_wrong.dat", 'r')
# sm = np.loadtxt(ftest, delimiter=',')
#
#
# ftest = open("data/nb_wrong.dat", 'r')
# nb = np.loadtxt(ftest, delimiter=',')
#
# svm = set(svm)
# sm = set(sm)
# nb = set(nb)
# venn3([svm, sm, nb], ('SVM', 'Softmax', 'Naive Bayes'))
# plt.show()
preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()

svm_clf = svm.SVC(kernel='linear', probability=True)
softmax_clf = LR(multi_class='ovr', C=4)
nb_clf = MultinomialNB(alpha=1, fit_prior=True, class_prior=None)
eclf1 = VotingClassifier(estimators=[('svm', svm_clf), ('softmax', softmax_clf), ('nb', nb_clf)], voting='soft', weights=[2,2,1])

scores = cross_val_score(eclf1, preprocess.traintfIdf, preprocess.train_target, cv=3)
print "the cross validated accuracy on training is " + str(scores)
print("the cross validated accuracy(standard deviation) on training is: %0.4f (+/- %0.4f)" % (
scores.mean(), scores.std() * 2))

eclf1.fit(preprocess.traintfIdf, preprocess.train_target)
# finding the training and test predictions
train_pred_nb = eclf1.predict(preprocess.traintfIdf)
test_pred_nb = eclf1.predict(preprocess.testtfIdf)

# wrong_pred = np.where(preprocess.test_target!=test_pred_nb)
# np.savetxt("data/nb_wrong.dat", wrong_pred, delimiter=',', fmt="%d")

nb_train_accuracy = metrics.accuracy_score(preprocess.train_target, train_pred_nb)
nb_test_accuracy = metrics.accuracy_score(preprocess.test_target, test_pred_nb)
nb_train_prec = metrics.precision_score(preprocess.train_target, train_pred_nb, average="macro")
nb_test_prec = metrics.precision_score(preprocess.test_target, test_pred_nb, average="macro")
nb_train_recall = metrics.recall_score(preprocess.train_target, train_pred_nb,average="macro")
nb_test_recall = metrics.recall_score(preprocess.test_target, test_pred_nb, average="macro")

print("Scores\t\t" + "Ensemble")
print("Train Accuracy" + "\t"  + str(nb_train_accuracy))
print("Test Accuracy" + "\t"  + str(nb_test_accuracy))
print("Train Precision" + "\t" + str(nb_train_prec))
print("Test Precision" + "\t" + str(nb_test_prec))
print("Train Recall" + "\t" + str(nb_train_recall))
print("Test Recall" + "\t\t" + str(nb_test_recall))

