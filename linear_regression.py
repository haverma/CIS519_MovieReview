from sklearn.grid_search import GridSearchCV
from pre_processing import PreProcess
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import matplotlib.pyplot as plt
from operator import add
from sklearn import datasets, linear_model
from sklearn.multiclass import OneVsRestClassifier

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()
#preprocess.add_pos_neg_feature()

#preprocess.polarity_POS_features()

regr = linear_model.LinearRegression()
clf = OneVsRestClassifier(linear_model.LinearRegression())
scores = cross_val_score(clf, preprocess.traintfIdf, preprocess.train_target, cv=3)
print "the cross validated accuracy on training is " + str(scores)
print("the cross validated accuracy(standard deviation) on training is: %0.4f (+/- %0.4f)" % (
scores.mean(), scores.std() * 2))

clf.fit(preprocess.traintfIdf, preprocess.train_target)

train_pred_softmax = clf.predict(preprocess.traintfIdf)
test_pred_softmax = clf.predict(preprocess.testtfIdf)
# c = test_pred_softmax!=preprocess.test_target
# print np.where(c==True)


# c = preprocess.test_target != test_pred_softmax
# print "Number of incorrect predictions: " + str(sum(c))
# d = preprocess.test_target[c]
# e = sum(d==0)
# f = sum(d==1)
# g = sum(d==2)
# print "O's predicted wronly " + str(e)
# print "1's predicted wrongly " + str(f)
# print "2's predicted wrongly " + str(g)
# a0 = [0,0,0]
# a1 = [0,0,0]
# a2 = [0,0,0]
#
# for idx, value in enumerate(preprocess.test_target):
#     if value != test_pred_softmax[idx]:
#         if test_pred_softmax[idx] == 0:
#             if value == 1:
#                 a0[1]+=1
#             else:
#                 a0[2]+=1
#         if test_pred_softmax[idx] == 1:
#             if value == 0:
#                 a1[0]+=1
#             else:
#                 a1[2]+=1
#         if test_pred_softmax[idx] == 2:
#             if value == 0:
#                 a2[0]+=1
#             else:
#                 a2[1]+=1
#
# N =3
# print "The arrays are: "
# print a0
# print a1
# print a2
# # menMeans = (20, 35, 30, 35, 27)
# # womenMeans = (25, 32, 34, 20, 25)
#
#
# ind = np.arange(N)    # the x locations for the groups
# width = 0.35       # the width of the bars: can also be len(x) sequence
#
# p1 = plt.bar(ind, a0, width, color='r')
# p2 = plt.bar(ind, a1, width, color='y',
#              bottom=a0)
# p3 = plt.bar(ind, a2, width, color='g',
#              bottom=map(add, a0, a1))
#
# plt.ylabel('Mispredicted Instance Count')
# plt.title('Error in Prediction')
# plt.xticks(ind + width/2., ('0', '1', '2'))
# plt.yticks(np.arange(0, 110, 10))
# plt.legend((p1[0], p2[0], p3[0]), ('Predicted 0', 'Predicted 1', 'Predicted 2'))
#
# plt.show()

softmax_train_accuracy = metrics.accuracy_score(preprocess.train_target, train_pred_softmax)
softmax_test_accuracy = metrics.accuracy_score(preprocess.test_target, test_pred_softmax)
softmax_train_prec = metrics.precision_score(preprocess.train_target, train_pred_softmax, average="macro")
softmax_test_prec = metrics.precision_score(preprocess.test_target, test_pred_softmax, average="macro")
softmax_train_recall = metrics.recall_score(preprocess.train_target, train_pred_softmax,average="macro")
softmax_test_recall = metrics.recall_score(preprocess.test_target, test_pred_softmax, average="macro")

print "Scores\t\t" + "softmax Metrics"
print "Train Accuracy" + "\t"  + str(softmax_train_accuracy)
print "Test Accuracy" + "\t"  + str(softmax_test_accuracy)
print "Train Precision" + "\t" + str(softmax_train_prec)
print "Test Precision" + "\t" + str(softmax_test_prec)
print "Train Recall" + "\t" + str(softmax_train_recall)
print "Test Recall" + "\t\t" + str(softmax_test_recall)
