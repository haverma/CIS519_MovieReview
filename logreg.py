from sklearn.grid_search import GridSearchCV
from pre_processing import PreProcess
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression as LR

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()

logreg_clf = LR()

scores = cross_val_score(logreg_clf, preprocess.traintfIdf, preprocess.train_target, cv=3)
print "the cross validated accuracy on training is " + str(scores)
print("the cross validated accuracy(standard deviation) on training is: %0.4f (+/- %0.4f)" % (
scores.mean(), scores.std() * 2))

logreg_clf.fit(preprocess.traintfIdf, preprocess.train_target)

train_pred_logreg = logreg_clf.predict(preprocess.traintfIdf)
test_pred_logreg = logreg_clf.predict(preprocess.testtfIdf)

logreg_train_accuracy = metrics.accuracy_score(preprocess.train_target, train_pred_logreg)
logreg_test_accuracy = metrics.accuracy_score(preprocess.test_target, test_pred_logreg)
logreg_train_prec = metrics.precision_score(preprocess.train_target, train_pred_logreg, average="macro")
logreg_test_prec = metrics.precision_score(preprocess.test_target, test_pred_logreg, average="macro")
logreg_train_recall = metrics.recall_score(preprocess.train_target, train_pred_logreg,average="macro")
logreg_test_recall = metrics.recall_score(preprocess.test_target, test_pred_logreg, average="macro")

print "Scores\t\t" + "LogReg Metrics"
print "Train Accuracy" + "\t"  + str(logreg_train_accuracy)
print "Test Accuracy" + "\t"  + str(logreg_test_accuracy)
print "Train Precision" + "\t" + str(logreg_train_prec)
print "Test Precision" + "\t" + str(logreg_test_prec)
print "Train Recall" + "\t" + str(logreg_train_recall)
print "Test Recall" + "\t\t" + str(logreg_test_recall)
