from sklearn.grid_search import GridSearchCV
from pre_processing import PreProcess
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()

ab_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
    n_estimators=500, learning_rate=1)

scores = cross_val_score(ab_clf, preprocess.traintfIdf, preprocess.train_target, cv=3)
print("the cross validated accuracy on training is " + str(scores))
print("the cross validated accuracy(standard deviation) on training is: %0.4f (+/- %0.4f)" % (
scores.mean(), scores.std() * 2))

ab_clf.fit(preprocess.traintfIdf, preprocess.train_target)
# finding the training and test predictions
train_pred_ab = ab_clf.predict(preprocess.traintfIdf)
test_pred_ab = ab_clf.predict(preprocess.testtfIdf)

ab_train_accuracy = metrics.accuracy_score(preprocess.train_target, train_pred_ab)
ab_test_accuracy = metrics.accuracy_score(preprocess.test_target, test_pred_ab)
ab_train_prec = metrics.precision_score(preprocess.train_target, train_pred_ab, average="macro")
ab_test_prec = metrics.precision_score(preprocess.test_target, test_pred_ab, average="macro")
ab_train_recall = metrics.recall_score(preprocess.train_target, train_pred_ab,average="macro")
ab_test_recall = metrics.recall_score(preprocess.test_target, test_pred_ab, average="macro")

print("Scores\t\t" + "AdaBoost (Decision Tree)")
print("Train Accuracy" + "\t"  + str(ab_train_accuracy))
print("Test Accuracy" + "\t"  + str(ab_test_accuracy))
print("Train Precision" + "\t" + str(ab_train_prec))
print("Test Precision" + "\t" + str(ab_test_prec))
print("Train Recall" + "\t" + str(ab_train_recall))
print("Test Recall" + "\t" + str(ab_test_recall))
