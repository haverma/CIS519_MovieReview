from sklearn.grid_search import GridSearchCV
from pre_processing import PreProcess
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
# from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import  VotingClassifier

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