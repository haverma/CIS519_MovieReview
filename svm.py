from sklearn.grid_search  import GridSearchCV
from sklearn.svm import SVC
from pre_processing import PreProcess
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

C_list = [5, 10, 20, 70, 150, 200, 400, 500, 600, 700, 900]
gamma_list = [0.0001, .001, .01, 1, 5, 10, 20, 30, 50, 70, 100, 120]
preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()
print C_list
print gamma_list

parameters = {'kernel':['linear', 'rbf'], 'C':C_list, 'gamma': gamma_list}
svm_clf = svm.SVC(kernel='linear',  probability=True)
# clf = GridSearchCV(svr, parameters)
# clf.fit(preprocess.traintfIdf, preprocess.target)
#print clf
# print "the best " + str(clf.best_estimator_)
# print "the Grid Score is " + str(clf.best_score_)
#print "The score on testing data is " + str(clf.score(X_test, y_test))


scores = cross_val_score(svm_clf, preprocess.traintfIdf, preprocess.train_target, cv=3)
print "the score is " + str(scores)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))