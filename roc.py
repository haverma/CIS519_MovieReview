from sklearn.grid_search import GridSearchCV
from pre_processing import PreProcess
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from pre_processing import PreProcess

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()
#preprocess.polarity_POS_features()

b_label = label_binarize(preprocess.test_target, np.unique(preprocess.test_target))

ada_clf = AdaBoostClassifier(MultinomialNB(alpha=0.1,fit_prior=True, class_prior=None),learning_rate=1, n_estimators=400)
ada_clf.fit(preprocess.traintfIdf, preprocess.train_target)

svm_clf = svm.SVC(kernel='linear', probability=True)
softmax_clf = LR(multi_class='ovr', C=4)
nb_clf = MultinomialNB(alpha=1, fit_prior=True, class_prior=None)
eclf1 = VotingClassifier(estimators=[('svm', svm_clf), ('softmax', softmax_clf), ('nb', nb_clf)], voting='soft', weights=[2,2,1])
eclf1.fit(preprocess.traintfIdf, preprocess.train_target)
# finding the training and test predictions probabilities


svm_clf1 = svm.SVC(kernel='linear', probability=True)
svm_clf1.fit(preprocess.traintfIdf, preprocess.train_target)

test_prob_ada = ada_clf.predict_proba(preprocess.testtfIdf)
test_prob_ens = eclf1.predict_proba(preprocess.testtfIdf)
test_prob_svm = svm_clf1.predict_proba(preprocess.testtfIdf)

ada_fpr = dict()
ada_tpr = dict()

ensemble_fpr = dict()
ensemble_tpr = dict()

svm_fpr = dict()
svm_tpr = dict()


no_classes = b_label.shape[1]
#Getting the FPR and TPR for all the categories
for i in range(no_classes):
        ada_fpr[i], ada_tpr[i], _ = roc_curve(b_label[:, i], test_prob_ada[:, i])
        ensemble_fpr[i], ensemble_tpr[i], _ = roc_curve(b_label[:, i], test_prob_ens[:, i])
        svm_fpr[i], svm_tpr[i], _ = roc_curve(b_label[:, i], test_prob_svm[:, i])
plt.figure()
target_names = ['bad', 'neutral','good']
# Plotting the ROC curve for all categories
for (key, value) in ada_fpr.items():
    category_name = target_names[key]
    plt.plot(ada_fpr[key], ada_tpr[key], label='AdaBoost Curve for ' + category_name, linestyle='-.')
    plt.plot(ensemble_fpr[key], ensemble_tpr[key], label='Ensemble Curve for ' + category_name, linestyle='dotted')
    plt.plot(svm_fpr[key], svm_tpr[key], label='SVM Curve for ' + category_name, linestyle='solid')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()
