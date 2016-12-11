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
from sklearn.decomposition import PCA


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
preprocess.polarity_POS_features()
pca = PCA(n_components=2)

X_r = pca.fit(preprocess.traintfIdf.toarray()).transform(preprocess.traintfIdf.toarray())
print "The number of features " + str(pca.n_components_)
target_names = ['Bad', 'Neutral', 'Good']
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[preprocess.train_target == i, 0], X_r[preprocess.train_target == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Movie Reviews')
plt.show()