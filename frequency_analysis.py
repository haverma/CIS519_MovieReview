from pre_processing import PreProcess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()
x_counts = preprocess.X_counts

sum_counts = np.sum(x_counts, axis=0)

sorted_arr = np.sort(sum_counts)
sorted_arr = sorted_arr.tolist()
sorted_arr = sorted_arr[0]
tokens = range(40, 1040)
sorted_arr.reverse()
plt.plot(tokens, sorted_arr)
plt.show()
print "ss"