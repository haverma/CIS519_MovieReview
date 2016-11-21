import matplotlib.pyplot as plt
from pre_processing import PreProcess
import numpy as np

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()
target_labels = preprocess.train_target
positive_ind = target_labels == 2
negative_ind = target_labels == 0
neutral_ind = target_labels == 1

train_np = np.array(preprocess.training_data)

positive_examples = train_np[positive_ind]
negative_examples = train_np[negative_ind]
neutral_examples = train_np[neutral_ind]

print "kdjk"
