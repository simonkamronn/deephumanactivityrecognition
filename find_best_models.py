import os
import cPickle as pkl
import numpy as np


best_validation_accuracy = 0
best_model = None

model_directory = 'output'
sub_dirs = sorted(os.listdir(model_directory))
for sub_dir in sub_dirs:
    validation_file = model_directory + '/' + sub_dir + '/training evaluations/evaluationvalidation_dict.pkl'
    if os.path.isfile(validation_file):
        validations = np.asarray(pkl.load(open(validation_file, "rb")).values())
        best_accuracy = np.max(validations[:, 1])
        if best_accuracy > best_validation_accuracy:
            best_validation_accuracy = best_accuracy
            best_accuracy_index = np.argmax(validations[:, 1])
            best_model = sub_dir

print("Best accuracy: %04f\nBest model: %s" % (best_validation_accuracy, best_model))
