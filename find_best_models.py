import os
import cPickle as pkl
import numpy as np

best_validation_accuracies = 0
best_model = None

best_validation_accuracies = [0]*10
best_accuracy_indexes = [0]*10
best_model_dirs = [0]*10
model_directory = 'output'
sub_dirs = sorted(os.listdir(model_directory))
for sub_dir in sub_dirs:
    validation_file = model_directory + '/' + sub_dir + '/training evaluations/evaluationvalidation_dict.pkl'
    if os.path.isfile(validation_file):
        validations = np.asarray(pkl.load(open(validation_file, "rb")).values())
        best_accuracy = np.max(validations[:, 1])
        if best_accuracy > np.min(best_validation_accuracies):
            best_validation_accuracies[0] = best_accuracy
            best_validation_accuracies.sort()

            best_accuracy_indexes[0] = np.argmax(validations[:, 1])
            best_accuracy_indexes.sort()

            best_model_dirs[0] = sub_dir
            best_model_dirs.sort()


for best_model_dir, best_validation_accuracy in zip(best_model_dirs, best_validation_accuracies):
    if best_validation_accuracy > 0:
        for file in os.listdir('./' + model_directory + '/' + best_model_dir):
            if file.endswith('log'):
                best_model_log = os.path.join(model_directory, best_model_dir, file)

        print("Best accuracy: %04f\nBest model: %s" % (best_validation_accuracy, best_model_dir))

        log = ""
        with open(best_model_log, 'r') as f:
            for line in f:
                if 'TRAINING MODEL' in line:
                    break
                log += line
        print(log)

