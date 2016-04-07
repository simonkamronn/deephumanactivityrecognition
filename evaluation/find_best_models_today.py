import os
import cPickle as pkl
import numpy as np
from datetime import datetime

best_model = None

best_validation_accuracies = np.zeros(10, dtype=float)
best_accuracy_indexes = np.zeros(10, dtype=int)
best_model_dirs = np.empty(10, dtype=object)
model_directory = 'output'
time_midnight = datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d')
sub_dirs = sorted(os.listdir(model_directory))
for sub_dir in sub_dirs:
    validation_file = model_directory + '/' + sub_dir + '/training evaluations/evaluationvalidation_dict.pkl'
    if os.path.isfile(validation_file):
        if datetime.fromtimestamp(os.path.getctime(validation_file)) > time_midnight:
            validations = np.asarray(pkl.load(open(validation_file, "rb")).values())
            best_accuracy = np.max(validations[:, 1])
            if best_accuracy > np.min(best_validation_accuracies):
                best_validation_accuracies[0] = best_accuracy
                best_accuracy_indexes[0] = np.argmax(validations[:, 1])
                best_model_dirs[0] = sub_dir

                # Sort according to accuracy
                sort_idx = np.argsort(best_validation_accuracies)
                best_validation_accuracies = np.array(best_validation_accuracies)[sort_idx]
                best_accuracy_indexes = np.array(best_accuracy_indexes)[sort_idx]
                best_model_dirs = np.array(best_model_dirs)[sort_idx]


for best_model_dir, best_validation_accuracy, best_accuracy_index \
        in zip(best_model_dirs, best_validation_accuracies, best_accuracy_indexes):

    if best_validation_accuracy > 0.5:
        for file in os.listdir('./' + model_directory + '/' + best_model_dir):
            if file.endswith('log'):
                best_model_log = os.path.join(model_directory, best_model_dir, file)

        print("Best accuracy: %04f in epoch %d\nBest model: %s" %
              (best_validation_accuracy, best_accuracy_index, best_model_dir))

        log = ""
        with open(best_model_log, 'r') as f:
            for line in f:
                if 'TRAINING MODEL' in line:
                    break
                log += line
        print(log)

