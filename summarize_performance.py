import os
import sys
import cPickle as pkl
import numpy as np
from datetime import datetime
import re
import pandas as pd
from glob import glob
best_model = None

model_accuracy = []
accuracy_indexes = []
model_dirs = []
model_ids = []
model_names = []
model_directory = 'output'
time_midnight = datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d')
sub_dirs = sorted(os.listdir(model_directory))
users = []
cv_ids = []


def main(argv):
    cv_mode, today_mode = False, False
    if "cv" in argv:
        cv_mode = True
    if "today" in argv:
        today_mode = True

    for sub_dir in sub_dirs:
        validation_file = model_directory + '/' + sub_dir + '/training evaluations/evaluationvalidation_dict.pkl'
        cv_file = model_directory + '/' + sub_dir + '/training evaluations/evaluation_cv.pkl'

        if os.path.exists(validation_file) and cv_mode and 'cv' in sub_dir:
            if today_mode and not datetime.fromtimestamp(os.path.getctime(cv_file)) > time_midnight:
                pass
            else:
                model_id, model_name, net_size, n_in, classes, cv_id, user = \
                    re.search('_([0-9]*)_(\w*\W*)_(\W[0-9]*, [0-9]*\W)_(.*)_([0-9]*)_cv_([0-9]*)_([0-9])', sub_dir).groups()
                validations = np.asarray(pkl.load(open(validation_file, "rb")).values())
                model_accuracy.append(np.max(validations[:, 1]))
                accuracy_indexes.append(np.argmax(validations[:, 1]))
                model_dirs.append(sub_dir)
                model_ids.append(model_id)
                model_names.append(model_name)
                users.append(user)
                cv_ids.append(cv_id)

        elif os.path.isfile(validation_file) and not cv_mode:
            if today_mode and not datetime.fromtimestamp(os.path.getctime(validation_file)) > time_midnight:
                pass
            else:
                model_id, model_name, net_size, n_in, classes = \
                    re.search('_([0-9]*)_(\w*\W*)_(\W[0-9]*, [0-9]*\W)_(.*)_([0-9]*)', sub_dir).groups()
                validations = np.asarray(pkl.load(open(validation_file, "rb")).values())
                model_accuracy.append(np.max(validations[:, 1]))
                accuracy_indexes.append(np.argmax(validations[:, 1]))
                model_dirs.append(sub_dir)
                model_ids.append(model_id)
                model_names.append(model_name)

    if cv_mode:
        df = pd.DataFrame(data=np.transpose([model_ids, model_names, model_accuracy, model_dirs, users, cv_ids]),
                          columns=['id', 'name', 'accuracy', 'dir', 'user', 'cv'])
        df = df.convert_objects(convert_numeric=True)
        df = df.sort('accuracy', ascending=False)

        for model in df['name'].unique():
            print("\nModel: %s" % model)
            df_model = df.loc[df['name'] == model]
            for cv_id in df_model['cv'].unique():
                print("CV id: %d" % cv_id)
                for model_id, model_name, accuracy, model_dir, user, cv in df_model.loc[df['cv'] == cv_id].values[:10]:
                    print("%.4f\t%d\t%s" % (accuracy, user, model_dir))

    if len(model_dirs) > 0 and not cv_mode:
        df = pd.DataFrame(data=np.transpose([model_ids, model_names, model_accuracy, model_dirs]),
                          columns=['id', 'name', 'accuracy', 'dir'])
        df = df.convert_objects(convert_numeric=True)
        df = df.sort('accuracy', ascending=False)

        for model in df['name'].unique():
            print("\nModel: %s" % model)
            print("Accuracy\tModel")
            for model_id, model_name, accuracy, model_dir in df.loc[df['name'] == model].values[:10]:
                print("%.4f\t%s" % (accuracy, model_dir))
                # log_files = glob('./' + model_directory + '/' + model_dir + "/*.log")
                # best_model_log = os.path.join(model_directory, model_dir, log_files[0])
                # log = ""
                # with open(best_model_log, 'r') as f:
                #     for line in f:
                #         if 'TRAINING MODEL' in line:
                #             break
                #         log += line
                # print(log)

if __name__ == "__main__":
    main(sys.argv[1:])

