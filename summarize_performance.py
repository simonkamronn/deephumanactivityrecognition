import os
import cPickle as pkl
import numpy as np
from datetime import datetime
import re
import pandas as pd
best_model = None

model_accuracy = []
accuracy_indexes = []
model_dirs = []
model_ids = []
model_names = []
model_directory = 'output'
time_midnight = datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d')
sub_dirs = sorted(os.listdir(model_directory))

for sub_dir in sub_dirs:
    validation_file = model_directory + '/' + sub_dir + '/training evaluations/evaluationvalidation_dict.pkl'
    if os.path.isfile(validation_file):
        model_id, model_name, net_size, n_in, classes = \
            re.search('_([0-9]*)_(\w*\W*)_(\W[0-9]*, [0-9]*\W)_(.*)_([0-9]*)', sub_dir).groups()
        validations = np.asarray(pkl.load(open(validation_file, "rb")).values())

        model_accuracy.append(np.max(validations[:, 1]))
        accuracy_indexes.append(np.argmax(validations[:, 1]))
        model_dirs.append(sub_dir)
        model_ids.append(model_id)
        model_names.append(model_name)

df = pd.DataFrame(data=np.transpose([model_ids, model_names, model_accuracy, model_dirs]),
                  columns=['id', 'name', 'accuracy', 'dir'])
df = df.convert_objects(convert_numeric=True)
df = df.sort('accuracy', ascending=False)

for model in df['name'].unique():
    print("\nModel: %s" % model)
    print("Accuracy\t\tModel")
    for model_id, model_name, accuracy, model_dir in df.loc[df['name'] == model].values[:10]:
        for file in os.listdir('./' + model_directory + '/' + model_dir):
            if file.endswith('log'):
                best_model_log = os.path.join(model_directory, model_dir, file)

        print("%04f\t%s" % (accuracy, model_dir))

        # log = ""
        # with open(best_model_log, 'r') as f:
        #     for line in f:
        #         if 'TRAINING MODEL' in line:
        #             break
        #         log += line
        # print(log)

