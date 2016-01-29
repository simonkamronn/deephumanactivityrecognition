import os
import sys
import cPickle as pkl
import numpy as np
from datetime import datetime
import re
import pandas as pd
import socket
import argparse

model_directory = 'output'
time_midnight = datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d')
sub_dirs = sorted(os.listdir(model_directory))
labels = ['accuracy', 'test error', 'index', 'dir', 'name', 'id', 'user', 'cv_id', 'train_acc']
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 300)
pd.set_option('display.column_space', 5)

hostname = socket.gethostname()
parser = argparse.ArgumentParser()
parser.add_argument('--cv', action='store_true')
parser.add_argument('--today', action='store_true')
parser.add_argument('--dataset', type=str, default='')


def main(argv):
    args = parser.parse_args(argv)

    today_only = args.today
    dataset = args.dataset

    if args.cv:
        cv_lookup(today_only, dataset)
    else:
        single_lookup(today_only, dataset)


def cv_lookup(today_only=False, dataset='', print_out=True):
    d = {label: [] for label in labels}
    for sub_dir in sub_dirs:
        validation_file = model_directory + '/' + sub_dir + '/training evaluations/evaluationvalidation_dict.pkl'
        test_file = model_directory + '/' + sub_dir + '/training evaluations/evaluationtest_dict.pkl'
        train_file = model_directory + '/' + sub_dir + '/training evaluations/evaluationtrain_dict.pkl'
        if os.path.exists(validation_file) and 'cv' in sub_dir and dataset in sub_dir:
            validations = np.asarray(pkl.load(open(validation_file, "rb")).values())
            if validations.shape[1] < 2:
                validations = np.asarray(pkl.load(open(test_file, "rb")).values())
            training_accuracy = np.asarray(pkl.load(open(train_file, "rb")).values())

            if today_only and not datetime.fromtimestamp(os.path.getctime(validation_file)) > time_midnight:
                pass
            elif len(validations.shape) > 1:
                model_id, model_name, net_size, n_in, classes, cv_id, user = \
                    re.search('_([0-9]*)_(\w*\W*)_(.*)_(.*)_([0-9]*)_cv_([0-9]*)_(.*)', sub_dir).groups()
                d['accuracy'].append(np.max(validations[:, 1]))
                d['index'].append(np.argmax(validations[:, 1]))
                d['test error'].append(validations[d['index'][-1], 0])
                d['dir'].append(sub_dir)
                d['id'].append(model_id)
                d['name'].append(model_name)
                d['user'].append(user)
                d['cv_id'].append(cv_id)

                if training_accuracy.shape[1] > 1:
                    d['train_acc'].append(np.max(training_accuracy[:, 1]))
                else:
                    d['train_acc'].append(0)
                # if ('HAPT10' in user) & (np.max(validations[:, 1]).astype(np.float) > 0.90):
                #     print('User: %s, CV id: %s, accuracy: %f' % ('HAPT10', cv_id, np.max(validations[:, 1])))

    df = pd.DataFrame.from_dict(d)
    df['accuracy'] = pd.to_numeric(df['accuracy'])
    df['train_acc'] = pd.to_numeric(df['train_acc'])
    df = df.sort_values(by='user', ascending=True)

    if print_out and len(df) > 0:
        df['cv_name'] = df['name'] + '_' + df['cv_id']
        df_pivot = df[['accuracy', 'cv_name', 'user']].pivot('user', 'cv_name')['accuracy']
        df_pivot = df_pivot.sort_index(ascending=True)
        print(df_pivot.describe().transpose().sort_values(by='mean', ascending=False))
        print(df_pivot)
        # print(df_pivot.transpose().describe().transpose())

        # print('Training accuracy')
        # df_pivot = df[['train_acc', 'cv_name', 'user']].pivot('user', 'cv_name')['train_acc']
        # df_pivot = df_pivot.sort_index(ascending=True)
        # print(df_pivot)

        path_to_file = 'evaluation/'+hostname+'_'+datetime.now().strftime('%Y-%m-%d')
        if not os.path.exists(path_to_file):
            os.mkdir(path_to_file)
        df.to_csv(path_to_file + '/cv_log.csv', sep=";", decimal=",")
        # for model in df['name'].unique():
        #     print("\nModel: %s" % model)
        #     df_model = df.loc[df['name'] == model]
        #     for cv_id in df_model['cv'].unique():
        #         print("CV id: %s" % cv_id)
        #         for model_id, model_name, accuracy, model_dir, user, cv in df_model.loc[df['cv'] == cv_id].values:
        #             print("%.4f\t%s\t%s" % (accuracy, user, model_dir))
        #         acc = df_model.loc[df['cv'] == cv_id]['accuracy']
        #         print('Mean: %f, std: %f, min: %f, max: %f\n' % (acc.mean(), acc.std(), acc.min(), acc.max()))

    return df


def single_lookup(today_only=False, dataset='', print_out=True):
    d = {label: [] for label in labels[:-3]}
    for sub_dir in sub_dirs:
        validation_file = model_directory + '/' + sub_dir + '/training evaluations/evaluationvalidation_dict.pkl'
        test_file = model_directory + '/' + sub_dir + '/training evaluations/evaluationtest_dict.pkl'
        if os.path.isfile(validation_file) and 'cv' not in sub_dir and dataset in sub_dir:
            if today_only and not datetime.fromtimestamp(os.path.getctime(validation_file)) > time_midnight:
                pass
            else:
                model_id, model_name, net_size, n_in, classes = \
                    re.search('_([0-9]*)_(\w*\W*)_(.*)_(.*)_([0-9]*)', sub_dir).groups()
                validations = np.asarray(pkl.load(open(validation_file, "rb")).values())
                if validations.shape[1] < 2:
                    validations = np.asarray(pkl.load(open(test_file, "rb")).values())
                d['accuracy'].append(np.max(validations[:, -1]))
                d['index'].append(np.argmax(validations[:, -1]))
                d['test error'].append(validations[d['index'][-1], 0])
                d['dir'].append(sub_dir)
                d['id'].append(model_id)
                d['name'].append(model_name)

    df = pd.DataFrame.from_dict(d)
    df['accuracy'] = pd.to_numeric(df['accuracy'])
    df = df.sort_values(by='accuracy', ascending=False)
    if print_out:
        for model in df['name'].unique():
            print("\nModel: %s" % model)
            print("Accuracy\tModel")
            for accuracy, model_dir in df[['accuracy', 'dir']].loc[df['name'] == model].values[:10]:
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
    return df

if __name__ == "__main__":
    main(sys.argv[1:])

