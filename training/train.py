import numpy as np
from utils import env_paths as paths
from .base import Train
import time


class TrainModel(Train):
    def __init__(self, model, output_freq=1, pickle_f_custom_freq=None,
                 f_custom_eval=None):
        super(TrainModel, self).__init__(model, pickle_f_custom_freq, f_custom_eval)
        self.output_freq = output_freq

    def train_model(self, f_train, train_args, f_test, test_args, f_validate, validation_args,
                    n_train_batches=600, n_valid_batches=1, n_test_batches=1, n_epochs=100, anneal=None):
        self.write_to_logger("### MODEL PARAMS ###")
        self.write_to_logger(self.model.model_info())
        self.write_to_logger("### TRAINING PARAMS ###")
        self.write_to_logger(
            "Train -> %s: %s" % (";".join(list(train_args['inputs'].keys())), str(list(train_args['inputs'].values()))))
        self.write_to_logger(
            "Test -> %s: %s" % (";".join(list(test_args['inputs'].keys())), str(list(test_args['inputs'].values()))))
        if anneal is not None:
            for t in anneal:
                key, freq, rate, min_val = t
                if isinstance(rate, int) or isinstance(rate, float):
                    self.write_to_logger(
                        "Anneal %s %0.4f after %i epochs with minimum value %f." % (key, rate, int(freq), min_val))

        self.write_to_logger("### TRAINING MODEL ###")

        if self.custom_eval_func is not None:
            self.custom_eval_func(self.model, paths.get_custom_eval_path(0, self.model.root_path))

        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            start_time = time.time()
            train_outputs = []
            for i in range(n_train_batches):
                train_output = f_train(i, *list(train_args['inputs'].values()))
                train_outputs.append(train_output)
            self.eval_train[epoch] = np.mean(np.array(train_outputs), axis=0)
            self.model.after_epoch()
            end_time = time.time() - start_time

            if anneal is not None:
                for t in anneal:
                    key, freq, rate, min_val = t
                    if isinstance(rate, int) or isinstance(rate, float):
                        new_val = train_args['inputs'][key] * rate
                    else:
                        new_val = rate(train_args['inputs'][key])
                    if new_val < min_val:
                        train_args['inputs'][key] = min_val
                    elif epoch % freq == 0:
                        train_args['inputs'][key] = new_val

            if epoch % self.output_freq == 0:
                if n_test_batches == 1:
                    self.eval_test[epoch] = f_test(*list(test_args['inputs'].values()))
                else:
                    test_outputs = []
                    for i in range(n_test_batches):
                        test_output = f_test(i, *list(test_args['inputs'].values()))
                        test_outputs.append(test_output)
                    self.eval_test[epoch] = np.mean(np.array(test_outputs), axis=0)

                if f_validate is not None:
                    if n_valid_batches == 1:
                        self.eval_validation[epoch] = f_validate(*list(validation_args['inputs'].values()))
                    else:
                        valid_outputs = []
                        for i in range(n_valid_batches):
                            valid_output = f_validate(i, *list(validation_args['inputs'].values()))
                            valid_outputs.append(valid_output)
                        self.eval_validation[epoch] = np.mean(np.array(valid_outputs), axis=0)
                else:
                    self.eval_validation[epoch] = [0.] * len(list(validation_args['outputs'].keys()))

                # Formatting the output string from the generic and the user-defined values.
                output_str = "epoch=%0" + str(len(str(n_epochs))) + "i; time=%0.2f;"
                output_str %= (epoch, end_time)

                def concatenate_output_str(out_str, d):
                    for k, v in zip(list(d.keys()), list(d.values())):
                        out_str += " %s=%s;" % (k, v)
                    return out_str

                output_str = concatenate_output_str(output_str, train_args['outputs'])
                output_str = concatenate_output_str(output_str, test_args['outputs'])
                output_str = concatenate_output_str(output_str, validation_args['outputs'])

                outputs = [float(o) for o in self.eval_train[epoch]]
                outputs += [float(o) for o in self.eval_test[epoch]]
                outputs += [float(o) for o in self.eval_validation[epoch]]

                output_str %= tuple(outputs)
                self.write_to_logger(output_str)

            if self.pickle_f_custom_freq is not None and epoch % self.pickle_f_custom_freq == 0:
                if self.custom_eval_func is not None:
                    self.custom_eval_func(self.model, paths.get_custom_eval_path(epoch, self.model.root_path))
                self.plot_eval(self.eval_train, list(train_args['outputs'].keys()), "_train")
                self.plot_eval(self.eval_test, list(test_args['outputs'].keys()), "_test")
                self.plot_eval(self.eval_validation, list(validation_args['outputs'].keys()), "_validation")
                self.dump_dicts()
                self.model.dump_model()
        if self.pickle_f_custom_freq is not None:
            self.model.dump_model()
