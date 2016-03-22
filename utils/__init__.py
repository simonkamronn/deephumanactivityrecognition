from os import path, mkdir
import shutil
import inspect


def copy_script(script, model):
    # Copy scripts to output folder
    model_path = inspect.getsourcefile(model.__class__)
    script_path = path.realpath(script)

    # Create output folder
    base_path = model.get_root_path() + '/script/'
    mkdir(base_path)

    # Copy files to folder
    shutil.copy(script_path, base_path + 'conf_' + path.basename(script_path))
    shutil.copy(model_path, base_path + 'model_' + path.basename(model_path))
