from os import path, mkdir
import shutil


def copy_script(scriptpath, rootpath):
    # Copy script to output folder
    filename = path.basename(scriptpath)
    basepath = rootpath + '/script/'
    mkdir(basepath)
    shutil.copy(scriptpath, basepath + filename)