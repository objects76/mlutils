#!/usr/bin/env python3

#
# pip3 install tf-nightly-gpu 
# pip3 install tensorflow-datasets
# conda install -n tf2gpu scikit-learn  scikit-image matplotlib pandas tqdm opencv  matplotlib
# pip3 install -U tensorboard 
# On windows, path lengths greater than 260 characters may result in an error.
#  - https://docs.python.org/3/using/windows.html#removing-the-max-path-limitation 

_= r''' [code snippet]
import datetime
tick = f'{datetime.datetime.now():%Y%m%d-%H%M%S}'


import matplotlib.pyplot as plt
%matplotlib inline


'''

# %%

import os
import sys


# import sys, os
# sys.path.append(r'D:\github\mydir')
# from jjkim_util import *

# color-text output
black, red, green, yellow, blue, magenta, sky, white = range(30, 38)


def clrprint(fg, *args, **kwargs) -> None:
    '''
    fg: text color
    args: args for print()
    kwargs: bg=color_number + kwargs for print()
    '''
    if 'bg' in kwargs:
        print(f"\x1b[{fg};{kwargs['bg']+10}m", end='')
        del kwargs['bg']
    else:
        print(f"\x1b[{fg}m", end='')
    print(*args, end='\x1b[0m')
    print('', **kwargs)


def clrprint_dump():
    for fg in range(30, 38):
        for bg in range(30, 38):
            if fg == bg:
                continue
            clrprint(fg, f'clrprint({fg},{bg}, )', end='  ', bg=bg)
        print()


def clrcode_dump():
    for fg in range(30, 38):
        for bg in range(30, 38):
            if fg == bg:
                continue
            clrprint(
                fg, rf' \x1b[{fg};{bg+10}m TEXT_HERE \x1b[0m ', bg=bg, end='  ')
        print()


warn = (lambda *args, **kwargs: clrprint(yellow, *args, **kwargs))
err = (lambda *args, **kwargs: clrprint(red, *args, **kwargs))
info = (lambda *args, **kwargs: clrprint(green, *args, **kwargs))
debug = print



# %%
# python utils
def static_vars(**kwargs):
    '''
     @static_vars(label_info= ds_info.features["label"])
     '''
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

# @static_vars(label_info= ds_info.features["label"])


# %%


# %% [markdown]
# # tensorflow relateds

# %%
import logging
import tensorflow as tf

def tf_set_loglevel(lvl):
    '''
    logging.WARNING
    logging.ERROR
    '''
    import tensorflow
    tensorflow.get_logger().setLevel(lvl)

    
def disable_tfgpu():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # or even "-1"


def dump_gpu():
    print('tf version:', tf.__version__)
    # check Cuda & GPU
    print('is_built_with_cuda:', tf.test.is_built_with_cuda())
    print('is_built_with_gpu_support:', tf.test.is_built_with_gpu_support())

    from tensorflow.python.client import device_lib
    local_devices = device_lib.list_local_devices()
    for x in local_devices:
        print('>', x.device_type)
        print(
            f'   memory_limit: {x.memory_limit/(1024.0*1024.*1024.):.2f}GB, {x.memory_limit}B')
        if (x.physical_device_desc):
            name = x.physical_device_desc
            print(f'   {name}')
        #print('x:', x)
    return


def tf_growable_gmem():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for i in gpus:
        tf.config.experimental.set_memory_growth(i, True)
        print('set_memory_growth true:', i)


def tfconfig(grow_gmem=True):
    import tensorflow as tf
    print('tf version:', tf.__version__)
    # check Cuda & GPU
    print('is_built_with_cuda:', tf.test.is_built_with_cuda())
    print('is_built_with_gpu_support:', tf.test.is_built_with_gpu_support())

    from tensorflow.python.client import device_lib
    local_devices = device_lib.list_local_devices()
    for x in local_devices:
        print('>', x.device_type)
        print(
            f'   memory_limit: {x.memory_limit/(1024.0*1024.*1024.):.2f}GB, {x.memory_limit}B')
        if (x.physical_device_desc):
            name = x.physical_device_desc
            print(f'   {name}')

    if grow_gmem:
        gpus = tf.config.list_physical_devices('GPU')
        for i in gpus:
            tf.config.experimental.set_memory_growth(i, True)
            print('set_memory_growth true:', i)
    return

def check_cuda():
    import subprocess
    import glob
    import platform
    if  platform.system() == 'Windows':
        ret = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(ret.stdout)

        cudnn_folders = [
            r'C:/Program Files/NVIDIA/CUDNN',
            r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA',
        ]
        for folder in cudnn_folders:
            if not os.path.exists(folder):
                continue
            files = glob.glob(os.path.join(
                folder, '**/cudnn_version.h'), recursive=True)
            if len(files) > 0:
                os.startfile(files[0])
    else:
        raise RuntimeError("Not impl in", platform.system())

#
#
#
class flags:
    '''
     flags for notebook. alternative for absl.flags
    '''
    class AttrDict(dict):
        def __init__(self):
            self.__dict__ = self
        def __repr__(self) -> str:
            lines = []
            for k,v in flags.FLAGS.items():
                lines.append(f'{k:30s} = {v}')
            return '\n'.join(lines)

    FLAGS = AttrDict()

    @staticmethod
    def DEFINE_string(key, default, help = None):
        flags.FLAGS[key] = str(default)

    @staticmethod
    def DEFINE_integer(key, default, help = None):
        flags.FLAGS[key] = int(default)

    @staticmethod
    def DEFINE_float(key, default, help = None):
        flags.FLAGS[key] = float(default)

    @staticmethod
    def DEFINE_bool(key, default, help = None):
        flags.FLAGS[key] = bool(default)
    @staticmethod
    def DEFINE_object(key, default, help = None):
        flags.FLAGS[key] = default


# %%
# tf checkpoint
import tensorflow as tf

class TFCheckPoint:
  def __init__(self, model, path, step):
    self.model = model
    self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=model)
    self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=path, max_to_keep=None)
    self.save_step = step
    latest_ckpt = tf.train.latest_checkpoint(path)
    if latest_ckpt:
      self.ckpt.restore(latest_ckpt)
      print('global_step : {}, checkpoint is restored!'.format(int(self.ckpt.step)))

  def inc_step(self):
    self.ckpt.step.assign_add(1) # global step

  def save(self):
    if (self.ckpt.step % self.save_step) == 0:
      self.ckpt_manager.save(checkpoint_number=self.ckpt.step)
      print('global_step : {}, checkpoint is saved!'.format(int(self.ckpt.step)))
    # self.inc_step()


# %% [markdown]
# # Filesystem
# 

# %%
import glob
import os

def get_latest_file(folder_path, file_type=''):
    '''
    get latest .ext file
    '''
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(folder_path)

    files = glob.glob(os.path.join(folder_path, '*' + file_type))
    if len(files) == 0:
        raise FileNotFoundError(os.path.join(folder_path, '*' + file_type))

    return max(files, key=os.path.getctime)


def get_notebook_folder(): 
    return str(globals()["_dh"][0])

import random, pathlib 
def get_random_files(folder, ext='*', count=sys.maxsize):
    path_gen = pathlib.Path(folder).rglob(ext)
    files = [f for f in path_gen]
    if count < len(files):
        files = random.sample(files, count)
    return files
    

# %%



