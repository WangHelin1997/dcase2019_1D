import torch
import os
import sys
import numpy as np
import soundfile
import librosa
import h5py
import math
import pandas as pd
from sklearn import metrics
import logging
import matplotlib.pyplot as plt
import config


def mixup_data(x1, x2, x3, x4, y, alpha=0.2):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        while True:
            lam = np.random.beta(alpha, alpha)
            if lam > 0.65 or lam < 0.35 :
                break
    else:
        lam = 1.
    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    mixed_x3 = lam * x3 + (1 - lam) * x3[index, :]
    mixed_x4 = lam * x4 + (1 - lam) * x4[index, :]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, mixed_x3, mixed_x4, y_a, y_b, lam

def mixup_criterion(class_criterion, pred, y_a, y_b, lam):
    return lam * class_criterion(pred, y_a) + (1 - lam) * class_criterion(pred, y_b)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging
    
    
def read_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs


def read_left_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = audio[0]

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs

def read_side_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = audio[0]-audio[1]

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs

def read_right_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = audio[1]

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs
    
def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]
    
    
def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x)
    std = np.std(x)

    return mean, std


def load_scalar(scalar_path):
    with h5py.File(scalar_path, 'r') as hf:
        mean = hf['mean']
        std = hf['std']

    scalar = {'mean': mean, 'std': std}
    return scalar
    
    
def scale(x, mean, std):
    return (x - mean) / std
    
    
def inverse_scale(x, mean, std):
    return x * std + mean
    
    
def get_subdir(subtask, data_type):
    if subtask == 'a':
        subdir = 'TAU-urban-acoustic-scenes-2019-{}'.format(data_type)
    elif subtask == 'b':
        subdir = 'TAU-urban-acoustic-scenes-2019-mobile-{}'.format(data_type)
    elif subtask == 'c':
        subdir = 'TAU-urban-acoustic-scenes-2019-openset-{}'.format(data_type)
    else:
        raise Exception('Incorrect argument!')
    
    return subdir
        
        
def read_metadata(metadata_path):
    '''Read metadata from a csv file. 
    
    Returns:
      meta_dict: dict of meta data, e.g.:
        {'audio_name': np.array(['a.wav', 'b.wav', ...]), 
         'scene_label': np.array(['airport', 'bus', ...]), 
         ...}
    '''
    df = pd.read_csv(metadata_path, sep='\t')
    
    meta_dict = {}
    
    meta_dict['audio_name'] = np.array(
        [name.split('/')[1] for name in df['filename'].tolist()])
    
    if 'scene_label' in df.keys():
        meta_dict['scene_label'] = np.array(df['scene_label'])
        
    if 'identifier' in df.keys():
        meta_dict['identifier'] = np.array(df['identifier'])
        
    if 'source_label' in df.keys():
        meta_dict['source_label'] = np.array(df['source_label'])
    
    return meta_dict
    
def sparse_to_categorical(x, n_out):
    x = x.astype(int)
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros((N,n_out))
    x_categ[np.arange(N), x] = 1
    return x_categ.reshape((shape)+(n_out,))
    
    
def get_sources(subtask):
    if subtask in ['a', 'c']:
        return ['a']
    elif subtask == 'b':
        return ['a', 'b', 'c']
    else:
        raise Exception('Incorrect argument!')
        
        
# def get_classes_num(subtask):
#     return len(config.labels) - 1   # Remove unknown label
