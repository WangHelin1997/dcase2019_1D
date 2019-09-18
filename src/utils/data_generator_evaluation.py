import numpy as np
import h5py
import csv
import time
import logging
import os
import glob
import matplotlib.pyplot as plt
import logging

from utilities import scale, read_metadata, sparse_to_categorical
import config


class DataGenerator(object):
    
    def __init__(self, feature_hdf5_path, feature_hdf5_path_left, feature_hdf5_path_right, feature_hdf5_path_side,
                 feature_hdf5_path_harmonic, feature_hdf5_path_percussive, evaluation_csv,
                 scalar, scalar_left, scalar_right, scalar_side, scalar_harmonic, scalar_percussive, batch_size, seed=1234):
        '''Data generator for training and validation. 
        
        Args:
          feature_hdf5_path: string, path of hdf5 feature file
          train_csv: string, path of train csv file
          validate_csv: string, path of validate csv file
          scalar: object, containing mean and std value
          batch_size: int
          seed: int, random seed
        '''

        self.scalar = scalar
        self.scalar_left = scalar_left
        self.scalar_right = scalar_right
        self.scalar_side = scalar_side
        self.scalar_harmonic = scalar_harmonic
        self.scalar_percussive = scalar_percussive
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        
        # self.classes_num = classes_num
        self.in_domain_classes_num = len(config.labels) - 1
        self.all_classes_num = len(config.labels)
        self.lb_to_idx = config.lb_to_idx
        self.idx_to_lb = config.idx_to_lb
        
        # Load training data
        load_time = time.time()
        
        self.data_dict = self.load_hdf5(feature_hdf5_path)
        self.data_dict_left = self.load_hdf5_left(feature_hdf5_path_left)
        self.data_dict_right = self.load_hdf5_right(feature_hdf5_path_right)
        self.data_dict_side = self.load_hdf5_side(feature_hdf5_path_side)
        self.data_dict_harmonic = self.load_hdf5_harmonic(feature_hdf5_path_harmonic)
        self.data_dict_percussive = self.load_hdf5_percussive(feature_hdf5_path_percussive)
        evaluation_meta = read_metadata(evaluation_csv)

        self.evaluation_audio_indexes = self.get_audio_indexes(
            evaluation_meta, self.data_dict, 'train')
        
        logging.info('Load data time: {:.3f} s'.format(time.time() - load_time))
        logging.info('Evaluation audio num: {}'.format(len(self.evaluation_audio_indexes)))
        
    def load_hdf5(self, hdf5_path):
        '''Load hdf5 file. 
        
        Returns:
          data_dict: dict of data, e.g.:
            {'audio_name': np.array(['a.wav', 'b.wav', ...]), 
             'feature': (audios_num, frames_num, mel_bins)
             'target': (audios_num,), 
             ...}
        '''
        data_dict = {}
        
        with h5py.File(hdf5_path, 'r') as hf:
            data_dict['audio_name'] = np.array(
                [audio_name.decode() for audio_name in hf['audio_name'][:]])

            data_dict['feature'] = hf['feature'][:].astype(np.float32)
            
            if 'scene_label' in hf.keys():
                data_dict['target'] = np.array(
                    [self.lb_to_idx[scene_label.decode()] \
                        for scene_label in hf['scene_label'][:]])
                
            if 'identifier' in hf.keys():
                data_dict['identifier'] = np.array(
                    [identifier.decode() for identifier in hf['identifier'][:]])
                
            if 'source_label' in hf.keys():
                data_dict['source_label'] = np.array(
                    [source_label.decode() \
                        for source_label in hf['source_label'][:]])
            
        return data_dict

    def load_hdf5_left(self, hdf5_path):
        '''Load hdf5 file.

        Returns:
          data_dict: dict of data, e.g.:
            {'audio_name': np.array(['a.wav', 'b.wav', ...]),
             'feature': (audios_num, frames_num, mel_bins)
             'target': (audios_num,),
             ...}
        '''
        data_dict = {}

        with h5py.File(hdf5_path, 'r') as hf:
            data_dict['audio_name'] = np.array(
                [audio_name.decode() for audio_name in hf['audio_name'][:]])

            data_dict['feature_left'] = hf['feature_left'][:].astype(np.float32)

            if 'scene_label' in hf.keys():
                data_dict['target'] = np.array(
                    [self.lb_to_idx[scene_label.decode()] \
                     for scene_label in hf['scene_label'][:]])

            if 'identifier' in hf.keys():
                data_dict['identifier'] = np.array(
                    [identifier.decode() for identifier in hf['identifier'][:]])

            if 'source_label' in hf.keys():
                data_dict['source_label'] = np.array(
                    [source_label.decode() \
                     for source_label in hf['source_label'][:]])

        return data_dict

    def load_hdf5_right(self, hdf5_path):
        '''Load hdf5 file.

        Returns:
          data_dict: dict of data, e.g.:
            {'audio_name': np.array(['a.wav', 'b.wav', ...]),
             'feature': (audios_num, frames_num, mel_bins)
             'target': (audios_num,),
             ...}
        '''
        data_dict = {}

        with h5py.File(hdf5_path, 'r') as hf:
            data_dict['audio_name'] = np.array(
                [audio_name.decode() for audio_name in hf['audio_name'][:]])

            data_dict['feature_right'] = hf['feature_right'][:].astype(np.float32)

            if 'scene_label' in hf.keys():
                data_dict['target'] = np.array(
                    [self.lb_to_idx[scene_label.decode()] \
                     for scene_label in hf['scene_label'][:]])

            if 'identifier' in hf.keys():
                data_dict['identifier'] = np.array(
                    [identifier.decode() for identifier in hf['identifier'][:]])

            if 'source_label' in hf.keys():
                data_dict['source_label'] = np.array(
                    [source_label.decode() \
                     for source_label in hf['source_label'][:]])

        return data_dict

    def load_hdf5_side(self, hdf5_path):
        '''Load hdf5 file.

        Returns:
          data_dict: dict of data, e.g.:
            {'audio_name': np.array(['a.wav', 'b.wav', ...]),
             'feature': (audios_num, frames_num, mel_bins)
             'target': (audios_num,),
             ...}
        '''
        data_dict = {}

        with h5py.File(hdf5_path, 'r') as hf:
            data_dict['audio_name'] = np.array(
                [audio_name.decode() for audio_name in hf['audio_name'][:]])

            data_dict['feature_side'] = hf['feature_side'][:].astype(np.float32)

            if 'scene_label' in hf.keys():
                data_dict['target'] = np.array(
                    [self.lb_to_idx[scene_label.decode()] \
                     for scene_label in hf['scene_label'][:]])

            if 'identifier' in hf.keys():
                data_dict['identifier'] = np.array(
                    [identifier.decode() for identifier in hf['identifier'][:]])

            if 'source_label' in hf.keys():
                data_dict['source_label'] = np.array(
                    [source_label.decode() \
                     for source_label in hf['source_label'][:]])

        return data_dict

    def load_hdf5_harmonic(self, hdf5_path):
        '''Load hdf5 file.

        Returns:
          data_dict: dict of data, e.g.:
            {'audio_name': np.array(['a.wav', 'b.wav', ...]),
             'feature': (audios_num, frames_num, mel_bins)
             'target': (audios_num,),
             ...}
        '''
        data_dict = {}

        with h5py.File(hdf5_path, 'r') as hf:
            data_dict['audio_name'] = np.array(
                [audio_name.decode() for audio_name in hf['audio_name'][:]])

            data_dict['feature_harmonic'] = hf['feature_harmonic'][:].astype(np.float32)

            if 'scene_label' in hf.keys():
                data_dict['target'] = np.array(
                    [self.lb_to_idx[scene_label.decode()] \
                     for scene_label in hf['scene_label'][:]])

            if 'identifier' in hf.keys():
                data_dict['identifier'] = np.array(
                    [identifier.decode() for identifier in hf['identifier'][:]])

            if 'source_label' in hf.keys():
                data_dict['source_label'] = np.array(
                    [source_label.decode() \
                     for source_label in hf['source_label'][:]])

        return data_dict

    def load_hdf5_percussive(self, hdf5_path):
        '''Load hdf5 file.

        Returns:
          data_dict: dict of data, e.g.:
            {'audio_name': np.array(['a.wav', 'b.wav', ...]),
             'feature': (audios_num, frames_num, mel_bins)
             'target': (audios_num,),
             ...}
        '''
        data_dict = {}

        with h5py.File(hdf5_path, 'r') as hf:
            data_dict['audio_name'] = np.array(
                [audio_name.decode() for audio_name in hf['audio_name'][:]])

            data_dict['feature_percussive'] = hf['feature_percussive'][:].astype(np.float32)

            if 'scene_label' in hf.keys():
                data_dict['target'] = np.array(
                    [self.lb_to_idx[scene_label.decode()] \
                     for scene_label in hf['scene_label'][:]])

            if 'identifier' in hf.keys():
                data_dict['identifier'] = np.array(
                    [identifier.decode() for identifier in hf['identifier'][:]])

            if 'source_label' in hf.keys():
                data_dict['source_label'] = np.array(
                    [source_label.decode() \
                     for source_label in hf['source_label'][:]])

        return data_dict

    def get_audio_indexes(self, meta_data, data_dict, data_type):
        '''Get train or validate indexes. 
        '''
        audio_indexes = []
        
        for name in meta_data['audio_name']:
            loct = np.argwhere(data_dict['audio_name'] == name)
            
            if len(loct) > 0:
                index = loct[0, 0]
                audio_indexes.append(index)
            
        return np.array(audio_indexes)
        
    def generate_train(self):
        '''Generate mini-batch data for training. 
        
        Returns:
          batch_data_dict: dict containing audio_name, feature and target
        '''
        batch_size = self.batch_size
        audio_indexes = np.array(self.evaluation_audio_indexes)
        self.random_state.shuffle(audio_indexes)
        pointer = 0

        while True:
            # Reset pointer
            if pointer >= len(audio_indexes):
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch audio_indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            batch_data_dict = {}
            batch_data_dict_left = {}
            batch_data_dict_right = {}
            batch_data_dict_side = {}
            batch_data_dict_harmonic = {}
            batch_data_dict_percussive = {}

            batch_data_dict['audio_name'] = \
                self.data_dict['audio_name'][batch_audio_indexes]
            batch_data_dict_left['audio_name'] = \
                self.data_dict_left['audio_name'][batch_audio_indexes]
            batch_data_dict_right['audio_name'] = \
                self.data_dict_right['audio_name'][batch_audio_indexes]
            batch_data_dict_side['audio_name'] = \
                self.data_dict_side['audio_name'][batch_audio_indexes]
            batch_data_dict_harmonic['audio_name'] = \
                self.data_dict_harmonic['audio_name'][batch_audio_indexes]
            batch_data_dict_percussive['audio_name'] = \
                self.data_dict_percussive['audio_name'][batch_audio_indexes]

            batch_feature = self.data_dict['feature'][batch_audio_indexes]
            batch_feature = self.transform(batch_feature)
            batch_data_dict['feature'] = batch_feature

            batch_feature_left = self.data_dict_left['feature_left'][batch_audio_indexes]
            batch_feature_left = self.transform_left(batch_feature_left)
            batch_data_dict_left['feature_left'] = batch_feature_left

            batch_feature_right = self.data_dict_right['feature_right'][batch_audio_indexes]
            batch_feature_right = self.transform_right(batch_feature_right)
            batch_data_dict_right['feature_right'] = batch_feature_right

            batch_feature_side = self.data_dict_side['feature_side'][batch_audio_indexes]
            batch_feature_side = self.transform_side(batch_feature_side)
            batch_data_dict_side['feature_side'] = batch_feature_side

            batch_feature_harmonic = self.data_dict_harmonic['feature_harmonic'][batch_audio_indexes]
            batch_feature_harmonic = self.transform_harmonic(batch_feature_harmonic)
            batch_data_dict_harmonic['feature_harmonic'] = batch_feature_harmonic

            batch_feature_percussive = self.data_dict_percussive['feature_percussive'][batch_audio_indexes]
            batch_feature_percussive = self.transform_percussive(batch_feature_percussive)
            batch_data_dict_percussive['feature_percussive'] = batch_feature_percussive
            
            sparse_target = self.data_dict['target'][batch_audio_indexes]
            batch_data_dict['target'] = sparse_to_categorical(
                sparse_target, self.in_domain_classes_num)
            
            yield batch_data_dict, batch_data_dict_left, batch_data_dict_right, batch_data_dict_side, batch_data_dict_harmonic, batch_data_dict_percussive
            
    def get_source_indexes(self, indexes, data_dict, source): 
        '''Get indexes of specific source. 
        '''
        source_indexes = np.array([index for index in indexes \
            if data_dict['source_label'][index] == source])
            
        return source_indexes
            
    def generate_validate(self, data_type, source, max_iteration=None):
        '''Generate mini-batch data for training. 
        
        Args:
          data_type: 'train' | 'validate'
          source: 'a' | 'b' | 'c'
          max_iteration: int, maximum iteration to validate to speed up validation
        
        Returns:
          batch_data_dict: dict containing audio_name, feature and target
        '''
        
        batch_size = self.batch_size
        audio_indexes = np.array(self.evaluation_audio_indexes)
            
        # audio_indexes = self.get_source_indexes(
        #     audio_indexes, self.data_dict, source)
            
        iteration = 0
        pointer = 0
        
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= len(audio_indexes):
                break

            # Get batch audio_indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]                
            pointer += batch_size
            iteration += 1

            batch_data_dict = {}
            batch_data_dict_left = {}
            batch_data_dict_right = {}
            batch_data_dict_side = {}
            batch_data_dict_harmonic = {}
            batch_data_dict_percussive = {}

            batch_data_dict['audio_name'] = \
                self.data_dict['audio_name'][batch_audio_indexes]
            batch_data_dict_left['audio_name'] = \
                self.data_dict_left['audio_name'][batch_audio_indexes]
            batch_data_dict_right['audio_name'] = \
                self.data_dict_right['audio_name'][batch_audio_indexes]
            batch_data_dict_side['audio_name'] = \
                self.data_dict_side['audio_name'][batch_audio_indexes]
            batch_data_dict_harmonic['audio_name'] = \
                self.data_dict_harmonic['audio_name'][batch_audio_indexes]
            batch_data_dict_percussive['audio_name'] = \
                self.data_dict_percussive['audio_name'][batch_audio_indexes]

            batch_feature = self.data_dict['feature'][batch_audio_indexes]
            batch_feature = self.transform(batch_feature)
            batch_data_dict['feature'] = batch_feature

            batch_feature_left = self.data_dict_left['feature_left'][batch_audio_indexes]
            batch_feature_left = self.transform_left(batch_feature_left)
            batch_data_dict_left['feature_left'] = batch_feature_left

            batch_feature_right = self.data_dict_right['feature_right'][batch_audio_indexes]
            batch_feature_right = self.transform_right(batch_feature_right)
            batch_data_dict_right['feature_right'] = batch_feature_right

            batch_feature_side = self.data_dict_side['feature_side'][batch_audio_indexes]
            batch_feature_side = self.transform_side(batch_feature_side)
            batch_data_dict_side['feature_side'] = batch_feature_side

            batch_feature_harmonic = self.data_dict_harmonic['feature_harmonic'][batch_audio_indexes]
            batch_feature_harmonic = self.transform_harmonic(batch_feature_harmonic)
            batch_data_dict_harmonic['feature_harmonic'] = batch_feature_harmonic

            batch_feature_percussive = self.data_dict_percussive['feature_percussive'][batch_audio_indexes]
            batch_feature_percussive = self.transform_percussive(batch_feature_percussive)
            batch_data_dict_percussive['feature_percussive'] = batch_feature_percussive


            yield batch_data_dict, batch_data_dict_left, batch_data_dict_right, batch_data_dict_side, batch_data_dict_harmonic, batch_data_dict_percussive
            
    def transform(self, x):
        return scale(x, self.scalar['mean'], self.scalar['std'])
    def transform_left(self, x):
        return scale(x, self.scalar_left['mean'], self.scalar_left['std'])
    def transform_right(self, x):
        return scale(x, self.scalar_right['mean'], self.scalar_right['std'])
    def transform_side(self, x):
        return scale(x, self.scalar_side['mean'], self.scalar_side['std'])
    def transform_harmonic(self, x):
        return scale(x, self.scalar_harmonic['mean'], self.scalar_harmonic['std'])
    def transform_percussive(self, x):
        return scale(x, self.scalar_percussive['mean'], self.scalar_percussive['std'])