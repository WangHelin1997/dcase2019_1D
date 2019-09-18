import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from SpecAugment import spec_augment_pytorch
import librosa
from utilities import (create_folder, get_filename, create_logging, load_scalar,
                       get_subdir, get_sources, mixup_data, mixup_criterion)
from data_generator_plus import DataGenerator
from models_plus import Cnn_9layers_AvgPooling_mix, Cnn_9layers_MaxPooling
from losses import nll_loss
from evaluate import Evaluator, StatisticsContainer
from pytorch_utils import move_data_to_gpu
import config


def train(args):
    '''Training. Model will be saved after several iterations.

    Args:
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      subtask: 'a' | 'b' | 'c', corresponds to 3 subtasks in DCASE2019 Task1
      data_type: 'development' | 'evaluation'
      holdout_fold: '1' | 'none', set 1 for development and none for training
          on all data without validation
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    subtask = args.subtask
    data_type = args.data_type
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename

    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    max_iteration = None  # Number of mini-batches to evaluate on training data
    reduce_lr = True

    sources_to_evaluate = get_sources(subtask)
    in_domain_classes_num = len(config.labels) - 1

    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    sub_dir = get_subdir(subtask, data_type)

    train_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup',
                             'fold1_train.csv')

    validate_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup',
                                'fold1_evaluate.csv')

    feature_hdf5_path = os.path.join(workspace, 'features',
                                     '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                     '{}.h5'.format(sub_dir))
    feature_hdf5_path_left = os.path.join(workspace, 'features_left',
                                     '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                     '{}.h5'.format(sub_dir))
    feature_hdf5_path_right = os.path.join(workspace, 'features_right',
                                     '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                     '{}.h5'.format(sub_dir))
    feature_hdf5_path_side = os.path.join(workspace, 'features_side',
                                          '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                          '{}.h5'.format(sub_dir))
    feature_hdf5_path_harmonic = os.path.join(workspace, 'features_harmonic',
                                          '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                          '{}.h5'.format(sub_dir))
    feature_hdf5_path_percussive = os.path.join(workspace, 'features_percussive',
                                          '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                          '{}.h5'.format(sub_dir))
    scalar_path = os.path.join(workspace, 'scalars',
                               '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                               '{}.h5'.format(sub_dir))
    scalar_path_left = os.path.join(workspace, 'scalars_left',
                               '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                               '{}.h5'.format(sub_dir))
    scalar_path_right = os.path.join(workspace, 'scalars_right',
                               '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                               '{}.h5'.format(sub_dir))
    scalar_path_side = os.path.join(workspace, 'scalars_side',
                                    '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                    '{}.h5'.format(sub_dir))
    scalar_path_harmonic = os.path.join(workspace, 'scalars_harmonic',
                                    '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                    '{}.h5'.format(sub_dir))
    scalar_path_percussive = os.path.join(workspace, 'scalars_percussive',
                                    '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                    '{}.h5'.format(sub_dir))
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename,
                                   '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                   '{}'.format(sub_dir), 'holdout_fold={}'.format(holdout_fold),
                                   model_type)
    create_folder(checkpoints_dir)

    validate_statistics_path = os.path.join(workspace, 'statistics', filename,
                                            '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                            '{}'.format(sub_dir), 'holdout_fold={}'.format(holdout_fold),
                                            model_type, 'validate_statistics.pickle')

    create_folder(os.path.dirname(validate_statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, args.mode,
                            '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                            '{}'.format(sub_dir), 'holdout_fold={}'.format(holdout_fold), model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)

    # Load scalar
    scalar = load_scalar(scalar_path)
    scalar_left = load_scalar(scalar_path_left)
    scalar_right = load_scalar(scalar_path_right)
    scalar_side = load_scalar(scalar_path_side)
    scalar_harmonic = load_scalar(scalar_path_harmonic)
    scalar_percussive = load_scalar(scalar_path_percussive)
    # Model
    Model = eval(model_type)

    if subtask in ['a', 'b']:
        model = Model(in_domain_classes_num, activation='logsoftmax')
        loss_func = nll_loss

    elif subtask == 'c':
        model = Model(in_domain_classes_num, activation='sigmoid')
        loss_func = F.binary_cross_entropy

    if cuda:
        model.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0., amsgrad=True)

    # Data generator
    data_generator = DataGenerator(
        feature_hdf5_path=feature_hdf5_path,
        feature_hdf5_path_left=feature_hdf5_path_left,
        feature_hdf5_path_right=feature_hdf5_path_right,
        feature_hdf5_path_side=feature_hdf5_path_side,
        feature_hdf5_path_harmonic=feature_hdf5_path_harmonic,
        feature_hdf5_path_percussive=feature_hdf5_path_percussive,
        train_csv=train_csv,
        validate_csv=validate_csv,
        scalar=scalar,
        scalar_left=scalar_left,
        scalar_right=scalar_right,
        scalar_side=scalar_side,
        scalar_harmonic=scalar_harmonic,
        scalar_percussive=scalar_percussive,
        batch_size=batch_size)

    # Evaluator
    evaluator = Evaluator(
        model=model,
        data_generator=data_generator,
        subtask=subtask,
        cuda=cuda)

    # Statistics
    validate_statistics_container = StatisticsContainer(validate_statistics_path)

    train_bgn_time = time.time()
    iteration = 0

    # Train on mini batches
    for batch_data_dict, batch_data_dict_left, batch_data_dict_right, batch_data_dict_side, batch_data_dict_harmonic,\
            batch_data_dict_percussive in data_generator.generate_train():

        # Evaluates
        if iteration % 200 == 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()

            for source in sources_to_evaluate:
                train_statistics = evaluator.evaluate(
                    data_type='train',
                    source=source,
                    max_iteration=None,
                    verbose=False)

            for source in sources_to_evaluate:
                validate_statistics = evaluator.evaluate(
                    data_type='validate',
                    source=source,
                    max_iteration=None,
                    verbose=False)

                validate_statistics_container.append_and_dump(
                    iteration, source, validate_statistics)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()

        # Save model
        if iteration % 200 == 0 and iteration > 0:
            checkpoint = {
                'iteration': iteration,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))

            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

        # Reduce learning rate
        if reduce_lr and iteration % 200 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.92

        # Move data to GPU
        for key in batch_data_dict.keys():
            if key in ['feature', 'target']:
                batch_data_dict[key] = move_data_to_gpu(batch_data_dict[key], cuda)
        for key in batch_data_dict_left.keys():
            if key in ['feature_left', 'target']:
                batch_data_dict_left[key] = move_data_to_gpu(batch_data_dict_left[key], cuda)
        for key in batch_data_dict_right.keys():
            if key in ['feature_right', 'target']:
                batch_data_dict_right[key] = move_data_to_gpu(batch_data_dict_right[key], cuda)
        for key in batch_data_dict_side.keys():
            if key in ['feature_side', 'target']:
                batch_data_dict_side[key] = move_data_to_gpu(batch_data_dict_side[key], cuda)
        for key in batch_data_dict_harmonic.keys():
            if key in ['feature_harmonic', 'target']:
                batch_data_dict_harmonic[key] = move_data_to_gpu(batch_data_dict_harmonic[key], cuda)
        for key in batch_data_dict_percussive.keys():
            if key in ['feature_percussive', 'target']:
                batch_data_dict_percussive[key] = move_data_to_gpu(batch_data_dict_percussive[key], cuda)

        # # Train
        # model.train()
        # data, data_left, data_right, data_side,\
        # data_harmonic, data_percussive, target_a, target_b, lam = mixup_data(x1=batch_data_dict['feature'],
        #                                                                      x2=batch_data_dict_left['feature_left'],
        #                                                                      x3=batch_data_dict_right['feature_right'],
        #                                                                      x4=batch_data_dict_side['feature_side'],
        #                                                                      x5=batch_data_dict_harmonic['feature_harmonic'],
        #                                                                      x6=batch_data_dict_percussive['feature_percussive'],
        #                                                                      y=batch_data_dict['target'],
        #                                                                      alpha=0.2)
        # data = spec_augment_pytorch.spec_augment(data, alpha=0.001)
        # data_left = spec_augment_pytorch.spec_augment(data_left, alpha=0.001)
        # data_right = spec_augment_pytorch.spec_augment(data_right, alpha=0.001)
        # data_side = spec_augment_pytorch.spec_augment(data_side, alpha=0.001)
        # data_harmonic = spec_augment_pytorch.spec_augment(data_harmonic, alpha=0.001)
        # data_percussive = spec_augment_pytorch.spec_augment(data_percussive, alpha=0.001)
        # batch_output = model(data=data,
        #                      data_left=data_left,
        #                      data_right=data_right,
        #                      data_side=data_side,
        #                      data_harmonic=data_harmonic,
        #                      data_percussive=data_percussive)
        #
        # # loss
        # # loss = loss_func(batch_output, batch_data_dict['target'])
        # loss = mixup_criterion(loss_func, batch_output, target_a, target_b, lam)
        # # Backward
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # # Train original data
        # model.train()
        # batch_output = model(data=batch_data_dict['feature'],
        #                      data_left=batch_data_dict_left['feature_left'],
        #                      data_right=batch_data_dict_right['feature_right'],
        #                      data_side=batch_data_dict_side['feature_side'],
        #                      data_harmonic=batch_data_dict_harmonic['feature_harmonic'],
        #                      data_percussive=batch_data_dict_percussive['feature_percussive'])
        # loss = loss_func(batch_output, batch_data_dict['target'])
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # Train mixup data
        model.train()
        data, data_left, data_right, data_side, \
        data_harmonic, data_percussive, target_a, target_b, lam = mixup_data(x1=batch_data_dict['feature'],
                                                                             x2=batch_data_dict_left['feature_left'],
                                                                             x3=batch_data_dict_right['feature_right'],
                                                                             x4=batch_data_dict_side['feature_side'],
                                                                             x5=batch_data_dict_harmonic[
                                                                                 'feature_harmonic'],
                                                                             x6=batch_data_dict_percussive[
                                                                                 'feature_percussive'],
                                                                             y=batch_data_dict['target'],
                                                                             alpha=0.3)
        batch_output = model(data=data,
                             data_left=data_left,
                             data_right=data_right,
                             data_side=data_side,
                             data_harmonic=data_harmonic,
                             data_percussive=data_percussive)
        loss = mixup_criterion(loss_func, batch_output, target_a, target_b, lam)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Train SpecAugment data
        model.train()
        data = spec_augment_pytorch.spec_augment(batch_data_dict['feature'],
                                                 using_frequency_masking=True,
                                                 using_time_masking=True)
        data_left = spec_augment_pytorch.spec_augment(batch_data_dict_left['feature_left'],
                                                      using_frequency_masking=True,
                                                      using_time_masking=True)
        data_right = spec_augment_pytorch.spec_augment(batch_data_dict_right['feature_right'],
                                                       using_frequency_masking=True,
                                                       using_time_masking=True)
        data_side = spec_augment_pytorch.spec_augment(batch_data_dict_side['feature_side'],
                                                      using_frequency_masking=True,
                                                      using_time_masking=True)
        data_harmonic = spec_augment_pytorch.spec_augment(batch_data_dict_harmonic['feature_harmonic'],
                                                          using_frequency_masking=True,
                                                          using_time_masking=True)
        data_percussive = spec_augment_pytorch.spec_augment(batch_data_dict_percussive['feature_percussive'],
                                                            using_frequency_masking=True,
                                                            using_time_masking=True)
        batch_output = model(data=data,
                             data_left=data_left,
                             data_right=data_right,
                             data_side=data_side,
                             data_harmonic=data_harmonic,
                             data_percussive=data_percussive)
        loss = loss_func(batch_output, batch_data_dict['target'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Stop learning
        if iteration == 25000:
            break

        iteration += 1


def inference_validation(args):
    '''Inference and calculate metrics on validation data.

    Args:
      dataset_dir: string, directory of dataset
      subtask: 'a' | 'b' | 'c', corresponds to 3 subtasks in DCASE2019 Task1
      data_type: 'development'
      workspace: string, directory of workspace
      model_type: string, e.g. 'Cnn_9layers'
      iteration: int
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
      visualize: bool
    '''
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subtask = args.subtask
    data_type = args.data_type
    workspace = args.workspace
    model_type = args.model_type
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    visualize = args.visualize
    filename = args.filename

    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second

    sources = get_sources(subtask)
    in_domain_classes_num = len(config.labels) - 1

    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    sub_dir = get_subdir(subtask, data_type)

    train_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup',
                             'fold1_train.csv')

    validate_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup',
                                'fold1_evaluate.csv')

    feature_hdf5_path = os.path.join(workspace, 'features',
                                     '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                     '{}.h5'.format(sub_dir))
    feature_hdf5_path_left = os.path.join(workspace, 'features_left',
                                          '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                          '{}.h5'.format(sub_dir))
    feature_hdf5_path_right = os.path.join(workspace, 'features_right',
                                           '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                           '{}.h5'.format(sub_dir))
    feature_hdf5_path_side = os.path.join(workspace, 'features_side',
                                           '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                           '{}.h5'.format(sub_dir))
    feature_hdf5_path_harmonic = os.path.join(workspace, 'features_harmonic',
                                          '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                          '{}.h5'.format(sub_dir))
    feature_hdf5_path_percussive = os.path.join(workspace, 'features_percussive',
                                          '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                          '{}.h5'.format(sub_dir))
    scalar_path = os.path.join(workspace, 'scalars',
                               '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                               '{}.h5'.format(sub_dir))
    scalar_path_left = os.path.join(workspace, 'scalars_left',
                                    '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                    '{}.h5'.format(sub_dir))
    scalar_path_right = os.path.join(workspace, 'scalars_right',
                                     '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                     '{}.h5'.format(sub_dir))
    scalar_path_side = os.path.join(workspace, 'scalars_side',
                                     '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                     '{}.h5'.format(sub_dir))
    scalar_path_harmonic = os.path.join(workspace, 'scalars_harmonic',
                                    '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                    '{}.h5'.format(sub_dir))
    scalar_path_percussive = os.path.join(workspace, 'scalars_percussive',
                                    '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                    '{}.h5'.format(sub_dir))

    checkpoint_path = os.path.join(workspace, 'checkpoints', filename,
                                   '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                   '{}'.format(sub_dir), 'holdout_fold={}'.format(holdout_fold),
                                   model_type, '{}_iterations.pth'.format(iteration))

    logs_dir = os.path.join(workspace, 'logs', filename, args.mode,
                            '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                            '{}'.format(sub_dir), 'holdout_fold={}'.format(holdout_fold),
                            model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)

    # Load scalar
    scalar = load_scalar(scalar_path)
    scalar_left = load_scalar(scalar_path_left)
    scalar_right = load_scalar(scalar_path_right)
    scalar_side = load_scalar(scalar_path_side)
    scalar_harmonic = load_scalar(scalar_path_harmonic)
    scalar_percussive = load_scalar(scalar_path_percussive)

    # Load model
    Model = eval(model_type)

    if subtask in ['a', 'b']:
        model = Model(in_domain_classes_num, activation='logsoftmax')
        loss_func = nll_loss

    elif subtask == 'c':
        model = Model(in_domain_classes_num, activation='sigmoid')
        loss_func = F.binary_cross_entropy

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    if cuda:
        model.cuda()

    # Data generator
    data_generator = DataGenerator(
        feature_hdf5_path=feature_hdf5_path,
        feature_hdf5_path_left=feature_hdf5_path_left,
        feature_hdf5_path_right=feature_hdf5_path_right,
        feature_hdf5_path_side=feature_hdf5_path_side,
        feature_hdf5_path_harmonic=feature_hdf5_path_harmonic,
        feature_hdf5_path_percussive=feature_hdf5_path_percussive,
        train_csv=train_csv,
        validate_csv=validate_csv,
        scalar=scalar,
        scalar_left=scalar_left,
        scalar_right=scalar_right,
        scalar_side=scalar_side,
        scalar_harmonic=scalar_harmonic,
        scalar_percussive=scalar_percussive,
        batch_size=batch_size)
    # Evaluator
    evaluator = Evaluator(
        model=model,
        data_generator=data_generator,
        subtask=subtask,
        cuda=cuda)

    if subtask in ['a', 'c']:
        evaluator.evaluate(data_type='validate', source='a', verbose=True)

    elif subtask == 'b':
        evaluator.evaluate(data_type='validate', source='a', verbose=True)
        evaluator.evaluate(data_type='validate', source='b', verbose=True)
        evaluator.evaluate(data_type='validate', source='c', verbose=True)

    # Visualize log mel spectrogram
    if visualize:
        evaluator.visualize(data_type='validate', source='a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--subtask', type=str, choices=['a', 'b', 'c'], required=True,
                              help='Correspond to 3 subtasks in DCASE2019 Task1.')
    parser_train.add_argument('--data_type', type=str, choices=['development', 'evaluation'], required=True)
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', 'none'], required=True,
                              help='Set 1 for development and none for training on all data without validation.')
    parser_train.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False,
                              help='Set True for debugging on a small part of data.')

    # Inference validation data
    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_validation.add_argument('--workspace', type=str, required=True,
                                             help='Directory of your workspace.')
    parser_inference_validation.add_argument('--subtask', type=str, choices=['a', 'b', 'c'], required=True,
                                             help='Correspond to 3 subtasks in DCASE2019 Task1.')
    parser_inference_validation.add_argument('--data_type', type=str, choices=['development'], required=True)
    parser_inference_validation.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_validation.add_argument('--model_type', type=str, required=True,
                                             help='E.g., Cnn_9layers_AvgPooling.')
    parser_inference_validation.add_argument('--iteration', type=int, required=True,
                                             help='This iteration of the trained model will be loaded.')
    parser_inference_validation.add_argument('--batch_size', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--visualize', action='store_true', default=False,
                                             help='Visualize log mel spectrogram of different sound classes.')
    parser_inference_validation.add_argument('--mini_data', action='store_true', default=False,
                                             help='Set True for debugging on a small part of data.')

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation':
        inference_validation(args)

    else:
        raise Exception('Error argument!')