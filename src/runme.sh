#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='/home/cdd/DCASE2019/task_1'

# You need to modify this path to your workspace to store features and models
WORKSPACE='/home/cdd/DCASE2019/task_1/workspace'

# Hyper-parameters
GPU_ID=2
MODEL_TYPE='Cnn_9layers_AvgPooling_mix'
#MODEL_TYPE='Cnn_9layers_MaxPooling'
BATCH_SIZE=32

# Subtask a:
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='a' --data_type='development' --workspace=$WORKSPACE
python utils/features_left.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='a' --data_type='development' --workspace=$WORKSPACE
python utils/features_right.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='a' --data_type='development' --workspace=$WORKSPACE
python utils/features_side.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='a' --data_type='development' --workspace=$WORKSPACE
#python utils/features_harmonic.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='a' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_percussive.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='a' --data_type='evaluation' --workspace=$WORKSPACE

python utils/features.py calculate_scalar --subtask='a' --data_type='development' --workspace=$WORKSPACE
python utils/features_left.py calculate_scalar --subtask='a' --data_type='development' --workspace=$WORKSPACE
python utils/features_right.py calculate_scalar --subtask='a' --data_type='development' --workspace=$WORKSPACE
python utils/features_side.py calculate_scalar --subtask='a' --data_type='development' --workspace=$WORKSPACE
#python utils/features_harmonic.py calculate_scalar --subtask='a' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_percussive.py calculate_scalar --subtask='a' --data_type='evaluation' --workspace=$WORKSPACE

#CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main_plus.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='a' --data_type='evaluation' --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda
#CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main_augment2.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='a' --data_type='evaluation' --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda
#
#CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main_plus_mixup.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='a' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --iteration=17800 --batch_size=$BATCH_SIZE --cuda

# Subtask b:
#python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_left.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_right.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_side.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_harmonic.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_percussive.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE

#python utils/features.py calculate_scalar --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_left.py calculate_scalar --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_right.py calculate_scalar --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_side.py calculate_scalar --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_harmonic.py calculate_scalar --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_percussive.py calculate_scalar --subtask='b' --data_type='evaluation' --workspace=$WORKSPACE

#CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='b' --data_type='evaluation' --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

#CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='b' --data_type='evaluation' --holdout_fold=1 --model_type=$MODEL_TYPE --iteration=5000 --batch_size=$BATCH_SIZE --cuda

# Subtask c:
#python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_left.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_right.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_side.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_harmonic.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_percussive.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE

#python utils/features.py calculate_scalar --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_left.py calculate_scalar --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_right.py calculate_scalar --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_side.py calculate_scalar --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_harmonic.py calculate_scalar --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE
#python utils/features_percussive.py calculate_scalar --subtask='c' --data_type='evaluation' --workspace=$WORKSPACE

#CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='c' --data_type='evaluation' --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

#CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='c' --data_type='evaluation' --holdout_fold=1 --model_type=$MODEL_TYPE --iteration=5000 --batch_size=$BATCH_SIZE --cuda

# Plot statistics
#python utils/plot_results.py --workspace=$WORKSPACE --subtask=a

############ END ############

