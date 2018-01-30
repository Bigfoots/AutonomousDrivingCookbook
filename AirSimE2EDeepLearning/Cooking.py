import random
import csv
from PIL import Image
import numpy as np
import pandas as pd
import sys
import os
import errno
from collections import OrderedDict
import h5py
from pathlib import Path
import copy
import re

def checkAndCreateDir(full_path):  
    """检查给出的路径是否存在，不存在，创建此路径。
            输入：
                full_path: 路径
    """
    if not os.path.exists(os.path.dirname(full_path)):
        try:
            os.makedirs(os.path.dirname(full_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
def readImagesFromPath(image_names):  
    """ 进入一个目录，导入一个图片文件名字的列表，返回一个导入图片后尺寸改变了的图片的列表。
           输入：
                image_names: 图片名字的列表
           返回值：
                所有导入并且被改变尺寸的图片的列表
    """
    returnValue = []
    for image_name in image_names:
        im = Image.open(image_name)
        imArr = np.asarray(im)
        
        #Remove alpha channel if exists
        if len(imArr.shape) == 3 and imArr.shape[2] == 4:
            if (np.all(imArr[:, :, 3] == imArr[0, 0, 3])):
                imArr = imArr[:,:,0:3]
        if len(imArr.shape) != 3 or imArr.shape[2] != 3:
            print('Error: Image', image_name, 'is not RGB.')
            sys.exit()            

        returnIm = np.asarray(imArr)

        returnValue.append(returnIm)
    return returnValue
    
    
    
def splitTrainValidationAndTestData(all_data_mappings, split_ratio=(0.7, 0.2, 0.1)):  
    """将原始数据按比例分配给train, validation and test 数据集。
            输入：
                all_data_mappings: 全部映射的数据集
                split_ratio: (train, validation, test) 分配比例，加起来等于一。

            返回值：
                train_data_mappings: 训练数据集映射
                validation_data_mappings: 验证数据集映射
                test_data_mappings: 测试数据集映射

    """
    if round(sum(split_ratio), 5) != 1.0:
        print("Error: Your splitting ratio should add up to 1")
        sys.exit()

    train_split = int(len(all_data_mappings) * split_ratio[0])
    val_split = train_split + int(len(all_data_mappings) * split_ratio[1])

    # 这里因为数据集映射在被创建时就随机打乱了，所以这里就取前面对应的比例就行。
    train_data_mappings = all_data_mappings[0:train_split]
    validation_data_mappings = all_data_mappings[train_split:val_split]
    test_data_mappings = all_data_mappings[val_split:]

    return [train_data_mappings, validation_data_mappings, test_data_mappings]
    
def generateDataMapAirSim(folders):
    """ 数据映射生成器。读取driving_log csv 文件，返回一个包含”图片名称 - 标注“ 元组的列表。
           输入：
               folders: 所有数据文件组成的一个列表

           返回值：
               mappings: 所有的数据映射为一个列表。列表由二维元组组成： （原函数标注有误）
                0 -> 图片的目录
                1 -> 一个二维元组：
                   0 -> 标注（double类型 的列表），标注就是steering  
                   1 -> 之前的状态（double类型 的列表），状态值都有：['Steering', 'Throttle', 'Brake', 'Speed (kmph)']
    """

    all_mappings = {}
    for folder in folders:
        print('Reading data from {0}...'.format(folder))
        current_df = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), sep='\t')
        
        for i in range(1, current_df.shape[0] - 1, 1):
            previous_state = list(current_df.iloc[i-1][['Steering', 'Throttle', 'Brake', 'Speed (kmph)']])
            current_label = list((current_df.iloc[i][['Steering']] + current_df.iloc[i-1][['Steering']] + current_df.iloc[i+1][['Steering']]) / 3.0)
            
            image_filepath = os.path.join(os.path.join(folder, 'images'), current_df.iloc[i]['ImageName']).replace('\\', '/')
            
            # Sanity check
            if (image_filepath in all_mappings):
                print('Error: attempting to add image {0} twice.'.format(image_filepath))
            
            all_mappings[image_filepath] = (current_label, previous_state)
    
    mappings = [(key, all_mappings[key]) for key in all_mappings]
    
    random.shuffle(mappings)# 随机打乱顺序
    
    return mappings

def generatorForH5py(data_mappings, chunk_size=32):  
    """
    批量化数据来保存到 h5 文件
    """
    for chunk_id in range(0, len(data_mappings), chunk_size):
        # Data is expected to be a dict of <image: (label, previousious_state)>
        # Extract the parts
        data_chunk = data_mappings[chunk_id:chunk_id + chunk_size]
        if (len(data_chunk) == chunk_size):
            image_names_chunk = [a for (a, b) in data_chunk]
            labels_chunk = np.asarray([b[0] for (a, b) in data_chunk])
            previous_state_chunk = np.asarray([b[1] for (a, b) in data_chunk])
            
            #Flatten and yield as tuple
            yield (image_names_chunk, labels_chunk.astype(float), previous_state_chunk.astype(float))
            if chunk_id + chunk_size > len(data_mappings):
                raise StopIteration
    raise StopIteration
    
def saveH5pyData(data_mappings, target_file_path):
    """
    保存 h5 数据文件。
    """
    chunk_size = 32
    gen = generatorForH5py(data_mappings,chunk_size)

    image_names_chunk, labels_chunk, previous_state_chunk = next(gen)
    images_chunk = np.asarray(readImagesFromPath(image_names_chunk))
    row_count = images_chunk.shape[0]

    checkAndCreateDir(target_file_path)
    with h5py.File(target_file_path, 'w') as f:

        # Initialize a resizable dataset to hold the output
        images_chunk_maxshape = (None,) + images_chunk.shape[1:]
        labels_chunk_maxshape = (None,) + labels_chunk.shape[1:]
        previous_state_maxshape = (None,) + previous_state_chunk.shape[1:]

        dset_images = f.create_dataset('image', shape=images_chunk.shape, maxshape=images_chunk_maxshape,
                                chunks=images_chunk.shape, dtype=images_chunk.dtype)

        dset_labels = f.create_dataset('label', shape=labels_chunk.shape, maxshape=labels_chunk_maxshape,
                                       chunks=labels_chunk.shape, dtype=labels_chunk.dtype)
        
        dset_previous_state = f.create_dataset('previous_state', shape=previous_state_chunk.shape, maxshape=previous_state_maxshape,
                                       chunks=previous_state_chunk.shape, dtype=previous_state_chunk.dtype)
                                       
        dset_images[:] = images_chunk
        dset_labels[:] = labels_chunk
        dset_previous_state[:] = previous_state_chunk

        for image_names_chunk, label_chunk, previous_state_chunk in gen:
            image_chunk = np.asarray(readImagesFromPath(image_names_chunk))
            
            # Resize the dataset to accommodate the next chunk of rows
            dset_images.resize(row_count + image_chunk.shape[0], axis=0)
            dset_labels.resize(row_count + label_chunk.shape[0], axis=0)
            dset_previous_state.resize(row_count + previous_state_chunk.shape[0], axis=0)
            # Write the next chunk
            dset_images[row_count:] = image_chunk
            dset_labels[row_count:] = label_chunk
            dset_previous_state[row_count:] = previous_state_chunk

            # Increment the row count
            row_count += image_chunk.shape[0]
            
            
def cook(folders, output_directory, train_eval_test_split):
    """ 数据预处理基本函数。读取和保存所有数据到 h5 文件。（可以从tutorial 文本中看到，cook 函数被直接调用）
            输入：
                folders: 所有数据文件组成的一个列表
                output_directory: 保存 h5 文件的目录
                train_eval_test_split: 数据分配比
    """
    output_files = [os.path.join(output_directory, f) for f in ['train.h5', 'eval.h5', 'test.h5']]# 构建保存文件.h5
    if (any([os.path.isfile(f) for f in output_files])): # any(x)判断x对象是否为空对象，如果都为空、0、false，则返回false，如果不都为空、0、false，则返回true
       print("Preprocessed data already exists at: {0}. Skipping preprocessing.".format(output_directory))

    else:
        all_data_mappings = generateDataMapAirSim(folders)
        
        split_mappings = splitTrainValidationAndTestData(all_data_mappings, split_ratio=train_eval_test_split)
        
        for i in range(0, len(split_mappings), 1):
            print('Processing {0}...'.format(output_files[i]))
            saveH5pyData(split_mappings[i], output_files[i])
            print('Finished saving {0}.'.format(output_files[i]))