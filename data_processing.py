import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from itertools import chain


TRAIN_DATA_PATH = 'data/train'
TEST_DATA_PATH = 'data/test'
PROCESSED_DATA_PATH = 'data/processed'

IMG_HEIGHT = 420
IMG_WIDTH = 580

RESIZE_HEIGHT = 96
RESIZE_WIDTH = 96


def transform_train_imgs_to_npy():
    
    train_data_files = os.listdir(TRAIN_DATA_PATH)
    imgs_number = int(len(train_data_files) / 2)
    
    imgs_array = np.ndarray((imgs_number, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    masks_array = np.ndarray((imgs_number, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    
    counter = 0
    
    for img_filename in train_data_files:
        
        if 'mask' in img_filename:
            continue
            
        img_path = '/'.join([TRAIN_DATA_PATH, img_filename])
        mask_filename = '{}_mask.tif'.format(img_filename.split('.')[0])
        mask_path = '/'.join([TRAIN_DATA_PATH, mask_filename])
        
        img = imread(img_path)
        img_array = np.array(img)
        mask = imread(mask_path)
        mask_array = np.array(mask)
        
        imgs_array[counter] = img
        masks_array[counter] = mask
        
        counter += 1
        
        if counter % 500 == 0:
            print('{} of {} images processed'.format(counter, imgs_number))
    
    np.save('{}/train_imgs.npy'.format(PROCESSED_DATA_PATH), imgs_array)
    np.save('{}/train_masks.npy'.format(PROCESSED_DATA_PATH), masks_array)
    
    
def transform_test_imgs_to_npy():
    
    test_data_files = os.listdir(TEST_DATA_PATH)
    imgs_number = int(len(test_data_files))
    
    imgs_array = np.ndarray((imgs_number, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    
    counter = 0
    
    with open('{}/tags.csv'.format(PROCESSED_DATA_PATH), 'w') as tags_file:
        for img_filename in test_data_files:
        
            img_path = '/'.join([TEST_DATA_PATH, img_filename])
            tags_file.write(img_filename.split('.')[0] + ',')
        
            img = imread(img_path)
            img_array = np.array(img)
        
            imgs_array[counter] = img
        
            counter += 1
        
            if counter % 500 == 0:
                print('{} of {} images processed'.format(counter, imgs_number))
    
    np.save('{}/test_imgs.npy'.format(PROCESSED_DATA_PATH), imgs_array)
    

def load_train_data():
    train_imgs = np.load('{}/train_imgs.npy'.format(PROCESSED_DATA_PATH))
    train_masks = np.load('{}/train_masks.npy'.format(PROCESSED_DATA_PATH))
    return train_imgs, train_masks


def load_test_data():
    with open('{}/tags.csv'.format(PROCESSED_DATA_PATH), 'r') as tags_file:
        tags = tags_file.read().split(',')[:-1]
    test_imgs = np.load('{}/test_imgs.npy'.format(PROCESSED_DATA_PATH))
    return test_imgs, tags
    
    
def resize_imgs(array, rows_num, cols_num):
    imgs = np.ndarray((array.shape[0], rows_num, cols_num), dtype=np.uint8)
    for i in range(array.shape[0]):
        imgs[i] = resize(array[i], (rows_num, cols_num), preserve_range=True)
    return imgs


def add_dim(array):
    return np.expand_dims(array, axis=3)


def rl_encode(array):
    flat_array = array.T.flatten()
    bounded = np.hstack(([0], flat_array, [0]))
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    lengths = run_ends - run_starts
                
    def block_gen():
        for pix, length in zip(run_starts, lengths):
            if length > 1:
                yield str(pix + 1), str(length)

    result = ' '.join(chain.from_iterable(block_gen()))
    return result
