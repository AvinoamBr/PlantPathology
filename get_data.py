import glob
import os
import re

import cv2
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


import consts as C
IMAGE_PATH = os.path.dirname(os.path.realpath(__file__)) + "\\data"

def normalize(image):
    # from [0:255] to [(-1):1]
    return (image / (255/2)) - 1

def unnormalize(image):
    # return image to 0-255 index.
    return ((image + 1)* (255/2))



def get_labels():
    labels = pd.read_csv(IMAGE_PATH + "\\train.csv",index_col = 'image_id')
    return labels

def get_class_weights():
    # since classes are imbalanced, we want to give more weight to the rare classes.
    labels = pd.read_csv(IMAGE_PATH + "\\train.csv", index_col='image_id')
    inv_freq =  1/labels[['healthy', 'multiple_diseases', 'rust', 'scab']].mean(axis=0).values
    return inv_freq/inv_freq.sum()

def augmentation(img, max_corp = [0.05]*4, noise_size = 2,
                 flip_h = 0.2, flip_v = 0.2,
                 gaussian_sigma = 5 ) :
    # A - corp image (can change aspect ratio also)
    max_corp = np.array(max_corp)
    h,w,_ = img.shape
    relative_corp = np.multiply(np.random.uniform(size = 4),max_corp)
    corp = np.multiply(
                np.array([[0, 1], [0, 1], [1, -1], [1, -1]]),
                np.array([[h, w, h, w], np.multiply(np.array([h, w, h, w]), relative_corp)]).T)\
           .sum(axis=1)
    y,x,h,w = corp.astype(int)
    img = img[x:w,y:h]

    # B add noise:
    noise = np.random.randint(int(-noise_size/2) , int(noise_size/2), img.shape)
    img = img.astype(np.int32) + noise
    img = np.maximum(np.zeros(img.shape),img)
    img = np.minimum(np.ones(img.shape)*255,img)
    img = img.astype(np.uint8)

    # C flip
    if np.random.uniform()< flip_h:
        img = cv2.flip(img,1)
    if np.random.uniform()< flip_v:
        img = cv2.flip(img,2)

    # D blur:
    img = cv2.GaussianBlur(img,(3,3),
                           sigmaX= np.random.uniform()* gaussian_sigma,
                           sigmaY= np.random.uniform()* gaussian_sigma)

    return img

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_names, labels, batch_size = C.BATCH_SIZE , dim = C.NEW_SIZE , n_channels = 3,
                 n_classes = 4, shuffle=False, use_augmentation = True
                ):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.file_names = file_names
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.use_augmentation = use_augmentation
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __iter__(self):
        self._iter_index = -1
        return self

    def __next__(self):
        self._iter_index +=1
        if self._iter_index == len(self):
            raise StopIteration
        return self[self._iter_index]

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        files_for_batch = [self.file_names[k] for k in indexes]

        # Generate data
        self.X, self.y = self.__data_generation(files_for_batch)

        return self.X, self.y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, files_for_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=float)

        # Generate data
        for i, fn in enumerate(files_for_batch):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')

            img = cv2.imread(fn)
            if self.use_augmentation:
                img = augmentation(img)

            img = normalize(img)
            img = cv2.resize(img, self.dim)
            X[i,] = img

            # Store class
            label_index = os.path.split(fn)[-1].replace(".jpg","")
            y[i] = self.labels.loc[label_index]
        return X, y

def preprocessing_function(img):
    # crop
    # It looks that data in the peripheries is relatively redundant
    # since we shrink the image and loose data, we refer to crop this out
    original_size = C.ORIGINAL_SIZE
    center_crop  = C.CENTER_CROP
    img = img[center_crop[0]:original_size[0]-center_crop[0],
          center_crop[1]:original_size[1]-center_crop[1]]

    # resize:
    img = cv2.resize(img,(C.NEW_SIZE[1],C.NEW_SIZE[0]))

    return img

def get_generators():
    images = sorted(glob.glob(IMAGE_PATH + "\\images\\*.jpg"))
    images = sorted(images,
                   key = lambda fn: \
                       int(re.findall(r'\d+', os.path.abspath(fn).split("\\")[-1])[0]))
    data = [i for i in images if "Train" in i]
    labels = get_labels()
    train_d , valid_d , train_l , valid_l = train_test_split(data, labels, shuffle = True, random_state=42 )
    train_data_generator = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.02,
        zoom_range=0.1,
        width_shift_range=0.03, height_shift_range=0.03,
        horizontal_flip=True,
        vertical_flip= True,
        preprocessing_function = preprocessing_function
        # brightness_range = [0.9,1.1]
    )
    # train_generator = DataGenerator(train_d,train_l, shuffle=True)
    # validation_generator = DataGenerator(valid_d, valid_l, shuffle=True, use_augmentation = False)

    train_generator = train_data_generator.flow_from_directory(
        IMAGE_PATH + "\\imagesby_cat\\train", class_mode='categorical',
        target_size=C.NEW_SIZE,
        batch_size = C.BATCH_SIZE,
        # save_to_dir = IMAGE_PATH +  "\\imagesby_cat\\train_augmented"
       )

    validation_data_generator = ImageDataGenerator(
        rescale=1. / 255,
        preprocessing_function=preprocessing_function
    )
    validation_generator = validation_data_generator.flow_from_directory(
        IMAGE_PATH+"\\imagesby_cat\\validation", class_mode='categorical',
        target_size=C.ORIGINAL_SIZE,
        batch_size=C.BATCH_SIZE,
    )

    return train_generator, validation_generator
