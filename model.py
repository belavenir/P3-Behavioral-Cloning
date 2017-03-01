import os
import json
import pandas as pd
import numpy as np
import cv2

import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# CONSTANT
Udacity_LOG = './data2/driving_log.csv'
IMG_PATH = './data2/'
RECOVERY_STEER = .20

# Data Preprocessing & Augmentation

#Crop useless information in image
def crop(image, top, bottom):
    
    top = int(np.ceil((image.shape[0]*top)))
    bottom = int(np.ceil((image.shape[0]*(1-bottom))))
    
    return image[top:bottom, :]


# Augment by illumination
def augment_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.4 + np.random.uniform() 
    image[:,:,2] = image[:,:,2]*random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image


# Translation
def trans(image,steer,trans_range):
   
    rows = image.shape[0]
    cols = image.shape[1]
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steering_angle = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return {'image':image_tr,'steer': steering_angle}

# Flipping
def flip(image, steer):
    return{'image': cv2.flip(image,1),'steer': -1*steer} 


# Shearing
def shear(image, steering_angle, shear_range=40):
  
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return {'image': image, 'steer':steering_angle}


# Image augmentation
def img_process(image, steering_angle, train=True):
    rows = image.shape[0]
    cols = image.shape[1]

    #Translate image and compensate for steering angle
    trans_range = 40
    image = trans(image, steering_angle, trans_range)['image']
    steer = trans(image, steering_angle, trans_range)['steer']

    #Crop image of ROI
    image = crop(image, 0.3, 0.1)

    #Flip image randomly
    if np.random.uniform()>= 0.5:
        image = flip(image, steer)['image']
        steer = flip(image, steer)['steer']
    #Brightness
    image = augment_brightness(image)

    return image, steer

# Data generator, generator on the fly
# Generate data for training rather than storing in memory

# Get a batch_size filename and steering angles
def get_batch_files(batch_size=256):

    data = pd.read_csv(Udacity_LOG)
    num_img = len(data)
    rnd_indices = np.random.randint(0, num_img, batch_size)
    
    file_steer = []

    for index in rnd_indices:
        cam_view = np.random.choice(['center', 'left','right'])
        if cam_view =='left':
            image_file = data.iloc[index][cam_view].strip()
            steer = data.iloc[index]['steering']+ RECOVERY_STEER
            file_steer.append((image_file, steer))
        elif  cam_view == 'center':
            image_file = data.iloc[index][cam_view].strip()
            steer = data.iloc[index]['steering']
            file_steer.append((image_file, steer))
        elif cam_view == 'right':
            image_file = data.iloc[index][cam_view].strip()
            steer = data.iloc[index]['steering'] - RECOVERY_STEER
            file_steer.append((image_file, steer))

    return file_steer

# Generate a batch_size data simple

def generate_batch_sample(batch_size = 256):
    while True:
        batch_x = []
        batch_y = []
        file_steer = get_batch_files(batch_size)
        for img_file, steer in file_steer:
            raw_img = plt.imread(img_file)
            raw_steer = steer

            new_img, new_steer = img_process(raw_img, raw_steer)
            
            new_img = cv2.resize(new_img, (200,66)) #cols, rows

            batch_x.append(new_img)
            batch_y.append(new_steer)


        
        batch_x, batch_y = shuffle(batch_x, batch_y, random_state=0)
        yield np.array(batch_x), np.array(batch_y)




# Create model according to Nvidia model in suggested paper
import tensorflow as tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Lambda
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.models import Model


nb_epochs = 15
batch_size = 256
learning_rate = 1e-4
nb_of_samples_per_epoch = 256*80
nb_of_validation_samples = 6880

def nvidia_model():
    # Define varibales


    model  = Sequential()
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=(66, 200, 3))) #inputshape = (row, col, ch)
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout (0.5))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.summary()

    # Compile the model using Adam Optimizer
    # loss computed by "mean squared error"
    model.compile(optimizer=Adam(learning_rate),loss='mse')

    return model



train_gen = generate_batch_sample()
valid_gen = generate_batch_sample()


model = nvidia_model()
model.compile('adam', 'mse')

history = model.fit_generator(train_gen,
                              samples_per_epoch=nb_of_samples_per_epoch,
                              nb_epoch=nb_epochs,
                              validation_data=valid_gen,
                              nb_val_samples=nb_of_validation_samples,
                              verbose=1)


model.save('model.h5')
#save_model(model)


'''
# Model of transfer learning with VGG16
from keras.applications.vgg16 import VGG16
from keras.layers import Input



def vgg(input_shape):
    input_layer = Input(shape=input_shape)
    # VGG base model with ImageNet weights
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
    # Add 2 FC dense layer with dropout til one neuron layer for steeer 
    layer = base_model.output
    layer = Flatten()(layer)
    layer = Dense(1164, activation='relu', name='fc1')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(50, activation='relu', name='fc2')(layer)
    layer = Dense(1, activation='linear', name='predic')(layer)
    model = Model(input=base_model.input, output=layer)
    model.summary()
    
    return model

input_shape = (64, 64, 3)

print("")
print("second model")
print("")

model2 = vgg(input_shape)
model2.compile('adam', 'mse')

history2 = model2.fit_generator(train_gen,
                              samples_per_epoch=nb_of_samples_per_epoch,
                              nb_epoch=nb_epochs,
                              validation_data=valid_gen,
                              nb_val_samples=nb_of_validation_samples,
                              verbose=1)

model2.save('model2.h5')
'''