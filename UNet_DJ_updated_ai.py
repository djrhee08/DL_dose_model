# -*- coding: utf-8 -*-
"""
@author: YZhao15
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Dropout, Conv3DTranspose
from skimage.transform import resize

def load(image_file):
    image_file = image_file.numpy().decode("utf-8")
    base_name = os.path.basename(image_file)
    prefix_match = re.match(r"(CT|dose|jaw|mlc)_test_(patient\d+_plan\d+)_(\d+)", base_name)
    
    if prefix_match is None:
        raise ValueError("Filename does not match expected pattern.")

    modality, prefix, number = prefix_match.groups()
    path = os.path.dirname(image_file)
    ct_file = os.path.join(path, f"CT_test_{prefix}_{number}.npy")
    dose_file = os.path.join(path, f"dose_test_{prefix}_{number}.npy")
    jaw_file = os.path.join(path, f"jaw_test_{prefix}_{number}.npy")
    mlc_file = os.path.join(path, f"mlc_test_{prefix}_{number}.npy")

    ct_image = np.load(ct_file)
    dose_image = np.load(dose_file)
    jaw_image = np.load(jaw_file)
    mlc_image = np.load(mlc_file)
    target_size = (256, 256, 256)
    
    ct_image_resized = resize(ct_image, target_size, order=3, mode='constant', cval=0, anti_aliasing=True)
    dose_image_resized = resize(dose_image, target_size, order=3, mode='constant', cval=0, anti_aliasing=True)
    jaw_image_resized = resize(jaw_image, target_size, order=3, mode='constant', cval=0, anti_aliasing=True)
    mlc_image_resized = resize(mlc_image, target_size, order=3, mode='constant', cval=0, anti_aliasing=True)

    """
    print("original")
    print(f"Minimum value of ct_image: {np.min(ct_image)}, Maximum value of ct_image: {np.max(ct_image)}")
    print(f"Minimum value of dose_image: {np.min(dose_image)}, Maximum value of dose_image: {np.max(dose_image)}")
    print(f"Minimum value of jaw_image: {np.min(jaw_image)}, Maximum value of jaw_image: {np.max(jaw_image)}")
    print(f"Minimum value of mlc_image: {np.min(mlc_image)}, Maximum value of mlc_image: {np.max(mlc_image)}")

    print("resized")
    print(f"ct_image_resized shape: {ct_image_resized.shape}, min: {np.min(ct_image_resized)}, max: {np.max(ct_image_resized)}")
    print(f"dose_image_resized shape: {dose_image_resized.shape}, min: {np.min(dose_image_resized)}, max: {np.max(dose_image_resized)}")
    print(f"jaw_image_resized shape: {jaw_image_resized.shape}, min: {np.min(jaw_image_resized)}, max: {np.max(jaw_image_resized)}")
    print(f"mlc_image_resized shape: {mlc_image_resized.shape}, min: {np.min(mlc_image_resized)}, max: {np.max(mlc_image_resized)}")
    """
        
    ct_image = tf.convert_to_tensor(ct_image_resized, dtype=tf.float32)
    dose_image = tf.convert_to_tensor(dose_image_resized, dtype=tf.float32)
    jaw_image = tf.convert_to_tensor(jaw_image_resized, dtype=tf.float32)
    mlc_image = tf.convert_to_tensor(mlc_image_resized, dtype=tf.float32)

    ct_image = tf.expand_dims(ct_image, -1)
    dose_image = tf.expand_dims(dose_image, -1)
    jaw_image = tf.expand_dims(jaw_image, -1)
    mlc_image = tf.expand_dims(mlc_image, -1)

    combined_input = tf.concat([ct_image, jaw_image, mlc_image], axis=-1)
    return combined_input, dose_image

def load_image_train(image_file):
    combined_input, dose_image = tf.py_function(func=load, inp=[image_file], Tout=[tf.float32, tf.float32])
    combined_input.set_shape([256, 256, 256, 3])
    dose_image.set_shape([256, 256, 256, 1])
    combined_input = tf.image.random_flip_left_right(combined_input)
    dose_image = tf.image.random_flip_left_right(dose_image)
    return combined_input, dose_image

def load_image_test(image_file):
    combined_input, dose_image = tf.py_function(func=load, inp=[image_file], Tout=[tf.float32, tf.float32])
    combined_input.set_shape([256, 256, 256, 3])
    dose_image.set_shape([256, 256, 256, 1])
    return combined_input, dose_image

def create_dataset(path_to_train_images, path_to_test_images, buffer_size, batch_size):
    train_dataset = tf.data.Dataset.list_files(path_to_train_images)
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.list_files(path_to_test_images)
    test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset

def show_predictions(dataset, num_samples=3):
    for images, true_doses in dataset.take(num_samples):
        predicted_doses = model(images, training=False)
        plt.figure(figsize=(15, 5))

        for i in range(num_samples):
            plt.subplot(3, num_samples, i + 1)
            plt.imshow(tf.squeeze(images[i,128, ..., 0]), cmap='gray')
            plt.title("Input Image")
            plt.axis('off')

            plt.subplot(3, num_samples, i + num_samples + 1)
            plt.imshow(tf.squeeze(true_doses[i,128, ..., 0]), cmap='viridis')
            plt.title("True Dose")
            plt.axis('off')

            plt.subplot(3, num_samples, i + 2 * num_samples + 1)
            plt.imshow(tf.squeeze(predicted_doses[i,128, ..., 0]), cmap='viridis')
            plt.title("Predicted Dose")
            plt.axis('off')
        plt.show()

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_dataset):
        super(DisplayCallback, self).__init__()
        self.train_dataset = train_dataset

    def on_epoch_end(self, epoch, logs=None):
        show_predictions(self.train_dataset, 3)
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))

def Unet(n_filters=64):
    inputs = Input(shape=[256, 256, 256, 3])
    conv1 = Conv3D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv3D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.3)(conv5)

    up6 = Conv3DTranspose(n_filters * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(drop5)
    merge6 = concatenate([up6, conv4], axis=-1)
    conv6 = Conv3D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv3D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv3DTranspose(n_filters * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
    merge7 = concatenate([up7, conv3], axis=-1)
    conv7 = Conv3D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv3D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv3DTranspose(n_filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
    merge8 = concatenate([up8, conv2], axis=-1)
    conv8 = Conv3D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv3D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
    merge9 = concatenate([up9, conv1], axis=-1)
    conv9 = Conv3D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv3D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv3D(1, (1, 1, 1), activation='linear', padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    return model

path = r'C:\Users\djrhe\Desktop\Research\dose_cal_yao'
path_train = r'C:\Users\djrhe\Desktop\Research\dose_cal_yao\numpy'
path_test = r'C:\Users\djrhe\Desktop\Research\dose_cal_yao\test'
batch_size = 1
train_dataset, test_dataset = create_dataset(
        os.path.join(path_train, 'CT_test_patient*_plan*_*.npy'),
        os.path.join(path_test, 'CT_test_patient*_plan*_*.npy'),
        buffer_size=2, batch_size=batch_size)

model = Unet(n_filters=8)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='mse', 
              metrics=['mae'], run_eagerly=True)

STEPS_PER_EPOCH = len(os.listdir(path_train)) // batch_size
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(os.listdir(path_test)) // batch_size // VAL_SUBSPLITS

model_checkpoint = tf.keras.callbacks.ModelCheckpoint('Unet_model_new.h5', verbose=1, save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(patience=40, verbose=1)
lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=15, verbose=1)

display_callback = DisplayCallback(test_dataset)

EPOCHS = 200
model_history = model.fit(train_dataset.repeat(), epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[model_checkpoint, early_stop, lr, display_callback])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
