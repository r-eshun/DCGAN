# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 18:13:34 2021

@author: paaes
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#import the required packages
import GPUtil
import os
import time
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
#matplotlib inline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import gdown
from zipfile import ZipFile
import glob
#import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import json
import math
import os
import cv2
from PIL import Image
from pathlib import Path

from IPython import display

# Data loading and preprocessing
def getListOfFiles(dirName):
    RESIZE=128
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
   
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
         
            allFiles = allFiles + getListOfFiles(fullPath)
            
        else:
            img = read(fullPath)
           
            img = cv2.resize(img, (RESIZE,RESIZE))
            #allFiles.append(fullPath)
            allFiles.append(img)   
           
           
    return allFiles   

train_images = np.array(getListOfFiles('C:/DataSets/BreaKHis_MX/100_X'))
print(len(train_images))
#train_images = np.array(getListOfFiles('C:/DataSets/breast/train/benign'))
train_images = train_images.astype('float32')
#train_images = (train_images - 255) / 255  # Normalize the images to [-1, 1]

BUFFER_SIZE = 655
BATCH_SIZE = 256
latent_dim = 100

# Batch and shuffle the data
train_ds = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


#dataset = np.array(getListOfFiles('C:/DataSets/breast/train/benign'))
#dataset = train_dataset.map(lambda x: x / 255)




#AUTOTUNE = tf.data.experimental.AUTOTUNE

#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

#normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./255)
#normalized_ds = train_ds.map(lambda x: normalization_layer(x))
normalized_ds = train_ds.map(lambda x: x / 255)
'''
# Display a sample image
for x in normalized_ds:
    plt.axis("off")
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    break
'''

# Generator function
def generator1():
    
    inputs = keras.Input(shape=(1, 1, 100), name='input_layer')
    # Block 1:input is latent(100), going into a convolution
    x = layers.Conv2DTranspose(64 * 8, kernel_size=4, strides= 4, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1')(inputs)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)
    
    # Block 2: input is 4 x 4 x (64 * 8)
    x = layers.Conv2DTranspose(64 * 4, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_2')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
    x = layers.ReLU(name='relu_2')(x)
    
    # Block 3: input is 8 x 8 x (64 * 4)
    x = layers.Conv2DTranspose(64 * 2, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_3')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_3')(x)
    x = layers.ReLU(name='relu_3')(x)
  
    # Block 4: input is 16 x 16 x (64 * 2)
    x = layers.Conv2DTranspose(64 * 1, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_4')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_4')(x)
    x = layers.ReLU(name='relu_4')(x)
    
    # Block 5: input is 32 x 32 x (64 * 1)
    x = layers.Conv2DTranspose(32, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False,  name='conv_transpose_5')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_5')(x)
    x = layers.ReLU(name='relu_5')(x)
       
     # Block 6: input is 64 x 64 x (32)
    outputs = layers.Conv2DTranspose(3, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, activation='tanh', name='conv_transpose_6')(x)
    # Output: output 128 x 128 x 3
    model = tf.keras.Model(inputs, outputs, name="Generator")
    return model

# Discriminator function
def discriminator1():
    
    inputs = keras.Input(shape=(128, 128, 3), name='input_layer')
    
     # Block 0: input is 112 x 112 x (3)
    x = layers.Conv2D(32, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_0')(inputs)
    x = layers.LeakyReLU(0.2, name='leaky_relu_0')(x)
    
    
    
    # Block 1: input is 64 x 64 x (3)
    x = layers.Conv2D(64, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_1')(inputs)
    x = layers.LeakyReLU(0.2, name='leaky_relu_1')(x)
    
    # Block 2: input is 32 x 32 x (64)
    x = layers.Conv2D(64 * 2, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_2')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_2')(x)
    
    # Block 3: input is 16 x 16 x (64*2)
    x = layers.Conv2D(64 * 4, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_3')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_3')(x)
  
    # Block 4: input is 8 x 8 x (64*4)
    x = layers.Conv2D(64 * 8, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_4')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_3')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_4')(x)
    
    # Block 5: input is 4 x 4 x (64*4)
    outputs = layers.Conv2D(1, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, activation='sigmoid', name='conv_5')(x)
    # Output: 1 x 1 x 1
    model = tf.keras.Model(inputs, outputs, name="Discriminator")
    return model

# Loss function
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Generator loss
def generator_loss(label, fake_output):
    gen_loss = binary_cross_entropy(label, fake_output)
    #print(gen_loss)
    return gen_loss

# Discriminator loss
def discriminator_loss(label, output):
    disc_loss = binary_cross_entropy(label, output)
    #print(total_loss)
    return disc_loss

'''
#--Generator Loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
'''
# Optimizer
learning_rate = 0.0002
generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999 )
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999 )

#--Use untrained generator to create an image
generator = generator1()
discriminator = discriminator1()
# Training the Discriminator and Generator Networks in Tandem
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".

@tf.function
def train_step(images):
    # noise vector sampled from normal distribution
    noise = tf.random.normal([BATCH_SIZE,1,1, latent_dim])

    # Train Discriminator with real labels
    with tf.GradientTape() as disc_tape1:
        generated_images = generator(noise, training=True)

        
        real_output = discriminator(images, training=True)
        real_targets = tf.ones_like(real_output)
        disc_loss1 = discriminator_loss(real_targets, real_output)
        
    # gradient calculation for discriminator for real labels    
    gradients_of_disc1 = disc_tape1.gradient(disc_loss1, discriminator.trainable_variables)
    
    # parameters optimization for discriminator for real labels   
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc1,\
    discriminator.trainable_variables))
    
    # Train Discriminator with fake labels
    with tf.GradientTape() as disc_tape2:
        fake_output = discriminator(generated_images, training=True)
        fake_targets = tf.zeros_like(fake_output)
        disc_loss2 = discriminator_loss(fake_targets, fake_output)
    # gradient calculation for discriminator for fake labels 
    gradients_of_disc2 = disc_tape2.gradient(disc_loss2, discriminator.trainable_variables)
    
    
    # parameters optimization for discriminator for fake labels        
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc2,\
    discriminator.trainable_variables))
    
    # Train Generator with real labels
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        real_targets = tf.ones_like(fake_output)
        gen_loss = generator_loss(real_targets, fake_output)

    # gradient calculation for generator for real labels     
    gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    
    # parameters optimization for generator for real labels
    generator_optimizer.apply_gradients(zip(gradients_of_gen,\
    generator.trainable_variables))  
        

OUT_DIR = "C:/DataSets/DCGAN/100X"
num_examples_to_generate = 1
image_sample_size=4

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, latent_dim, ])


#--Defining the training loop
EPOCHS = 6000
noise_dim = 100
#num_examples_to_generate = 4
image_sample_size=3000

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
print(seed.shape)

noise = tf.random.normal([1,1,1,100])
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

   

    # Save the model every 15 epochs
#    if (epoch + 1) % 50 == 0:
#         checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    GPUtil.showUtilization()
  save_images()
  
    
    
def save_images(directory=OUT_DIR):
    for k in range(image_sample_size):
        generated_image = generator(tf.random.normal([1,1,1, noise_dim]), training=False)
        f = str(k)+'.png'
        f = os.path.join(directory, f)
        img = np.array(generated_image)
        img = (img[0, :, :, :] + 1.) / 2.
        img = Image.fromarray((255*img).astype('uint8'))
        #img = Image.fromarray((255*img).astype('uint8').reshape((image_height,image_width,image_channels)))
        img.save(f,'PNG')
        #if k % 1000==0: print(k)
    print('Saved temporary images for evaluation.')
    

'''    
def generate_and_save_image(model, epoch, test_input):
  for i in range(image_sample_size):
   #noise = tf.random.normal([1,100])
   generated_image = generator(noise, training=False)
   print(generated_image.shape)
   plt.imshow(generated_image[0, :, :, :])
   plt.savefig('image_at_epoch_{:04d}.png'.format(i))
'''

'''
#--Generate and save images
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 1))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 1, i+1)
      #plt.imshow(predictions[i, :, :,  0] , cmap='RdPu')
      plt.imshow(predictions[i, :, :, :] * 255 + 255, cmap='twilight_shifted')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  '''
#--Train the model
train(normalized_ds, EPOCHS)

#--Restore the latest checkpoint
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

'''
#--Create a GIF
# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)
'''

