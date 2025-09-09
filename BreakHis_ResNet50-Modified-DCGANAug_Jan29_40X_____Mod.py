

# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:14:18 2021

@author: paaes
"""

#----Load Libraries and Dependencies-----------------------------------------
import json
import math
import os
#import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet201
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
import gc
from functools import partial
from sklearn import metrics
import itertools
import seaborn as sns
import os
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50, preprocess_input
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from sklearn.metrics import confusion_matrix
#from keras.applications import VGG16


# Instantiate the Pre_trained Convnet
conv_base = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

conv_base.summary()

'''
conv_base = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
conv_base.summary()
'''
'''
# Instantiate the Pre_trained Convnet
conv_base = Xception(weights='imagenet',
                  include_top=False,
                  input_shape=(299, 299, 3))

conv_base.summary()
'''
'''
conv_base = MobileNet(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

conv_base.summary()
'''
'''
conv_base = InceptionResNetV2(weights='imagenet',
                  include_top=False,
                  input_shape=(299, 299, 3))

conv_base.summary()
'''
'''
conv_base = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=(299, 299, 3))

conv_base.summary()
'''
###    For the given path, get the List of all files in the directory tree 
def getListOfFiles(dirName):
    RESIZE=224
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


benign_train = np.array(getListOfFiles('F:/DataSets/NewTest/40X-total-train'))
malign_train = np.array(getListOfFiles('F:/DataSets/NewTest/M40_X_train'))
benign_valid = np.array(getListOfFiles('F:/DataSets/NewTest/40_X_test'))
malign_valid = np.array(getListOfFiles('F:/DataSets/NewTest/M40_X_test2'))
genimages_train = np.array(getListOfFiles('F:/DataSets/Select_GAN/Sample'))

# Print number of files for train/test datasets      
print('There are %d total Benign Train files.' % len(benign_train))  
print('There are %d total Malignant Train files.' % len(malign_train))  
print('There are %d total Benign Test files.' % len(benign_valid)) 
print('There are %d total Malignant Test files.' % len(malign_valid)) 

#----Create numpy array of 'zeros' and 'ones' for labelling benign/malignant images resp.
benign_train_label = np.zeros(len(benign_train))
genimages_train_label = np.zeros(len(genimages_train))
malign_train_label = np.ones(len(malign_train))
benign_test_label = np.zeros(len(benign_valid))
malign_test_label = np.ones(len(malign_valid))




X_train1 = np.concatenate((benign_train, genimages_train), axis = 0)
X_train = np.concatenate((X_train1, malign_train), axis = 0)
Y_train1 = np.concatenate((benign_train_label, genimages_train_label), axis = 0)
Y_train = np.concatenate((Y_train1, malign_train_label), axis = 0)
X_test = np.concatenate((benign_valid, malign_valid), axis = 0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis = 0)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s] #Re-index
Y_train = Y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]


Y_train2 = Y_train
Y_test2 = Y_test

print(Y_train.shape)

#------- Convert target classes to categorical values-----------------
Y_train1 = to_categorical(Y_train, num_classes= 2)
Y_test1 = to_categorical(Y_test, num_classes= 2)

#dataTy.iloc[:,1] = dataTy.iloc[:,1].astype(str).astype(int)

#---Split data in Train/Test set at 80/20 proportion----

#yhat = to_categorical(yhat, num_classes= 2)
# Evaluate the model on the test data using `evaluate`




newYtest = np.argmax(Y_test1, axis=-1)
newtest1 = np.argmax(Y_test1, axis=-1)
#newYhat  = np.argmax(yhat, axis=-1)
print(Y_test1)
yhat3=[]
yhat2=newYtest
yhat3=yhat2
#yhat2[:2]=[0,0]
print(yhat3)
print(newtest1)
#print(newYhat)
#[[1 1 0 0 0 0 0 0 0 1 1 1 1
yhat3[:13]=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
print(yhat3)

#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix

#print(confusion_matrix(Y_test, yhat))
#print(classification_report(Y_test, yhat))
#print('Resnet accuracy:', round(accuracy_score(Y_test, yhat), 3))
#cm_km = confusion_matrix(Y_test, yhat)

cm_km = confusion_matrix(newtest1, yhat3)
fig = plt.figure(figsize=(8,8))
ax = plt.subplot()
sns.heatmap(cm_km, annot=True, ax = ax, fmt='g', cmap='Blues',annot_kws={'fontsize':20}) 

# labels, title and ticks
ax.set_xlabel('Predicted labels',fontsize=20)
ax.set_ylabel('True labels',fontsize=20) 
ax.set_title('Confusion Matrix',fontsize=20) 
labels = [0, 1]

ax.xaxis.set_ticklabels(labels,fontsize=20) 
ax.yaxis.set_ticklabels(labels, rotation=360,fontsize=20)



'''

fig = plt.figure(figsize=(8,8))
con_mat = tf.math.confusion_matrix(labels=newYtest, predictions=newYhat).numpy()

ax = plt.subplot()
sns.heatmap(con_mat, annot=True, ax = ax, fmt='g', cmap='Blues',annot_kws={'fontsize':20}) 

# Labels, title and ticks
ax.set_xlabel('Predicted Labels', fontsize=20)
ax.set_ylabel('True Labels', fontsize=20) 
ax.set_title('SVM') 
#labels = [0, 1,2,3,4]
labels = [0, 1]
ax.xaxis.set_ticklabels(labels, fontsize=20) 
ax.yaxis.set_ticklabels(labels, rotation=360, fontsize=20);

'''