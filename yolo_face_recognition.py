# import tensorflow as tf 

import tensorflow as tf
import pandas as pd
import random
import numpy as np
import cv2
import pickle
from tensorflow import keras
import tensorflow_addons as tfa
from statistics import mode

def all_diffs(a, b):
    # Returns a tensor of all combinations of a - b
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)

def euclidean_dist(embed1, embed2):
    # Measures the euclidean dist between all samples in embed1 and embed2
    
    diffs = all_diffs(embed1, embed2) # get a square matrix of all diffs
    return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)

train_labels = []
for i in range(10):
    train_labels += [i] * 10
index_to_class_name = {0:'Hung',1:'Tram',2:'Thai',3:'Serena_Williams',4:'Roberto_Carlos',5:'Roger_Federer',
6:'Yasser_Arafat',7:'Yao_Ming',8:'Vladimir_Putin',9:'Yashwant_Sinha'}
train_embeddings = np.load('/Users/tramlam/GitHub/Facial_Recognition_with_YOLO/train_embeddings.npy')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,3)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=None), # No activation on final dense layer
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings

])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tfa.losses.TripletSemiHardLoss(margin=0.2))

model.load_weights('/Users/tramlam/GitHub/Facial_Recognition_with_YOLO/model-weight/model-weight')

def retrieval(query_image):
    img = query_image/ 255.0
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC)
    query_embeddings = model(img.reshape(1,28,28,3))
    query_distance = tf.keras.backend.eval(tf.squeeze(euclidean_dist(query_embeddings, train_embeddings)))
    min_distance_index = query_distance.argsort()[:7] 
    output = np.array(train_labels)[min_distance_index]
    
    # return predict_labels(output, index_to_class_name)
    return index_to_class_name[mode(output)]

def process_image(img_path):
  img = cv2.imread(img_path) / 255.0
  img = cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC)
  return img

def all_diffs(a, b):
    # Returns a tensor of all combinations of a - b
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)

def euclidean_dist(embed1, embed2):
    # Measures the euclidean dist between all samples in embed1 and embed2
    diffs = all_diffs(embed1, embed2) # get a square matrix of all diffs
    return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)





