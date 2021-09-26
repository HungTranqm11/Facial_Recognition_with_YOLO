# import tensorflow as tf 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import random
import numpy as np
import cv2
import pickle


def process_image(img_path):
  img = cv2.imread(img_path) / 255.0
  img = cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC)
  return img

def pdist(vectors):
    distance_matrix = -2 * tf.tensordot(vectors, tf.transpose(vectors),axes=1) + tf.reduce_sum(tf.math.pow(vectors,2),1) + tf.transpose(tf.reduce_sum(tf.math.pow(vectors,2),1))
    return distance_matrix


def all_diffs(a, b):
    # Returns a tensor of all combinations of a - b
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)

def euclidean_dist(embed1, embed2):
    # Measures the euclidean dist between all samples in embed1 and embed2
    
    diffs = all_diffs(embed1, embed2) # get a square matrix of all diffs
    return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)

TL_MARGIN = 0.5 # The minimum distance margin
def bh_triplet_loss(dists, labels):
    # Defines the "batch hard" triplet loss function.
    
    same_identity_mask = tf.equal(tf.expand_dims(labels, axis=1),
                                  tf.expand_dims(labels, axis=0))
    negative_mask = tf.logical_not(same_identity_mask)
    positive_mask = tf.logical_xor(same_identity_mask,
                                   tf.eye(tf.shape(labels)[0], dtype=tf.bool))

    furthest_positive = tf.reduce_max(dists*tf.cast(positive_mask, tf.float32), axis=1)
    closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),
                                (dists, negative_mask), tf.float32)
    
    diff = furthest_positive - closest_negative
    
    return tf.maximum(diff + TL_MARGIN, 0.0)

EMBEDDING_DIM = 8 # Size of the embedding dimension (units in the last layer)
def embedImages(Images):
    conv1 = tf.layers.conv2d(Images,
                             filters=64, kernel_size=(3, 3),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer)
    
    pool1 = tf.layers.max_pooling2d(conv1,
                                    pool_size=(2, 2), strides=(1, 1),
                                    padding='same')
    
    conv2 = tf.layers.conv2d(pool1,
                             filters=128, kernel_size=(3, 3),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer)
    
    pool2 = tf.layers.max_pooling2d(conv2,
                                    pool_size=(2, 2), strides=(1, 1),
                                    padding='same')
    
    flat = tf.layers.flatten(pool2)
    
    # Linear activated embeddings
    embeddings = tf.layers.dense(flat,
                                 activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer,
                                 units=EMBEDDING_DIM)
    
    return embeddings

# Placeholders for inserting data
Images = tf.placeholder(tf.float32,[None, 28, 28, 3])
Labels = tf.placeholder(tf.int32,[None])

# Embeds images using the defined model
embedded_images = embedImages(Images)

# Measure distance between al embeddings
# dists = pdist(embedded_images)

# Calculate triplet loss for the give dists
# loss = tf.reduce_mean(bh_triplet_loss(dists, Labels))

# global_step = tf.Variable(0, trainable=False, name='global_step')
# learning_rate = tf.train.exponential_decay(0.001, global_step, 5000, 0.96, staircase=True)

# optimizer = tf.train.AdamOptimizer(learning_rate)
# train_step = optimizer.minimize(loss=loss, global_step=global_step)

"""## Training the model for 5000 epochs
Now, we feed the model some data for some time and later visualize the result embeddings with t-SNE.
"""

# from tqdm import tqdm

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

# saver = tf.train.import_meta_graph('/Users/tramlam/GitHub/Facial_Recognition_with_YOLO/saved_model/my-model.meta')
# tf.keras.backend.set_session(sess)
# sess.run(tf.global_variables_initializer())
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
saver.restore(sess, '/Users/tramlam/GitHub/Facial_Recognition_with_YOLO/saved_model_1/my-model-1' )

test_img = process_image('/Users/tramlam/GitHub/Facial_Recognition_with_YOLO/Tram_YOLO_image14.jpg')

f = open('./saved_model/embeddings_dict', 'rb')
embeddings_dict = pickle.load(f)
f.close()

variable = tf.trainable_variables()
# print(sess.run(variable[0]))

def retrieval(query_image):
    img = query_image / 255.0
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC)
    query_embeddings = sess.run(embedded_images,
                              feed_dict={Images: img.reshape(-1,28,28,3)})
    
    query_distances = []
    for i in range(10):
        query_distance = tf.squeeze(euclidean_dist(query_embeddings, embeddings_dict[i]))
        min_distance_index = tf.math.argmin(query_distance) #dis between the query image and embeddings of class i
        query_distances.append(tf.keras.backend.eval(query_distance[min_distance_index]))

    return np.argmin(np.array(query_distances))

# print(retrieval(test_img))






# embeddings_dict[0]

# embeddings_dict[0]

# saver.restore(sess,'my-model')




