# Facial_Recognition_with_YOLO
This project is to use YOLO to detect human faces and feed the data for training. 

Regarding the multi-class classification problem, we attemped to use the method of image similarity using Triplet Loss. The idea is to learn distributed embeddings representation of data points in a way that in the high dimensional vector space, similar imgae data points are projected in the nearby region whereas disimilar data points are projected away from each other.

One big advantage of this so-called Few Shots Detection is that allowing us to use few samples for training, yet still get the good prediction.

# Requirements
YOLOv3, CV2, Tensorflow Keras, CNN

# Usage

## 1. YOLO
- Create your own face database
- As for YOLO, we uploaded the pre-trained model and weights used for human face dectection.

## 2. Training Model
- After training, we saved and uploaded the model weight and model architecture.