# Importing the libraries
from PIL import Image
import time
import base64
from io import BytesIO
import os
import random
import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

# img = cv2.imread(r'.\human_face\Roberto_Carlos\Roberto_Carlos_0001.jpg')
# print(img.shape)

MODEL = r'D:\Python\ML\yolov3-face.cfg'
WEIGHT = r'D:\Python\ML\yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
IMG_WIDTH, IMG_HEIGHT = 416, 416

human_face_path = r'.\human_face'
# for folder_name in os.listdir(human_face_path):
#     folder_path = human_face_path + '/' + folder_name

def process_image(image_path):

    img = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(img, 1/255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)
    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)

    blobb = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3], 1)

    frame_height = img.shape[0]
    frame_width = img.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.

    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
            # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with high confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                
                            # Find the top left point of the bounding box 
                topleft_x = int(center_x - width / 2)  #YOUR CODE HERE
                topleft_y = int(center_y - height / 2) #YOUR CODE HERE
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    final_boxes = []
    result = img.copy()
    for i in indices:
        i = i[0]
        box = boxes[i]
        # final_boxes.append(box)

        # Extract position data
        
        topleft_x = box[0]
        topleft_y = box[1]
        width = box[2]
        height = box[3]

        bottomright_x = topleft_x + width
        bottomright_y = topleft_y + height
        # Draw bouding box with the above measurements
        face = img[topleft_y:bottomright_y, topleft_x:bottomright_x]  
        
        return face
    
def read_images_from_folder(folder_path, k = 10):
    all_img = []
    for i,file_name in enumerate(os.listdir(folder_path)):
        file_path = folder_path + '/' + file_name
        img = process_image(file_path)
        all_img.append(img)
        if i >= k:
            break

    return all_img

def read_folders(folder_path):
    data = {}
    for i,child_folder in enumerate(os.listdir(folder_path)):
        child_path = folder_path + '/' + child_folder
        data[child_folder] = read_images_from_folder(child_path)

    return data

data = read_folders(human_face_path)

print('Have detected all faces!')

def write_image(img_array, img_path):
    cv2.imwrite(img_path, img_array)

save_base_path = './detected_face'

for key in data.keys():
    all_images = data[key]
    save_folder_path = save_base_path + '/' + key
    if os.path.exists(save_folder_path) == False:
        os.makedirs(save_folder_path)
    for i,img in enumerate(all_images):
        file_path = save_folder_path + '/' + str(i) + '.jpg'
        write_image(img,file_path)
print('Have written all detected faces!')
