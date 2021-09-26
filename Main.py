import cv2
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import keras
import numpy as np


MODEL = 'D:\Python\ML\YOLO\yolov3-face.cfg'
WEIGHT = 'D:\Python\ML\YOLO\yolov3-wider_16000.weights'
net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
IMG_WIDTH, IMG_HEIGHT = 416, 416

cap = cv2.VideoCapture(0)
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0
base_name = 'YOLO_image'
flag_1 = time.time()
flag_2 = time.time()
count = 0

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    # new_frame_time = time.time()
 
    # # Calculating the fps
 
    # # fps will be number of frame processed in given time frame
    # # since their will be most of time error of 0.001 second
    # # we will be subtracting it to get more accurate result
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # # print(fps)
    # # fps = 1/(new_frame_time-prev_frame_time)
    # # prev_frame_time = new_frame_time
 
    # # converting the fps into integer
    # # fps = int(fps)
 
    # # converting the fps to string so that we can display it on frame
    # # by using putText function
    # # fps = str(fps)

    # # putting the FPS count on the frame
    # cv2.putText(frame, str(fps), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # frame is now the image capture by the webcam (one frame of the video)
    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(frame, 
                                1/255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)

    blobb = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3])
    # print(blobb.shape)

    # plt.figure(figsize=(30, 30))
    # plt.imshow(blobb, cmap='gray')
    # plt.axis('off')
    # plt.show()

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

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
    result = frame.copy()
    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)

        # Extract position data
        # left = box[0]
        # top = box[1]
        # width = box[2]
        # height = box[3]
        topleft_x = box[0]
        topleft_y = box[1]
        width = box[2]
        height = box[3]

        bottomright_x = topleft_x + width
        bottomright_y = topleft_y + height
        # Draw bouding box with the above measurements
        ### YOUR CODE HERE
        
        cv2.rectangle(result, (int(topleft_x), int(topleft_y)), (int(bottomright_x), int(bottomright_y)), (255,0,0), 2)
        face = frame[topleft_y:bottomright_y, topleft_x:bottomright_x]
        # print((int(topleft_x), int(topleft_y))) 
        # print((int(bottomright_y), int(bottomright_x)))
        # Display text about confidence rate above each box
        text = f'{confidences[i]:.2f}'
        cv2.putText(result, text, (int(topleft_x) - 10, int(topleft_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
        color=(255,255,255), thickness=2)
        ### YOUR CODE HERE

    # Display text about number of detected faces on topleft corner
    # YOUR CODE HERE
    cv2.imshow('face detection', result)
    c = cv2.waitKey(1)
		# Break when pressing ESC
    if c == 27:
        break
        # Break when pressing ESC
    flag_1 = time.time()
    if flag_1 - flag_2 >3:
        cv2.imwrite('D:\Python\ML\YOLO\sample'+ base_name + str(count)+'.jpg', face)
        count +=1
        flag_2 = flag_1
        if count == 100:
            break
    elif c == 83:
        cv2.imwrite('YOLO_bdb.jpg', face)
        break
cap.release()
cv2.destroyAllWindows()


# 
