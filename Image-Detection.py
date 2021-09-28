import cv2
import matplotlib.pyplot as plt
import yolo_face_recognition
from yolo_face_recognition import retrieval
import time 

MODEL = '/Users/tramlam/GitHub/Facial_Recognition_with_YOLO/yolov3-face.cfg'
WEIGHT = '/Users/tramlam/GitHub/Facial_Recognition_with_YOLO/yolov3-wider_16000.weights'
index_to_class_name = {0:'Hung',1:'Tram',2:'Thai',3:'Serena_Williams',4:'Roberto_Carlos',5:'Roger_Federer',6:'Yasser_Arafat',7:'Yao_Ming',8:'Vladimir_Putin',9:'Yashwant_Sinha'}
net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
IMG_WIDTH, IMG_HEIGHT = 416, 416

filename = '/Users/tramlam/GitHub/Facial_Recognition_with_YOLO/saved_model/Thai_01.jpg'
frame = cv2.imread(filename)
blob = cv2.dnn.blobFromImage(frame,1/255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)

# Set model input
net.setInput(blob)

# Define the layers that we want to get the outputs from
output_layers = net.getUnconnectedOutLayersNames()

# Run 'prediction'
outs = net.forward(output_layers)

blobb = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3])


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
            topleft_y =  int(center_y - height / 2) #YOUR CODE HERE
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


    cv2.rectangle(result, (int(topleft_x), int(topleft_y)), (int(bottomright_x), int(bottomright_y)), (255,0,0), 2)
    face = frame[topleft_y:bottomright_y, topleft_x:bottomright_x]
    # Display text about confidence rate above each box

    index = retrieval(face)
    text = f'{index_to_class_name[index]} {confidences[i]:.2f}'
    cv2.putText(result, text, (int(topleft_x) - 10, int(topleft_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255,255,255), thickness=2)

    # Display text about number of detected faces on topleft corner

cv2.imwrite(filename+'detected.jpg', result)



