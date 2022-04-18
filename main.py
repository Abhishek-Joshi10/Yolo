# Loading the libraries
import cv2
import numpy as np

# Loading YOLO version 3 using downloaded weights
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# Loading classes which YOLOv3 can identify
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# Defining Layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# To identify object detection

# colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading the image
img = cv2.imread("sample_image5.jpeg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
# Original Size of image
height, width, channels = img.shape

# Converting the image into blob so that our algorithm is able detect objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
# 0.00392 is scaling factor, 416x416 is size, (0,0,0) is color, swap rgb true so image is black and white. crop = false

# To see the blob images
#for b in blob:
   # for n, img_blob in enumerate(b):
   #     cv2.imshow(str(n), img_blob)

# Lets pass this blob image into YOLO
net.setInput(blob)
# Forwarding it to output layer for results
outs = net.forward(output_layers)

# To see results in image itself
# Defining empty arrays to show confidence, boxes and class_ids
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]  #
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5: #
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2) # circle is green as rgb, only g is 255
            # Rectangle Coordinates instead of circle
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Putting the labels into image
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #
# print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), font, 2, color, 1)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


