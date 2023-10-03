from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
from sort import*
cap=cv2.VideoCapture("../Videos/cars.mp4")

model=YOLO("../Yolo_Weights/yolov8l.pt")
class_names={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
mask=cv2.imread("mask.png")

#creating an instance of Sort
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
Totalcount=[]

while True:
    success,img=cap.read()
    img_reg=cv2.bitwise_and(img,mask)#masking the video to detect cars only in a particular region
    results=model(img_reg,stream=True)

    detections=np.empty((0,5))
    for r in results:
        for box in r.boxes:
            cls_id = box.cls[0].item()
            cls_name=class_names[cls_id]
            conf=np.round(box.conf[0].item(),2)

            #detecting and putting rectangles over cars,trucks,rectangles
            if cls_name=="car" or cls_name=="motorcycle" or cls_name=="bus" or cls_name=="truck" and conf>0.3:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                currentdetection = np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentdetection))


    resultsTracker=tracker.update(detections)
    cv2.line(img,(400,297),(673,297),(0,0,255),5) #The line which if crossed we count the cars
    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w = x2 - x1
        h = y2 - y1 #We draw the bounding box using the corrdinates obtained from tracker results and put ID as text on the bound box
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, t=3,colorR=(255,0,255))
        cvzone.putTextRect(img, f"{id}", (max(0,x1), max(20,y1)), scale=2,thickness=3)

        cx,cy=x1+w//2,y1+h//2 #coordinates for the centre of each bounding box
        cv2.circle(img,(cx,cy),radius=4,color=(255,0,255), thickness=cv2.FILLED) #denoting it as a circle

        #counting logic
        if 400<cx<673 and 277<cy<317:
               if Totalcount.count(id)==0:
                   Totalcount.append(id)
                   cv2.line(img, (400, 297), (673, 297), (0, 255, 0), 5)
    cvzone.putTextRect(img, f"Count:{len(Totalcount)}", (50,50))

    cv2.imshow("WebLive",img)
    cv2.waitKey(1)