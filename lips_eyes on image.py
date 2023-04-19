import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time
from TFLiteFaceDetector import UltraLightFaceDetecion   
from TFLiteFaceAlignment import CoordinateAlignmentModel  

fd = UltraLightFaceDetecion("weights/RFB-320.tflite",conf_threshold=0.88)
fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")

image = cv2.imread("input/face.webp")

color = (0, 0, 255)  # رنگ قرمز


boxes, scores = fd.inference(image)

for pred in fa.get_landmarks(image, boxes):  
    # for i,p in enumerate(np.round(pred).astype(np.int)):   
    #     cv2.circle(image, tuple(p), 2, color, -1) 
    #     cv2.putText(image, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)) # چاپ اندیس

    lips_landmarks = []
    for i in [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]:
        lips_landmarks.append(pred[i]) 
    lips_landmarks = np.array(lips_landmarks, dtype=int)   
    # print(lips_landmarks)    

    x, y, w, h = cv2.boundingRect(lips_landmarks)  
    mask = np.zeros(image.shape, dtype=np.uint8)   
    cv2.drawContours(mask, [lips_landmarks], -1, (255, 255, 255), -1)  
     
    mask = mask // 255

    lips = image * mask   
    lips = lips[y:y+h, x:x+w]  
    z = 2  #ضریب
    
    big_lips = cv2.resize(lips, (0, 0), fx=z, fy=z)  
    
    cv2.drawContours(image, [lips_landmarks], -1, (0, 0, 0), -1) 
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
            cv2.resize(contour, (0,0), fx=z, fy=z)
        
    y = y- (h//2)
    x = x - (w//2)
    image[y:y+h*z, x:x+w*z] += big_lips
                     
    cv2.imshow("result", image)
    #cv2.imwrite(f"output/result000{pred[i]}.jpg", result_big)
    cv2.waitKey() 

  
