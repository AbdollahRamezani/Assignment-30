import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time
from TFLiteFaceDetector import UltraLightFaceDetecion   
from TFLiteFaceAlignment import CoordinateAlignmentModel  

fd = UltraLightFaceDetecion("weights/RFB-320.tflite",conf_threshold=0.88)
fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")

image = cv2.imread("input/face.png")

color = (0, 0, 255)  # رنگ قرمز


boxes, scores = fd.inference(image)

def zoom(image, point):
    for pred in fa.get_landmarks(image, boxes):  
        # for i,p in enumerate(np.round(pred).astype(np.int)):   
        #     cv2.circle(image, tuple(p), 2, color, -1) 
        #     cv2.putText(image, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)) 

        landmarks = []
        for i in point:
            landmarks.append(pred[i])  
        landmarks = np.array(landmarks, dtype=int)   
        # print(landmarks)    

        x, y, w, h = cv2.boundingRect(landmarks)  
        mask = np.zeros(image.shape, dtype=np.uint8)   
        cv2.drawContours(mask, [landmarks], -1, (255, 255, 255), -1)  
        
        mask = mask // 255

        black_mask = image * mask   
        black_mask = black_mask[y:y+h, x:x+w]  
        z = 2  #ضریب
        
        result_big = cv2.resize(black_mask, (0, 0), fx=z, fy=z)  
        return result_big, x ,y, w, h, landmarks
    # cv2.drawContours(image, [lips_landmarks], -1, (0, 0, 0), -1) 

lip = [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]
left_eye  = [39, 37, 33, 36, 35, 41, 40, 42]
right_eye = [95, 94, 96, 93, 91, 87, 90, 89]

result_big, x, y, w, h, landmarks = zoom(image, lip)             
y = y- (h//2)
x = x - (w//2)                    
image[y:y+h*2, x:x+w*2] = np.add(image[y:y+h*2, x:x+w*2], result_big)

result_big, x, y, w, h, landmarks = zoom(image, left_eye)             
y = y- (h//2)
x = x - (w//2)    
cv2.drawContours(image, [landmarks], -1, (0, 0, 0), -1)                 
image[y:y+h*2, x:x+w*2] = np.add(image[y:y+h*2, x:x+w*2], result_big)

result_big, x, y, w, h, landmarks = zoom(image, right_eye)             
y = y- (h//2)
x = x - (w//2)    
cv2.drawContours(image, [landmarks], -1, (0, 0, 0), -1)                 
image[y:y+h*2, x:x+w*2] = np.add(image[y:y+h*2, x:x+w*2], result_big)


                    
cv2.imshow("result", image)
cv2.imwrite(f"output/result.jpg", image)
cv2.waitKey() 

  