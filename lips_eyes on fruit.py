import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time
from TFLiteFaceDetector import UltraLightFaceDetecion   
from TFLiteFaceAlignment import CoordinateAlignmentModel 

fd = UltraLightFaceDetecion("weights/RFB-320.tflite",conf_threshold=0.88)
fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")

image = cv2.imread("input/Abdullah-Ramezani.jpg")
image_fruit = cv2.imread("input/fruit.jpg")
fix_image_fruit = cv2.resize(image_fruit, [image.shape[1], image.shape[0]])

boxes, scores = fd.inference(image)

z = 2  #ضریب 

def zoom(image, fix_image_fruit, point):
    for pred in fa.get_landmarks(image, boxes):  
        # for i,p in enumerate(np.round(pred).astype(np.int)):   
        #     cv2.circle(image, tuple(p), 2, color, -1) 
        #     cv2.putText(image, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)) # چاپ اندیس

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
        mask = mask[y:y+h, x:x+w]    
       
        black_mask_big = cv2.resize(black_mask, (0, 0), fx=z, fy=z)
        mask_big = cv2.resize(mask, (0, 0), fx=z, fy=z)  

        black_mask_big_image = np.zeros(image.shape, dtype=np.uint8)
        mask_big_image = np.zeros(image.shape, dtype=np.uint8)

        x = x - (w//2)
        y = y- (h//2)

        black_mask_big_image[y:y+h*2, x:x+w*2] = black_mask_big
        mask_big_image[y:y+h*2, x:x+w*2] = mask_big

        result = black_mask_big_image + fix_image_fruit *(1 - mask_big_image)

        return result

lip = [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]
left_eye  = [39, 37, 33, 36, 35, 41, 40, 42]
right_eye = [95, 94, 96, 93, 91, 87, 90, 89]

result = zoom(image, fix_image_fruit, lip)             
result = zoom(image, result, left_eye)             
result = zoom(image,result, right_eye)             
                
cv2.imshow("result", result)
cv2.imwrite("output/result_lip_eyes on fruit.jpg", result)
cv2.waitKey() 

  
