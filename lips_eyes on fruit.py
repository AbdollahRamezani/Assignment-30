import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time
from TFLiteFaceDetector import UltraLightFaceDetecion   #از فایل دیگه ایمپورت شده
from TFLiteFaceAlignment import CoordinateAlignmentModel  #از فایل دیگه ایمپورت شده

fd = UltraLightFaceDetecion("weights/RFB-320.tflite",conf_threshold=0.88)
fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")

image = cv2.imread("input/face.png")
image_fruit = cv2.imread("input/fruit.jpg")

color = (0, 0, 255)  # رنگ قرمز


boxes, scores = fd.inference(image)

def zoom(image, point):
    for pred in fa.get_landmarks(image, boxes):  # به ازای هر تعداد چهره تکرار میشود
        # for i,p in enumerate(np.round(pred).astype(np.int)):   #به تعداد لندمارکهای روی چهره تکرار میشود #enumerate اندیس برمیگرداند
        #     cv2.circle(image, tuple(p), 2, color, -1) #منفی یک باعث میشود دایره توپر شود
        #     cv2.putText(image, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)) # چاپ اندیس

        landmarks = []
        for i in point:
            landmarks.append(pred[i])  #مختصات ایکس و وای نقاط خط قبل را به لیست لیپ لند مارکس اپند میکنیم
        landmarks = np.array(landmarks, dtype=int)   # اینجا باید اینت ساده باشه    
        # print(landmarks)    

        x, y, w, h = cv2.boundingRect(landmarks)  #ـ مختصات دور لب ـ مختصات نقاطی که بهش دادیم را برمیگرداند
        mask = np.zeros(image.shape, dtype=np.uint8)   # اینجا باید np.uint8 باشه
        cv2.drawContours(mask, [landmarks], -1, (255, 255, 255), -1)  
        
        mask = mask // 255

        black_mask = image * mask   # تصویر لب با حاشیه سیاه
        black_mask = black_mask[y:y+h, x:x+w]  # برای اینکه لب را بزرگ کنیم دور لب را کراپ میکنیم
        z = 2  #ضریب
        
        result_big = cv2.resize(black_mask, (0, 0), fx=z, fy=z)  #اندازه خاصی بهش نمیدیم فقط میگیم هرچی که هست سه برابر شود
        return result_big, x ,y, w, h, z, landmarks
    # cv2.drawContours(image, [lips_landmarks], -1, (0, 0, 0), -1) 

lip = [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]
left_eye  = [39, 37, 33, 36, 35, 41, 40, 42]
right_eye = [95, 94, 96, 93, 91, 87, 90, 89]

result_big, x, y, w, h,z, landmarks = zoom(image, lip)             
# y = y- (h//2)
# x = x - (w//2)                    
image_fruit[y:y+h*2, x:x+w*2] = np.add(image_fruit[y:y+h*2, x:x+w*2], result_big)

result_big, x, y, w, h, z, landmarks = zoom(image, left_eye)             
y = y- (h//z)
x = x - (w//z)                 
image_fruit[y:y+h*z, x:x+w*z] = np.add(image_fruit[y:y+h*z, x:x+w*z], result_big)

result_big, x, y, w, h, z, landmarks = zoom(image, right_eye)             
y = y- (h//z)
x = x - (w//z)           
image_fruit[y:y+h*z, x:x+w*z] = np.add(image_fruit[y:y+h*z, x:x+w*z], result_big)


                    
cv2.imshow("result", image_fruit)
cv2.imwrite(f"output/result_fruit.jpg", image_fruit)
cv2.waitKey() 

  