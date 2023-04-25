import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time
from TFLiteFaceDetector import UltraLightFaceDetecion   #از فایل دیگه ایمپورت شده
from TFLiteFaceAlignment import CoordinateAlignmentModel  #از فایل دیگه ایمپورت شده

fd = UltraLightFaceDetecion("weights/RFB-320.tflite",conf_threshold=0.88)
fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")

image = cv2.imread("input/Abdullah-Ramezani.jpg")

boxes, scores = fd.inference(image)

z = 1  #ضریب 

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
        black_mask = black_mask[y:y+h, x:x+w]  # کراپ کردن مستطیلی دور لب از تصویر لب با حاشیه سیاه
        mask = mask[y:y+h, x:x+w]    # کراپ کردن مستطیلی به اندازه دور لب از تصویر سیاه خالی
       
        black_mask_big = cv2.resize(black_mask, (0, 0), fx=z, fy=z)#بزرگ شده تصویر کراپ مستطیلی از دور لب کوچک حاشیه سیاه
        mask_big = cv2.resize(mask, (0, 0), fx=z, fy=z)  #بزرگ شده تصویر کراپ مستطیلی به اندازه دور لب از تصویر سیاه خالی

        black_mask_big = cv2.flip(black_mask_big, 0)  #چرخش 180 درجه
        mask_big = cv2.flip(mask_big, 0)

        black_mask_big_image = np.zeros(image.shape, dtype=np.uint8)
        mask_big_image = np.zeros(image.shape, dtype=np.uint8)

        black_mask_big_image[y:y+h*z, x:x+w*z] = black_mask_big
        mask_big_image[y:y+h*z, x:x+w*z] = mask_big

        result = black_mask_big_image + image *(1 - mask_big_image)

        return result

lip = [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]
left_eye  = [39, 37, 33, 36, 35, 41, 40, 42]
right_eye = [95, 94, 96, 93, 91, 87, 90, 89]

result = zoom(image, lip)             
result = zoom(result, left_eye)             
result = zoom(result, right_eye)  

result = cv2.flip(result, 0)   

cv2.imshow("result", result)
cv2.imwrite("output/result_rotate.jpg", result)
cv2.waitKey() 

  