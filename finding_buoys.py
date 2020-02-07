import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img=cv.imread('Buoy.jpeg')


##filters for pics
blur = cv.blur(img,(5,5))
blur0=cv.medianBlur(blur,5)
blur1= cv.GaussianBlur(blur0,(5,5),0)
blur2= cv.bilateralFilter(blur1,9,75,75)


##conversions of pics
hsv = cv.cvtColor(blur2, cv.COLOR_BGR2HSV)
gray = cv.cvtColor(blur2, cv.COLOR_BGR2GRAY) 
ret, thresh = cv.threshold(gray, 127, 255, 0)


##for yellow buoy
low_yellow = np.array([70, 240 , 50])
high_yellow = np.array([80, 255, 255])
mask1 = cv.inRange(hsv, low_yellow, high_yellow)


##for red buoy
low_red = np.array([100, 60, 50])
high_red = np.array([105, 255, 255])
mask2 = cv.inRange(hsv, low_red, high_red)


##for blue buoy
low_red = np.array([80, 50, 50])
high_red = np.array([87, 255, 255])
mask3 = cv.inRange(hsv, low_red, high_red)


##for adding the different detections
mask=mask1+mask2+mask3

_, contours, _=cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

##to check and show only contours having area more than 1500 units
print(str(len(contours)))
for contour in contours:
    area= cv.contourArea(contour)
    
    
    if area>1500:
     for c in contours:
   
        x, y, w, h = cv.boundingRect(c)
   
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
	
    print(area)
for c in contours:
        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
 
        cv.circle(img, (cX, cY), 5, (255, 255, 255), -1)
        print(cX,cY)     
 


 

cv.imshow('img',img)

cv.waitKey(0)
cv.destroyAllWindows()
