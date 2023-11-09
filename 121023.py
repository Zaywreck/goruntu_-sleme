import numpy as np
import cv2 

img = cv2.imread("sena.jpeg")
img1 = cv2.imread("sena.jpeg",0)

cv2.imshow("kangal",img)
cv2.waitKey(0)
cv2.imshow("gri Kangal",img1)
cv2.waitKey(0)
img2 = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
cv2.imshow("kangal b",img2)
cv2.waitKey(0)

image = img

r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]

gbr_img = cv2.merge((g,b,r))
cv2.imshow("şekilli kangal",gbr_img)
cv2.waitKey(0)