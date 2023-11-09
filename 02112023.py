import cv2 
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("kaplan.jpg")
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

cv2.imshow("orijinal", gray)


edges = cv2.Canny(image = gray,threshold1=100,threshold2=200)
cv2.imshow("canny", edges)


# sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0)
# sobely = cv2.Sobel(gray,cv2.CV_8U,0,1)
# sobelxy = cv2.Sobel(gray,cv2.CV_8U,1,1)


# cv2.imshow("Sobelx Gradient",sobelx)
# cv2.imshow("Sobely Gradient",sobely)
# cv2.imshow("Sobelxy Gradient",sobelxy)

# kernel = np.array([
#     [0,1,0],
#     [1,-4,1],
#     [0,1,0]
# ])

# output2 = cv2.filter2D(gray,-1,kernel)
# cv2.imshow("2.derivative", output2)

# lap1 = cv2.Laplacian(gray,cv2.CV_8U)
# cv2.imshow("Laplacian der2", lap1)
# lap2 = cv2.Laplacian(gray,cv2.CV_64F)
# lap3 = np.absolute(lap2)
# lap4 = np.uint(lap3)


# img = cv2.imread("kaplan.jpg")
# img = cv2.resize(img,(800,750))
# cv2.imshow("orijinal",img)
# # img = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)

# kernel_x = np.array([[-1,0,1]])
# kernel_y = np.array([[-1],[0],[1]])

# output_x = cv2.filter2D(img,-1,kernel_x)
# output_y = cv2.filter2D(img,-1,kernel_y)

# cv2.imshow("yatay",output_x)
# cv2.imshow("dikey",output_y)

cv2.waitKey(0)