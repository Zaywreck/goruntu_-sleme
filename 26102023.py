import numpy as np
import cv2

img = cv2.imread("kangal.jpg")
cv2.imshow("orjinal",img)

blur = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)

cv2.imshow("gausblur",blur)

# blur = cv2.blur(img,(9,9))
# cv2.imshow("blur",blur)

kernel = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])

output = cv2.filter2D(blur,-1,kernel)
cv2.imshow("output",output)

# median = cv2.medianBlur(output,5)
# cv2.imshow("median",median)


# img = cv2.imread('sehir.jpg',0)
# img = cv2.resize(img,(480,320))
# cv2.imshow("image",img)

# equ = cv2.equalizeHist(img)
# cv2.imshow("negatif goruntu",equ)

cv2.waitKey(0)