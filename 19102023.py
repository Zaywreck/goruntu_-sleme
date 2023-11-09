import numpy as np 
import cv2 
from sklearn import preprocessing as p
img = cv2.imread("sehir.jpg")
gamma = 5
gamma_corrected = np.array(255*(img/255)**gamma,dtype='uint8')
cv2.imshow("gamma",gamma_corrected)



# im1 = img.astype(float)
# c = 255 / np.log(1 + np.max(im1))
# log_img = c*np.log(1+im1)
# log_img1 = (log_img).astype(np.uint8)
# cv2.imshow("1",log_img1)



# img = cv2.resize(img,(400,350))
# cv2.imshow("orijinal resim",img)

# mx = np.max(img)
# neg_img = 255-img
# cv2.imshow("negatif görüntü",neg_img)

# img = cv2.resize(img,(480,320))

# mn = np.min(img)
# mx = np.max(img)
# cv2.imshow("sehir",img)

# im1 = (img-mn)/(mx-mn)
# cv2.imshow("kontrast germe-1",im1)

# img_new = cv2.resize(img,(1920,1080))
# cv2.imshow("büyük sena",img_new),

# img_new = cv2.flip(img,0)
# img_new_1 = cv2.flip(img,1)
# img_new_2 = cv2.flip(img,-1)


# hsv_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
# xyz_img = cv2.cvtColor(img,cv2.COLOR_RGB2XYZ)
# lab_img = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
# ycbcr_img = cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
# L = lab_img[:,:,1]
# A = lab_img[:,1,:]
# B = lab_img[1,:,:]
# X = xyz_img[:,:,1]
# Y = xyz_img[:,1,:]
# Z = xyz_img[1,:,:]
# H = hsv_img[:,:,1]
# S = hsv_img[:,1,:]
# V = hsv_img[1,:,:]

# cropped_img = img[:100,:300]

# rows,cols = img.shape
# M = np.float32([[1,0,100],[0,1,50]])
# trans = cv2.warpAffine(img,M,(cols,rows))

# cv2.imshow("ycbcr",ycbcr_img)
# cv2.imshow("lab",lab_img)
# cv2.imshow("l",L)
# cv2.imshow("A",A)
# cv2.imshow("B",B)
# cv2.imshow("xyz",xyz_img)
# cv2.imshow("X",X)
# cv2.imshow("Y",Y)
# cv2.imshow("Z",Z)
# cv2.imshow("hsv",hsv_img)
# cv2.imshow("hsv 2 ", H)
# cv2.imshow("hsv 2 ", S)
# cv2.imshow("hsv 2 ", V)
# cv2.imshow("kırpılmış sena", cropped_img)
# cv2.imshow("değişik sena",trans)
# cv2.imshow("sena 0",img_new)
# cv2.imshow("sena 1",img_new_1)
# cv2.imshow("sena -1",img_new_2)
cv2.waitKey()
