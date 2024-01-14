# import cv2
# import numpy as np

# img = cv2.imread("renk.png")
# img = cv2.resize(img, (450, 300))
# cv2.imshow("orijinal", img)

# # Sadece kırmızı renk kanalını vurgula
# img_r = np.zeros_like(img)
# img_r[:, :, 2] = img[:, :, 2]  # Kırmızı kanalı orijinal renk kanalından al

# cv2.imshow("kirmizi", img_r)

# # Sadece yeşil renk kanalını vurgula
# img_g = np.zeros_like(img)
# img_g[:, :, 1] = img[:, :, 1]  # Yeşil kanalı orijinal renk kanalından al

# cv2.imshow("yesil", img_g)

# # Sadece mavi renk kanalını vurgula
# img_b = np.zeros_like(img)
# img_b[:, :, 0] = img[:, :, 0]  # Mavi kanalı orijinal renk kanalından al

# cv2.imshow("mavi", img_b)

# import numpy as np
# import cv2

# def translate_image(image, tx, ty):
#     rows, cols = image.shape

#     # Ötelenmiş görüntü için boş bir matris oluştur
#     translated_image = np.zeros_like(image, dtype=np.uint8)

#     # Öteleme işlemi
#     for i in range(rows):
#         for j in range(cols):
#             new_i = i + ty
#             new_j = j + tx

#             # Sınırları kontrol et
#             if 0 <= new_i < rows and 0 <= new_j < cols:
#                 translated_image[new_i, new_j] = image[i, j]

#     return translated_image

# # Görüntüyü oku
# img = cv2.imread('kangal.jpg', 0)
# cv2.imshow('image', img)

# # Görüntüyü 100 birim sağa, 50 birim aşağı ötele
# translated_img = translate_image(img, 100, 50)
# cv2.imshow('Translation', translated_img)


# img = cv2.imread("kaplan.jpg")
# img = cv2.resize(img,(450,340))
# cv2.imshow("orijina",img)
# img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

# image = img
# h = image[:,:,2]
# zeros_h = np.zeros_like(img)
# zeros_h[:,:,2] = img[:,:,0]
# s = image[:,:,1]
# zeros_s = np.zeros_like(img)
# zeros_s[:,:,1] = img[:,:,1]
# v = image[:,:,0]
# zeros_v = np.zeros_like(img)
# zeros_v[:,:,0] = img[:,:,2]

# cv2.imshow("h",h)
# cv2.imshow("s",s)
# cv2.imshow("v",v)

# cv2.imshow("oh",zeros_h)
# cv2.imshow("os",zeros_s)
# cv2.imshow("ov",zeros_v)
# cv2.waitKey(0)


# from sklearn import preprocessing as p
# img=cv2.imread('sehir.jpg',0)
# img = cv2.resize(img, (480, 320))
# cv2.imshow("Orjinal Görüntü",img)
# mn=np.min(img)
# mx=np.max(img)
# im1=(img-mn)/(mx-mn)
# cv2.imshow("Kontrast Germe-1",im1)
# min_max_scaler = p.MinMaxScaler()
# normalizedData = min_max_scaler.fit_transform(img)
# cv2.imshow("Kontrast Germe-2",normalizedData)


# img = cv2.imread("sehir.jpg",0)
# cv2.imshow("orijinal",img)
# negatif =  np.max(img) - img
# cv2.imshow("a",negatif) 
# cv2.imshow("a",~img)

# gamma=1
# gamma_corrected = np.array(255*(img / 255)
# ** gamma, dtype = 'uint8')
# cv2.imshow('log_image-3',gamma_corrected )


# from matplotlib import pyplot as plt
# img=cv2.imread('kangal.jpg')
# img = cv2.resize(img, (400, 350))
# cv2.imshow("Orjinal Image",img)
# blur = cv2.blur(img,(5,5))
# cv2.imshow("Blur Image",blur)





import cv2
import numpy as np

# Görüntüyü oku
img = cv2.imread("kangal.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original Image", img)

# Sobel Gradient
rows, cols = img.shape
gradient_x = np.zeros_like(img, dtype=np.float32)
gradient_y = np.zeros_like(img, dtype=np.float32)

for i in range(1, rows-1):
    for j in range(1, cols-1):
        gradient_x[i, j] = img[i-1, j-1] - img[i-1, j+1] + 2*(img[i, j-1] - img[i, j+1]) + img[i+1, j-1] - img[i+1, j+1]
        gradient_y[i, j] = img[i-1, j-1] + 2*img[i-1, j] + img[i-1, j+1] - img[i+1, j-1] - 2*img[i+1, j] - img[i+1, j+1]

magnitude_gradient = np.sqrt(gradient_x**2 + gradient_y**2)

cv2.imshow("Gradient X", cv2.convertScaleAbs(gradient_x))
cv2.imshow("Gradient Y", cv2.convertScaleAbs(gradient_y))
cv2.imshow("Magnitude Gradient", cv2.convertScaleAbs(magnitude_gradient))

# Laplacian
laplacian = np.zeros_like(img, dtype=np.float32)

for i in range(1, rows-1):
    for j in range(1, cols-1):
        laplacian[i, j] = img[i-1, j] + img[i, j-1] - 4*img[i, j] + img[i, j+1] + img[i+1, j]

cv2.imshow("Laplacian", cv2.convertScaleAbs(laplacian))

cv2.waitKey(0)
cv2.destroyAllWindows()






