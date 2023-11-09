import cv2

# vid = cv2.VideoCapture(0)
# while(True):
#     ret,frame = vid.read()
#     cv2.imshow('frame',frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# vid.release()
# cv2.destroyAllWindows() 

#yüz tanıma
# import cv2
# cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# video_capture = cv2.VideoCapture(0)
# while True:
#     check,frame = video_capture.read()
#     gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     face = cascade.detectMultiScale(
#         gray_image ,scaleFactor=2.0,minNeighbors=4)
#     for x,y,w,h in face:
#         image = cv2.rectangle(frame,(x,y),(x+w,y+h),
#                               (0,255,0),3)
#         image[y:y+h, x:x+w] = cv2.medianBlur(image[y:y+h,x:x+w],
#                                              35)
        
#     cv2.imshow('face blurred',frame)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# video_capture.release()
# cv2.destroyAllWindows()

# import numpy as np
# from numpy import array
# import matplotlib.pyplot as plt



# im_1 = cv2.imread("kangal.jpg",0)
# ar = array(im_1)
# print(ar)



# freq, bins, patches = plt.hist(ar, edgecolor='white', label='d', bins=range(1,101,10))

# # x coordinate for labels
# bin_centers = np.diff(bins)*0.5 + bins[:-1]

# n = 0
# for fr, x, patch in zip(freq, bin_centers, patches):
#   height = int(freq[n])
#   plt.annotate("{}".format(height),
#                xy = (x, height),             # top left corner of the histogram bar
#                xytext = (0,0.2),             # offsetting label position above its bar
#                textcoords = "offset points", # Offset (in points) from the *xy* value
#                ha = 'center', va = 'bottom'
#                )
#   n = n+1

# plt.legend()
# plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("kangal.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

titles = ['original image', 'global thresholding (v = 127)','adaptive mean thresholding','adaptive gaussian thresholding']
images = [img,th1,th2,th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

