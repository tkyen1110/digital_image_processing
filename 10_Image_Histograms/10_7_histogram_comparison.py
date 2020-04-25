import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
img1 = cv2.imread('app1.png')
img2 = cv2.imread('app2.png')
 
# Convert it to HSV
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
 
# Calculate the histogram and normalize it
hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# find the metric value
metric_val_1 = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
metric_val_2 = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CHISQR)
metric_val_3 = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CHISQR_ALT)
metric_val_4 = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_INTERSECT)
metric_val_5 = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
metric_val_6 = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_HELLINGER)
metric_val_7 = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_KL_DIV)

print("metric_val of  HISTCMP_CORREL        = ", metric_val_1)
print("metric_val of  HISTCMP_CHISQR        = ", metric_val_2)
print("metric_val of  HISTCMP_CHISQR_ALT    = ", metric_val_3)
print("metric_val of  HISTCMP_INTERSECT     = ", metric_val_4)
print("metric_val of  HISTCMP_BHATTACHARYYA = ", metric_val_5)
print("metric_val of  HISTCMP_HELLINGER     = ", metric_val_6)
print("metric_val of  HISTCMP_KL_DIV        = ", metric_val_7)