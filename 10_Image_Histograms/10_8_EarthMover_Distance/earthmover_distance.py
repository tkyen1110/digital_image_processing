import cv2
import numpy as np
import matplotlib.pyplot as plt

# image to signature for color image
def img_to_sig(img):
    sig = np.empty((img.size, 4), dtype=np.float32)
    idx = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                sig[idx] = np.array([img[i,j,k], i, j,k])
                idx += 1
    return sig

# Load the images
img1 = cv2.imread('app1.png')
img2 = cv2.imread('app2.png')

sig1 = img_to_sig(img1)
sig2 = img_to_sig(img2)
distance, lowerbound, flow_matrix = cv2.EMD(sig1, sig2, cv2.DIST_L1,lowerBound=0)