import cv2
import numpy as np

img = np.array([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]], dtype='uint8')

# Calculate the standard deviation
# Here I'm taking the full image, you can take any rectangular region
# Method-1: using cv2.meanStdDev()
mean, std_1 = cv2.meanStdDev(img, mask=None)

# Method-2: using the formulae 1/n(S2 - (S1**2)/n)
sum_1, sqsum_2 = cv2.integral2(img)
n = img.size
# sum of the region can be easily found out using the integral image as
#  Sum = Bottom right + top left - top right - bottom left
s1 = sum_1[-1,-1]
s2 = sqsum_2[-1,-1]
std_2 = np.sqrt((s2 - (s1**2)/n)/n)
print(std_1, std_2)  # [[0.45825757]] 0.4582575694