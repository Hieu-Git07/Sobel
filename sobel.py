import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

# Author: ------------NguyenDinhTrongHieu--------------
##   /\_/\  
##  ( o.o ) 
##   > ^ < 


# Convert image to 2D vector (grayscale)
img = Image.open("leaf(rgb).png").convert("L")
img = np.array(img)

kernel_x = [
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
]

kernel_y = [
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
]


# Convolution
# Padding image (them bien = 0)
img_pad = np.pad(img, 1, mode='constant', constant_values=0)

grad_x = np.zeros_like(img, dtype=float)
grad_y = np.zeros_like(img, dtype=float)
grad_val = np.zeros_like(img, dtype=float)
new_edge_image = np.zeros_like(img, dtype=np.uint8)
new_edge_image_threshold = np.zeros_like(img, dtype=np.uint8)




# Convolution (tich chap vs kernel)
max_val = 0
prob = np.zeros_like(img, dtype = float)
for i in range(1, img_pad.shape[0]-1):
    for j in range(1, img_pad.shape[1]-1):
        grad_x[i-1][j-1] = np.sum(kernel_x * img_pad[i-1:i+2, j-1:j+2])
        grad_y[i-1][j-1] = np.sum(kernel_y * img_pad[i-1:i+2, j-1:j+2])
        grad_val[i-1][j-1] = math.sqrt(grad_x[i-1][j-1]**2 + grad_y[i-1][j-1]**2)
        max_val = max(max_val,grad_val[i - 1][j - 1])
        
# Hard Threshold 
threshold = 0
for i in range(img.shape[0] - 1):
    for j in range(img.shape[1] - 1):
        threshold += grad_val[i][j]
        
threshold /= (img.shape[0]) * (img.shape[1])

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if grad_val[i][j] < threshold:
            new_edge_image_threshold[i][j] = 0
        else:
            new_edge_image_threshold[i][j] = grad_val[i][j]


# Tim kiem max val de xu ly min-max scaling
for i in range(1, img_pad.shape[0]-1):
    for j in range(1, img_pad.shape[1] - 1):
        prob[i-1][j-1] = grad_val[i-1][j-1]/max_val


# Min-max scaling
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        new_edge_image[i][j] = prob[i][j] * 255



plt.figure(figsize=(12, 4))

# Image 1
plt.subplot(1, 3, 1)
plt.title("Gradient Value")
plt.imshow(grad_val, cmap='gray')
plt.axis('off')

# Image 2
plt.subplot(1, 3, 2)
plt.title("Min-max scaling")
plt.imshow(new_edge_image, cmap='gray')
plt.axis('off')

# Image 3
plt.subplot(1, 3, 3)
plt.title("Hardthreshold")
plt.imshow(new_edge_image_threshold, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()









