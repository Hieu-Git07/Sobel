import numpy as np
import sympy as sp
from PIL import Image
import matplotlib.pyplot as plt
import math
import time


# Convert image to 2D vector (grayscale)
img = Image.open("parrot(gs).png").convert("L")
img = np.array(img)
print(img)
print(img.shape)

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
        

for i in range(1, img_pad.shape[0]-1):
    for j in range(1, img_pad.shape[1] - 1):
        # Xac xuat = do thay doi / do thay doi max (for soft_edge)
        prob[i-1][j-1] = grad_val[i-1][j-1]/max_val

# Hard Threshold (Goi la Hard Threshold la vi se so sanh voi Threshold va set 0 / 255)
threshold = 0
for i in range(img.shape[0] - 1):
    for j in range(img.shape[1] - 1):
        threshold += grad_val[i - 1][j - 1]
        
threshold /= (img.shape[0] - 1) * (img.shape[1] - 1)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if grad_val[i][j] > threshold:
            new_edge_image_threshold[i][j] = 255
        else:
            new_edge_image_threshold[i][j] = 0


# Soft edge
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        new_edge_image[i][j] = prob[i][j] * 255

# Boosting edge for soft edge
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        new_edge_image[i][j] = min(255, new_edge_image[i][j] * 1.25)

# Plot anh cuc ki gay (co the plot thu threshold de so sanh)

plt.imshow(new_edge_image, cmap = 'gray')
plt.show()










