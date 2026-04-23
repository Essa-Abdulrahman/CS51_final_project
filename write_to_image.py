import numpy as np
import cv2

# matrix
mat_red = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
], dtype=np.float32)

mat_blue = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
    ], dtype=np.float32)
'''
mat_red = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
           ], dtype=np.float32)
mat_blue = np.array([
            [0, 1, 0],
            [1, 4, 1],
            [0, 1, 0]
            ], dtype=np.float32)
'''
# Normalize to 0–255
red = (mat_red + 1) / 5 * 255
blue = (mat_blue + 1) / 5 * 255

red = red.astype(np.uint8)
blue = blue.astype(np.uint8)

img = np.zeros((3, 3, 3), dtype=np.uint8)

img[:, :, 0] = blue   # B channel
img[:, :, 1] = 0      # G channel (unused)
img[:, :, 2] = red    # R channel

img = img.astype(np.uint8)

# Save PNG
cv2.imwrite("sobel_mat.png", img)