import matplotlib.pyplot as plt
import numpy as np

from sobel_edge_detection_01 import convolve

# Create a sample matrix
#matrix = np.random.rand(10, 10)
matrix =[[0.25, 0.5, 0.75],
         [0.0,  0.5, 1.0],
         [0.25, 0.5, 0.75]]

matrix =[[0.25, 0.0, 0.25],
         [0.5,  0.5, 0.5],
         [0.75, 1.0, 0.75]]

matrix =[[0, 0.5, 0],
         [0.0,  0.5, 1.0],
         [0.25, 0.5, 0.75]]
# Display matrix as an image
plt.imshow(matrix, cmap='gray') # Use cmap='viridis' or 'plasma' for color
#plt.colorbar() # Show intensity scale
plt.show()

arr = [
[6,	9,	8,	4,	4,	4,	3,	5,	2],
[7,	8,	8,	3,	4,	3,	2,	1,	1],
[6,	8,	8,	0,	0,	0,	1,	1,	0],
[7,	8,	6,	0,	0,	0,	1,	1,	0],
[7,	7,	6,	5,	5,	3,	0,	2,	2],
[7,	7,	5,	4,	4,	4,	5,	5,	3],
[6,	5,	4,	3,	4,	3,	5,	5,	3],
[5,	5,	5,	5,	4,	3,	5,	5,	3],
[4,	4,	6,	7,	5,	6,	6,	7,	6]]
k =[[1, 0, -1],
         [1, 0, -2],
         [1, 0, -1]]

from scipy.ndimage import convolve1d, convolve

print(convolve(arr, k))

