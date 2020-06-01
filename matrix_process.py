import cv2
import numpy as np
from scipy import sparse

# Read the image
image = cv2.imread('MicrosoftTeams-image (2).png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set threshold level
threshold_level = 100

# Find coordinates of all pixels below threshold
coords = np.column_stack(np.where(gray < threshold_level))
x = np.array([coords[:, 1]])
y = np.array([coords[:, 0]])

div = 100000
# Tomorrow, try to change len(x) to the pixel values. which is 273x283 something like this
for i in range(len(x)):
    # pair = [x[i], y[i]]
    # pair = np.array([pair])
    original = np.array(y[i] - x[i])
    tran = original.T
    dist = np.array([tran * original]) / div

# Make distance matrix
print("distance matrix:\n", dist)

# Gaussian Kernel & Diagonal Degree
for k in range(len(dist)):
    diag_deg = []
    Gaussian = np.exp(((-dist[k]) ** 2) / 6)
    Gaussian = np.array([Gaussian])
    diag_deg.append(np.sum(k!=0))
    print("Gausian:\n", Gaussian)
    print("Diagonal Degree:\n", diag_deg)

# Algorithm num 5
'''
a1 =
'''

'''
# Make Similarity matrix and return Gaussian kernel (apply previous things to this one) then observe is Memory Error solved.
def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

# There is memory problem if I choose len as 29801(# of dist matrix) probably need dim_reduction
print("Gausian Kernel:\n", gkern(276, 1)*273)
'''

cv2.imshow('image', image)
cv2.waitKey()