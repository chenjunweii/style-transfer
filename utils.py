import cv2
import numpy as np


#mean = [123.68, 116.779, 103.939]
mean = [103.939, 116.779, 123.68]

def preprocess(image, shape = None):
    
    image = image - mean
    
    if shape is not None:
        
        image = cv2.resize(image, tuple(shape[0:2]))
    
    return np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)

def unprocess(image):
    
    return np.swapaxes(np.swapaxes(image, 0, 1), 1, 2) + mean


