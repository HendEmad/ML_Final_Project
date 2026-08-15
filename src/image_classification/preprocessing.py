import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

def create_hog_scaler() -> StandardScaler:
    return StandardScaler()

def extract_hog_features(images):
    features = []
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_features = hog(
            gray_img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        features.append(img_features)
    return np.array(features)