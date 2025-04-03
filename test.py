import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
train_df = pd.read_csv("sign_mnist/sign_mnist_train.csv")

# Extract the first image and reshape it
first_image = train_df.iloc[1, 1:].values.reshape(28, 28).astype(np.uint8)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints
keypoints, descriptors = sift.detectAndCompute(first_image, None)

# Convert keypoints to a list of tuples (x, y, size, angle)
keypoint_list = [(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in keypoints]

for i in range(len(keypoint_list)):
    print(f"{i+1}: {descriptors[i]}")

