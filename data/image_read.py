import cv2
import numpy as np
# 256 192
def read_images(image_list):
    image_data = []
    for path in image_list:
        image = cv2.imread(path,1)
        image = cv2.resize(image, (256, 256))
        image_data.append(image / 255)
    return np.array(image_data).astype(np.float32)