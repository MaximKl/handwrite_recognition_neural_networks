import cv2
import numpy as np

def preprocess_image(image_path, input_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.convertScaleAbs(image, alpha=3, beta=0)
    image = np.invert(image)
    image = cv2.resize(image, (input_size, input_size))
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)
    image = image / 255.0
    image = np.reshape(image, (1, input_size, input_size, 3))
    return image
