import cv2
import numpy as np
from PIL import Image

def apply_padding(x1, y1, width, height, padding):
    return x1 - padding, y1 - padding, width + (2 * padding), height + (2 * padding)

def convert_to_pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    return pil_img

def convert_to_cv2(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img