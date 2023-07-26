
import os
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def open_image( filename, astype='PIL' ):
    if astype.lower() in ['pil']:
        image = open_image_pil( filename )
    elif astype.lower() in ['cv','cv2']:
        image = open_image_cv( filename )
    else:
        raise ValueError('Unknown type: {}'.format(astype))
    return image

def open_image_pil( filename ):
    image = Image.open(filename)
    return image

def open_image_cv( filename ):
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.heic']:
        image = convert_from_image_to_cv2(Image.open(filename))
    else:
        image = cv2.imread(filename)
    return image