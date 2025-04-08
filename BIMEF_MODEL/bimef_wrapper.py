# bimef_wrapper.py

import cv2
import numpy as np
from PIL import Image
from BIMEF import BIMEF
import time

def enhance_bimef_image(pil_image: Image.Image) -> Image.Image:
    """
    Enhances a low-light image using the BIMEF algorithm.

    Args:
        pil_image (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Enhanced image.
    """
    # Convert PIL RGB to OpenCV RGB
    rgb_image = np.array(pil_image)

    # Run BIMEF
    start = time.time()
    enhanced_rgb = BIMEF(rgb_image)
    end = time.time()
    print('BIMEF run time:', round(end - start, 3), 'seconds')

    # Convert NumPy RGB to PIL image
    enhanced_pil = Image.fromarray(np.uint8(enhanced_rgb))

    return enhanced_pil
