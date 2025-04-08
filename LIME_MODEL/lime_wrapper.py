# LIME_MODEL/lime_wrapper.py

import cv2
from exposure_enhancement import enhance_image_exposure

def enhance_lime_image(image_bgr,
                       gamma=0.6, lambda_=0.15, use_lime=True,
                       sigma=3, bc=1.0, bs=1.0, be=1.0, eps=1e-3):
    """
    Applies LIME enhancement to a single BGR image (NumPy array).
    Returns the enhanced BGR image.
    """
    enhanced_image = enhance_image_exposure(
        image_bgr,
        gamma,
        lambda_,
        not use_lime,  # LIME=True means DUAL=False
        sigma=sigma,
        bc=bc,
        bs=bs,
        be=be,
        eps=eps
    )
    return enhanced_image
