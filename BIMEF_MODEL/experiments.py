import cv2
from BIMEF import BIMEF
import os
import time

def enhance_img(filename):
    bgr_img = cv2.imread(filename, 1)
    if bgr_img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {filename}")

    # Convert BGR to RGB
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])

    start = time.time()
    enhanced_rgb_img = BIMEF(rgb_img)
    end = time.time()
    print('BIMEF run time:', round(end - start, 3), 'seconds')

    # Convert back RGB to BGR
    r, g, b = cv2.split(enhanced_rgb_img)
    enhanced_bgr_img = cv2.merge([b, g, r])
    return enhanced_bgr_img

if __name__ == '__main__':
    input_file = 'test2.png'
    output_file = 'test2_BIMEF.png'

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    enhanced_img = enhance_img(input_file)
    cv2.imwrite(output_file, enhanced_img)
    print(f"Enhanced image saved as: {output_file}")
