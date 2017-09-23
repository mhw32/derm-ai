"""helper.py: Utility functions to crop images
and prepare for input into the net.
"""

import base64
import numpy as np

import cv2
from io import StringIO


def read_base64_image(base64_str):
    """Converts base64 string to numpy array.
    
    @param base64_str: base64 encoded string
    @return: numpy array RGB (H x W x C)
    """
    nparr = np.fromstring(base64.b64decode(base64_str),  # base64_str.decode('base64'),
                          np.uint8)
    bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image
