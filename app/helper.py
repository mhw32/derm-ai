"""helper.py: Utility functions to crop images
and prepare for input into the net.
"""

import base64
import numpy as np

import cv2
from io import StringIO

from app import model, embedder
from utils import preprocessing
from utils import CLASS_IX_TO_NAME


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


def gen_probabilities(image):
    """Call PyTorch ResNet + Fine-tuned net.
    
    @param image: RGB face image.
    @return probability vector of size 23.
    """

    # first get the image into standard form
    image = Image.from_numpy(image)
    image = image.convert('RGB')
    image = preprocessing(image).unsqueeze(0)
    image = Variable(image, volatile=True)

    # pass image through ResNet to get embedding
    embedding = embedder(image)

    # pass image through model to get prediction
    log_probas = model(image)
    probas = torch.exp(log_probas)  # probabilities

    return probas


def gen_prediction(probas):
    probas = probas.cpu().data.numpy()
    probas = probas.flatten()
    ix = np.argmax(probas)
    
    class_name = CLASS_IX_TO_NAME[ix]
    class_proba = probas[ix]

    return class_name, class_proba

