from app import app
from flask import request, Response

import json
from .helper import read_base64_image
from .helper import gen_prediction
from .helper import gen_probabilities

@app.route('/')
@app.route('/index')
def index():
    return "DermAI Inference Server"


@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.data)
    base64_str = data.get('image')
    rgb_image = read_base64_image(base64_str)

    probas = gen_probabilities(rgb_image)
    klass, score = gen_prediction(probas)

    response = {'class': klass, 'score': score}
    response = json.dumps(response)
    return Response(response=response)
