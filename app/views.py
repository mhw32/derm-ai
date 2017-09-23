from app import app
from flask import request, Response

import json
from .helper import read_base64_image


@app.route('/')
@app.route('/index')
def index():
    return "DermAI Inference Server"
