from flask import Flask

import sys
import torch
from model import FineTuneNet
from utils.embeddings import ResNet152Embedder

from config import MODEL_FILE

embedder = ResNet152Embedder()
embedder.eval()

checkpoint = torch.load(MODEL_FILE, map_location=lambda storage, location: storage)
model = FineTuneNet()
model.load_state_dict(checkpoint['state_dict'])
model.eval()


app = Flask(__name__)
from app import views
