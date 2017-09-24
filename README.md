# DermAI

Deep convolutional network to predict 23 classes of skin diseases found in http://www.dermnet.com/dermatology-pictures-skin-disease-pictures from raw images. We make use of a pretrained ResNet152 since many visual semantics like edges or object shapes should be transferrable from Imagenet. We append 2 fully connected layers to fine-tune for our use case.

This project is an entry to API World Hackathon 2017.

## Installation

Run the `setup.sh` script to download raw images and ResNet152 embeddings. The script will also download a trained model (this is a 2-fully-connected network stacked on top of ResNet152). Unzipping the files should produce the following folders: `train/`, `test/`, `train_emb/`, `test_emb/`. `trained_models` includes a trained version of the fine-tune net with 0.53 percent accuracy on held-out test data.

To install all the libraries needed, run `pip install -r requirements.txt`.

## Instructions

To start the Flask app, do `python run.py`. There is a single POST route (`/predict`) that takes a JSON from key `image` to a base64 encoded image. It will return a `class` and a `score`. 
