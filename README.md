# DermAI

Deep convolutional network to predict 23 classes of skin diseases found in http://www.dermnet.com/dermatology-pictures-skin-disease-pictures from raw images. We make use of a pretrained ResNet152 since many visual semantics like edges or object shapes should be transferrable from Imagenet. We append 2 fully connected layers to fine-tune for our use case.

This project is an entry to API World Hackathon 2017.

## Installation

Run the `setup.sh` script to download raw images and ResNet152 embeddings. The script will also download a trained model (this is a 2-fully-connected network stacked on top of ResNet152). Unzipping the files should produce the following folders: `train/`, `test/`, `train_emb/`, `test_emb/`. 

To install all the libraries needed, run `pip install -r requirements.txt`.
