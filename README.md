# DermAI

Deep convolutional network to predict 23 classes of skin diseases found in http://www.dermnet.com/dermatology-pictures-skin-disease-pictures from raw images. We make use of a pretrained ResNet152 since many visual semantics like edges or object shapes should be transferrable from Imagenet. We append 2 fully connected layers to fine-tune for our use case.

This project is an entry to API World Hackathon 2017.

## Installation

Run the `setup.sh` script to download training and testing images and training and testing ResNet152 embeddings. The script will also download a trained finetuned model (this is a 2-fully-connected network stacked on top of ResNet152).
