"""Given pre-trained ResNet152, pre-compute embeddings for 
all images once so that we do not need to do this every 
epoch during training"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

from .data import clone_directory_structure
from .data import preprocessing


class ResNet152Embedder(nn.Module):
    """Grabs the average pool embedding layer from ResNet152, 
    which seems to do very nicely in organizing images.
    """
    def __init__(self):
        super(ResNet152Embedder, self).__init__()
        resnet152 = models.resnet152(pretrained=True)
        self.embedder = nn.Sequential(*(list(resnet152.children())[:-1]))

    def forward(self, x):
        return self.embedder(x)


if __name__ == '__main__':
    import os
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('img_folder', type=str, help='path to where images are stored.')
    parser.add_argument('out_folder', type=str, help='path to save embeddings to.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    embedder = ResNet152Embedder()
    embedder.eval()

    if args.cuda:
        embedder.cuda()

    for param in embedder.parameters():
        param.requires_grad = False

    image_paths = glob(os.path.join(args.img_folder, '*', '*.jpg'))
    n_images = len(image_paths)

    # We are going to loop through these in batches and run them 
    # through ResNet152.
    image_raw_batch, image_raw_batches = [], []

    # First, we load each image into 32 batch groups.
    for i in range(n_images):
        print('Loading image [{}/{}]'.format(i + 1, n_images))

        image_torch = Image.open(image_paths[i])
        image_torch = image_torch.convert('RGB')
        image_torch = preprocessing(image_torch).unsqueeze(0)
        image_raw_batch.append(image_torch)

        if (i + 1) % args.batch_size == 0:
            image_raw_batch = torch.cat(image_raw_batch, dim=0)
            image_raw_batches.append(image_raw_batch)
            image_raw_batch = []

    if len(image_raw_batch) > 0:
        image_raw_batch = torch.cat(image_raw_batch, dim=0)
        image_raw_batches.append(image_raw_batch)

    # we then shove each batch through ResNet
    n_batches = len(image_raw_batches)
    image_emb_batches = []

    for i in range(n_batches):
        print('Getting embeddings [batch {}/{}]'.format(i + 1, n_batches))
        image_inputs = image_raw_batches[i]
        image_inputs = Variable(image_inputs, volatile=True)

        if args.cuda:
            image_inputs = image_inputs.cuda()

        image_emb = embedder(image_inputs)
        image_emb_batches.append(image_emb)

    image_embs = torch.cat(image_emb_batches, dim=0)
    image_embs = image_embs.cpu().data.numpy()

    # Finally, we save the resulting embedding as a NumPy object.
    clone_directory_structure(args.img_folder, args.out_folder)

    for i in range(n_images):
        print('Saving numpy object [{}/{}]'.format(i + 1, n_images))
        image_name = os.path.splitext(image_paths[i])[0]
        image_path = image_name + '.npy'
        image_path = image_path.replace(args.img_folder, args.out_folder)
        np.save(image_path, image_embs[i])

