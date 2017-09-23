"""Given how PyTorch works, we need to be able to serve 
images in batches. Also include general data utilities."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
import random
import numpy as np
from glob import glob

import torch
from PIL import Image
import torchvision.transforms as transforms

# use ImageNet preprocessing (since ResNet is trained on it)
preprocessing = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])


CLASS_NAME_TO_IX = {
    u'Acne and Rosacea Photos': 2,
    u'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': 3,
    u'Atopic Dermatitis Photos': 11,
    u'Bullous Disease Photos': 17,
    u'Cellulitis Impetigo and other Bacterial Infections': 1,
    u'Eczema Photos': 22,
    u'Exanthems and Drug Eruptions': 8,
    u'Hair Loss Photos Alopecia and other Hair Diseases': 6,
    u'Herpes HPV and other STDs Photos': 12,
    u'Light Diseases and Disorders of Pigmentation': 4,
    u'Lupus and other Connective Tissue diseases': 9,
    u'Melanoma Skin Cancer Nevi and Moles': 16,
    u'Nail Fungus and other Nail Disease': 18,
    u'Poison Ivy Photos and other Contact Dermatitis': 0,
    u'Psoriasis pictures Lichen Planus and related diseases': 5,
    u'Scabies Lyme Disease and other Infestations and Bites': 10,
    u'Seborrheic Keratoses and other Benign Tumors': 13,
    u'Systemic Disease': 19,
    u'Tinea Ringworm Candidiasis and other Fungal Infections': 21,
    u'Urticaria Hives': 7,
    u'Vascular Tumors': 15,
    u'Vasculitis Photos': 14,
    u'Warts Molluscum and other Viral Infections': 20,
}

CLASS_IX_TO_NAME = {v: k for k, v in CLASS_NAME_TO_IX.iteritems()}


class DataLoader(object):
    """Load (image, class) pairs into PyTorch Tensors as batches.

    @param folder: which folder to read from.
    @param batch_size: number of images to process at once.
    """
    def __init__(self, folder, batch_size=1, embedding_size=2048, extension='npy'):
        files = glob(os.path.join(folder, '*', '*.{}'.format(extension)))
        random.shuffle(files)
        self.generator = iter(files)
        self.files = files
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.size = len(files)
        self.batch_idx = -1
        
    def load(self):
        data = torch.Tensor(self.batch_size, self.embedding_size)
        target = torch.LongTensor(self.batch_size)
        out_of_data = False

        for i in range(self.batch_size):
            try:
                path = next(self.generator)
                klass = path.split('/')[-2]  # HACK
                klass = CLASS_NAME_TO_IX[klass]
                embedding = np.load(path)
                embedding = torch.from_numpy(embedding)
                data[i] = embedding
                target[i] = int(klass)
            except StopIteration:
                data = data[:i]
                target = target[:i]
                out_of_data = True
        
        self.batch_idx += 1
        return out_of_data, self.batch_idx, (data, target)

    def reset(self):
        self.generator = iter(self.files)
        self.batch_idx = -1


def train_test_split(img_folder, train_folder, test_folder, split_frac=0.8):
    """Given img_folder that includes a sub-directory for each 
    class, we will clone the structure and put <split_frac> % 
    of the training data into the train_folder and the rest into
    the test_folder.

    @param img_folder: where unsplit data lives.
    @param train_folder: where to save training split data.
    @param test_folder: where to save testing split data.
    @param split_frac: percent of data per class to put into training.
    """
    clone_directory_structure(img_folder, train_folder)
    clone_directory_structure(img_folder, test_folder)

    _, dirs, _ = os.walk(img_folder).next()
    
    for d in dirs:
        class_folder = os.path.join(img_folder, d)
        class_images = os.listdir(class_folder)
        
        n_images = len(class_images)
        n_train_images = int(n_images * split_frac)

        random.shuffle(class_images)
        train_images = class_images[:n_train_images]
        test_images = class_images[n_train_images:]

        _train_folder = os.path.join(train_folder, d)
        _test_folder = os.path.join(test_folder, d)

        for i, image in enumerate(train_images):
            shutil.copy(os.path.join(class_folder, image), 
                        os.path.join(_train_folder, image))
            print('Copied [{}/{}] images for training.'.format(i + 1, n_train_images))
        
        for i, image in enumerate(test_images):
            shutil.copy(os.path.join(class_folder, image), 
                        os.path.join(_test_folder, image))
            print('Copied [{}/{}] images for testing.'.format(i + 1, n_images - n_train_images))


def clone_directory_structure(in_folder, out_folder):
    """Creates a new directory (out_folder) with all the sub-directory
    structure of in_folder but does not copy content.

    @arg in_folder: folder to be copied.
    @arg out_folder: folder to store new folders.
    """
    child_folders = []
    for _, dirs, _ in os.walk(in_folder):
        dirs[:] = [d for d in dirs if not d[0] == '.']
        child_folders += dirs

    for folder in child_folders:
        folder = os.path.join(in_folder, folder)
        new_folder = folder.replace(in_folder, out_folder)
        if not os.path.isdir(new_folder):
            os.mkdir(new_folder)
            print('Created directory: {}.'.format(new_folder))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img_folder', type=str)
    parser.add_argument('train_folder', type=str)
    parser.add_argument('test_folder', type=str)
    parser.add_argument('--split_frac', type=float, default=0.8)
    args = parser.parse_args()

    train_test_split(args.img_folder, args.train_folder, 
                     args.test_folder, split_frac=args.split_frac)

