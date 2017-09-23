"""Given how PyTorch works, we need to be able to serve 
images in batches. Also include general data utilities."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
import random
from glob import glob

from PIL import Image
import torchvision.transforms as transforms

# use ImageNet preprocessing (since ResNet is trained on it)
preprocessing = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])


class DataLoader(object):
    """Load (image, class) pairs into PyTorch Tensors as batches.

    @param folder: which folder to read from.
    @param batch_size: number of images to process at once.
    """
    def __init__(self, folder, batch_size=1, embedding_size=4096):
        files = glob(os.path.join(folder, '*.jpg'))
        random.shuffle(files)
        self.generator = iter(files)
        self.files = files
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.size = len(files)
        
    def load(self):
        data = torch.Tensor(self.batch_size, self.embedding_size)
        for i in range(batch_size):
            try:
                path = next(self.generator)
                im = Image.open(path)
                im = im.convert('RGB')
                im = preprocessing(im).unsqueeze(0)
                data[i] = im
            except StopIteration:
                data = data[:i]
        return data

    def reset(self):
        self.generator = iter(self.files)


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

        train_folder = os.path.join(train_folder, d)
        test_folder = os.path.join(test_folder, d)

        for i, image in enumerate(train_images):
            shutil.copy(os.path.join(class_folder, image), 
                        os.path.join(train_folder, image))
            print('Copied [{}/{}] images for training.'.format(i + 1, n_train_images))
        
        for i, image in enumerate(test_images):
            shutil.copy(os.path.join(class_folder, image), 
                        os.path.join(test_folder, image))
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
