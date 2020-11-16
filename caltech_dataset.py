from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        # Open file in read only mode and read all lines
        file = open(self.split, "r")
        lines = file.readlines()

        # Filter out the lines which start with 'BACKGROUND_Google' as asked in the homework
        self.elements = [i for i in lines if not i.startswith('BACKGROUND_Google')]

        # Delete BACKGROUND_Google class from dataset labels
        self.classes = sorted(os.listdir(os.path.join(self.root, "")))
        self.classes.remove("BACKGROUND_Google")


    def __getitem__(self, index):
        ''' 
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        img = Image.open(os.path.join(self.root, self.elements[index].rstrip()))

        target = self.classes.index(self.elements[index].rstrip().split('/')[0])

        image, label = img, target # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        # Provides a way to get the length (number of elements) of the dataset
        length =  len(self.elements)
        return length
