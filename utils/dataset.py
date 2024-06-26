import os
import cv2
import glob
import numpy as np
from PIL import Image
from torch.utils import data
from utils.util import get_transforms
from torchvision.transforms import functional as F


class Dataset(data.Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.transform = get_transforms(is_train)
        self.images, self.labels = self.load_input(data_dir)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        label = Image.open(self.labels[index]).convert('L')

        image, label = self.process(image, label)

        if self.transform:
            image = self.transform(image)
            label = F.resize(label, size=image.shape[-2:])
            label = F.to_tensor(label)

        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.images)

    @staticmethod
    def load_input(data_dir):
        images = glob.glob(os.path.join(data_dir, 'images', '*' + '.jpg'))
        labels = [os.path.join(data_dir, 'labels', os.path.basename(f).replace('.jpg', '.png')) for f in images]

        return images, labels

    @staticmethod
    def process(image, label):
        # Convert PIL Images to numpy arrays
        image_np = np.array(image)
        label_np = np.array(label)

        # Ensure label is 2D
        if len(label_np.shape) == 3:
            label_np = label_np[:, :, 0]

        # Ensure image is 3D
        if len(image_np.shape) == 2:
            image_np = np.expand_dims(image_np, axis=2)

        # Convert back to PIL Images
        image = Image.fromarray(image_np)
        label = Image.fromarray(label_np)

        return image, label
