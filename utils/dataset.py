import os
import glob
import numpy as np
from skimage import io
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.images, self.labels = self.load_input(data_dir)

    def __getitem__(self, idx):
        image = io.imread(self.images[idx])

        if 0 == len(self.labels):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.labels[idx])

        label = np.zeros(label_3.shape[0:2])
        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    @staticmethod
    def load_input(data_dir):
        images = glob.glob(os.path.join(data_dir, 'images', '*.jpg'))
        labels = [os.path.join(data_dir, 'labels', os.path.basename(f).replace('.jpg', '.png')) for f in images]

        return images, labels
