from __future__ import print_function, division
import math
import torch
import random

import numpy as np
from torchvision import transforms


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def loss_fusion(output, labels_v):
    bce_loss = torch.nn.BCELoss(size_average=True)
    losses = [bce_loss(d, labels_v) for d in output]
    total_loss = sum(losses)

    # print(
    #     "l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % tuple(loss.data.item() for loss in losses))

    return losses[0], total_loss


def get_transforms(train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomCrop((288, 288)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform
