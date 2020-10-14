import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from vgg import VGG


if __name__ == '__main__':
    learning_rate = .001
    batch_size = 64
    epoch = 10


