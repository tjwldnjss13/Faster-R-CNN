import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from model import FasterRCNN


if __name__ == '__main__':
    learning_rate = .001
    batch_size = 64
    epoch = 10

    train_data_loader = 0

    faster_rcnn_model = FasterRCNN((600, 1000), 3)

    for e in range(epoch):
        for i, (image, label) in enumerate(train_data_loader):
            backbone_feature = faster_rcnn_model.backbone(image)
            reg, cls = faster_rcnn_model.rpn(backbone_feature)




