import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import model
from model import FasterRCNN
from rpn import anchor_box_generator, anchor_target_generator
from torchsummary import summary

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    learning_rate = .001
    batch_size = 64
    epoch = 10

    # Train sample image
    import numpy as np
    import cv2 as cv

    in_size = (600, 1000)

    img_pth = 'samples/dogs.jpg'
    img = cv.imread(img_pth)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_h_og, img_w_og = img.shape[0], img.shape[1]
    img = cv.resize(img, (in_size[1], in_size[0]), interpolation=cv.INTER_CUBIC)
    img_numpy = np.asarray(img)
    img = torch.FloatTensor(img_numpy)
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)

    bbox = np.array([[120, 70, 570, 280], [220, 270, 580, 450], [30, 440, 570, 700]])
    bbox[:, 0] = bbox[:, 0] * (in_size[0] / img_h_og)
    bbox[:, 1] = bbox[:, 1] * (in_size[1] / img_w_og)
    bbox[:, 2] = bbox[:, 2] * (in_size[0] / img_h_og)
    bbox[:, 3] = bbox[:, 3] * (in_size[1] / img_w_og)
    bbox_numpy = bbox
    bbox = torch.from_numpy(bbox)

    train_data_loader = (img, bbox)

    faster_rcnn_model = FasterRCNN(in_size, 3).to(device)

    ratios = [.5, 1, 2]
    scales = [128, 256, 512]
    anchor_boxes = anchor_box_generator(ratios, scales, in_size, 16)
    anchor_labels = anchor_target_generator(anchor_boxes, bbox_numpy, .7, .3)



    for e in range(epoch):
        backbone_feature = faster_rcnn_model.backbone(img)
        reg, cls = faster_rcnn_model.rpn(backbone_feature)


        break

        # for i, data in enumerate(train_data_loader):
        # backbone_feature = faster_rcnn_model.backbone(image)
        # reg, cls = faster_rcnn_model.rpn(backbone_feature)

    # # Visualize
    # import copy
    # import matplotlib.pyplot as plt

    # img_copy = copy.deepcopy(img_numpy)

    # pos_args_label = np.where(anchor_labels == 1)
    # neg_args_label = np.where(anchor_labels == -1)
    # pos_boxes = anchor_boxes[pos_args_label]
    # neg_boxes = anchor_boxes[neg_args_label]

    # print(pos_boxes.shape, neg_boxes.shape)

    # pos_box = pos_boxes[-1]
    # y1, x1, y2, x2 = int(pos_box[0]), int(pos_box[1]), int(pos_box[2]), int(pos_box[3])
    # cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # y1, x1, y2, x2 = int(bbox_numpy[2][0]), int(bbox_numpy[2][1]), int(bbox_numpy[2][2]), int(bbox_numpy[2][3])
    # cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # neg_box = neg_boxes[-1]
    # y1, x1, y2, x2 = int(neg_box[0]), int(neg_box[1]), int(neg_box[2]), int(neg_box[3])
    # cv.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # plt.figure(figsize=(5, 5))
    # plt.imshow(img_copy)
    # plt.show()

    print('Done')

