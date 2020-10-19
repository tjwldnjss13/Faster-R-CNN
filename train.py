import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import model
from model import FasterRCNN
from rpn import anchor_box_generator, anchor_label_generator, anchor_label_generatgor_2dim, \
                anchor_ground_truth_generator, loc_delta_generator
from loss import rpn_reg_loss, rpn_cls_loss
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

    gt = np.array([[120, 70, 570, 280], [220, 270, 580, 450], [30, 440, 570, 700]])
    gt[:, 0] = gt[:, 0] * (in_size[0] / img_h_og)
    gt[:, 1] = gt[:, 1] * (in_size[1] / img_w_og)
    gt[:, 2] = gt[:, 2] * (in_size[0] / img_h_og)
    gt[:, 3] = gt[:, 3] * (in_size[1] / img_w_og)
    gt_numpy = gt
    gt = torch.from_numpy(gt)

    train_data_loader = (img, gt)

    faster_rcnn_model = FasterRCNN(in_size, 3).to(device)

    ratios = [.5, 1, 2]
    scales = [128, 256, 512]
    anchor_boxes = anchor_box_generator(ratios, scales, in_size, 16)
    anchor_labels_ = anchor_label_generator(anchor_boxes, gt_numpy, .7, .3)
    anchor_labels2_ = anchor_label_generatgor_2dim(anchor_labels_)
    anchor_gts_ = anchor_ground_truth_generator(anchor_boxes, gt)
    anchor_gts_ = loc_delta_generator(anchor_gts_, anchor_boxes)

    anchor_labels = torch.Tensor(anchor_labels_).to(device)
    anchor_labels2 = torch.Tensor(anchor_labels2_).to(device)
    anchor_gts = torch.Tensor(anchor_gts_).to(device)

    del anchor_labels_
    del anchor_labels2_
    del anchor_gts_

    for e in range(epoch):
        backbone_feature = faster_rcnn_model.backbone(img)
        reg, cls = faster_rcnn_model.rpn(backbone_feature)
        print(reg.shape, cls.shape)

        n_reg, reg_loss = rpn_reg_loss(reg, anchor_gts, anchor_labels)
        print(n_reg, reg_loss)
        cls_loss = rpn_cls_loss(cls, anchor_labels2, anchor_labels)



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

