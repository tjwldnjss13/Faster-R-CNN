import numpy as np
import torch
import torch.nn as nn


class RPN(nn.Module):
    def __init__(self, in_dim, out_dim, in_size, n_anchor):
        super(RPN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_size = in_size
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv.weight.data.normal_(0, .01)
        self.conv.bias.data.zero_()
        # self.reg_layer = nn.Conv2d(out_dim, n_anchor * 4, 1, 1, 0)
        self.reg_layer = nn.Linear(out_dim * self.in_size[0] * in_size[1], n_anchor * 4)
        self.reg_layer.weight.data.normal_(0, .01)
        self.reg_layer.bias.data.zero_()
        # self.cls_layer = nn.Conv2d(out_dim, n_anchor * 2, 1, 1, 0)
        self.cls_layer = nn.Linear(out_dim * self.in_size[0] * self.in_size[1], n_anchor * 2)
        self.cls_layer.weight.data.normal_(0, .01)
        self.cls_layer.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)

        x = x.view(x.size(0), -1)
        reg = self.reg_layer(x)
        cls = self.cls_layer(x)

        # reg = reg.permute(0, 2, 3, 1).contiguous().view(reg.size(0), -1, 4)
        # cls = cls.permute(0, 2, 3, 1).contiguous().view(cls.size(0), -1, 2)

        return reg, cls


def anchor_generator(feature_size, anchor_stride):
    feat_h, feat_w = feature_size[0], feature_size[1]
    ctr_x = np.arange(anchor_stride, (feat_w + 1) * anchor_stride, anchor_stride)
    ctr_y = np.arange(anchor_stride, (feat_h + 1) * anchor_stride, anchor_stride)

    anchors = np.zeros((len(ctr_x) * len(ctr_y), 2))

    c_i = 0
    for x in ctr_x:
        for y in ctr_y:
            anchors[c_i, 1] = x - feat_w // 2
            anchors[c_i, 0] = y - feat_h // 2
            c_i += 1

    return anchors


def anchor_box_generator(ratios, scales, input_size, anchor_stride):
    # 50 = 800 // 16

    # ratios = [.5, 1, 2]
    # scales = [128, 256, 512]

    in_h, in_w = input_size[0], input_size[1]

    feat_h, feat_w = in_h // anchor_stride, in_w // anchor_stride
    anchors = anchor_generator((feat_h, feat_w), anchor_stride)
    anchor_boxes = np.zeros((len(anchors) * len(ratios) * len(scales), 4))

    anc_i = 0
    for anc in anchors:
        anc_y, anc_x = anc
        for r in ratios:
            for s in scales:
                # h = anchor_stride * s * np.sqrt(r)
                # w = anchor_stride * s * np.sqrt(1. / r)
                if r < 1:
                    h, w = s, s * (1. / r)
                elif r > 1:
                    h, w = s * r, s
                else:
                    h, w = s, s

                anchor_boxes[anc_i, 0] = anc_y - .5 * h
                anchor_boxes[anc_i, 1] = anc_x - .5 * w
                anchor_boxes[anc_i, 2] = anc_y + .5 * h
                anchor_boxes[anc_i, 3] = anc_x + .5 * w

                anc_i += 1

    idx_valid = np.where((anchor_boxes[:, 0] >= 0) &
                         (anchor_boxes[:, 1] >= 0) &
                         (anchor_boxes[:, 2] <= in_h) &
                         (anchor_boxes[:, 3] <= in_w))[0]
    anchor_boxes = anchor_boxes[idx_valid]

    return anchor_boxes


if __name__ == '__main__':
    import cv2 as cv
    import matplotlib.pyplot as plt
    import copy

    ratios = [.5, 1, 2]
    scales = [128, 256, 512]
    anchor_boxes = anchor_box_generator(ratios, scales, (600, 1000), 16)
    print(anchor_boxes.shape)

    # img_pth = 'samples/dogs.jpg'
    # img = cv.imread(img_pth)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # img = cv.resize(img, (1000, 600))
    # bbox = np.array([[120, 70, 570, 280], [220, 270, 580, 450], [30, 440, 570, 700]])
    #
    # img_copy = copy.deepcopy(img)
    #
    # for i, box in enumerate(anchor_boxes):
    #     y1, x1, y2, x2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    #     cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
    #
    # plt.figure(figsize=(15, 9))
    # plt.imshow(img_copy)
    # plt.show()

    # rpn = RPN(512, 256, (60, 40), 9).cuda()
    # from torchsummary import summary
    # summary(rpn, (512, 60, 40))