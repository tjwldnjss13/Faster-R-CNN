import torch
import numpy as np


def crop(in_, out_size):
    in_size = (in_.shape[1], in_.shape[2])
    drop_size = (in_size[0] - out_size[0], in_size[1] - out_size[1])
    drop_size = (drop_size[0] // 2, drop_size[1] // 2)
    out_ = in_[:, drop_size[0]:drop_size[0] + out_size[0], drop_size[1]:drop_size[1] + out_size[1]]

    return out_


def pad_4dim(x, ref, cuda=True):
    zeros = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3])
    if cuda:
        zeros = zeros.cuda()
    while x.shape[2] < ref.shape[2]:
        x = torch.cat([x, zeros], dim=2)
    zeros = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 1)
    if cuda:
        zeros = zeros.cuda()
    while x.shape[3] < ref.shape[3]:
        x = torch.cat([x, zeros], dim=3)

    return x


def pad_3dim(x, ref_size):
    zeros = torch.zeros(x.shape[0], 1, x.shape[2]).cuda()
    while x.shape[1] < ref_size[0]:
        x = torch.cat([x, zeros], dim=1)
    zeros = torch.zeros(x.shape[0], x.shape[1], 1).cuda()
    while x.shape[2] < ref_size[1]:
        x = torch.cat([x, zeros], dim=2)

    return x


def pad_2dim(x, ref_size):
    zeros = torch.zeros(1, x.shape[1], dtype=torch.long).cuda()
    while x.shape[0] < ref_size[0]:
        x = torch.cat([x, zeros], dim=0)
    zeros = torch.zeros(x.shape[0], 1, dtype=torch.long).cuda()
    while x.shape[1] < ref_size[1]:
        x = torch.cat([x, zeros], dim=1)

    return x


def calculate_ious(box1, box2):
    ious = np.zeros((box1.shape[0], box2.shape[0]), dtype=np.float32)

    for i, b1 in enumerate(box1):
        b1 = box1[i]
        for j, b2 in enumerate(box2):
            b2 = box2[j]
            ious[i, j] = calculate_iou(b1, b2)

    return ious


def calculate_iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    y1 = np.maximum(box1[0], box2[0])
    x1 = np.maximum(box1[1], box2[1])
    y2 = np.minimum(box1[2], box2[2])
    x2 = np.minimum(box1[3], box2[3])

    iou = 0
    if y1 < y2 and x1 < x2:
        inter = (y2 - y1) * (x2 - x1)
        union = area1 + area2 - inter
        iou = inter / union

    return iou


def mean_iou_segmentation(output, predict):
    a, b = (output[:, 1, :, :] > 0), (predict > 0)

    a_area = len(a.nonzero())
    b_area = len(b.nonzero())
    union = a_area + b_area
    inter = len((a & b).nonzero())
    iou = inter / (union - inter)

    return iou


def nms(anchor_boxes, ground_truth, score, iou_threshold):
    n_gt = ground_truth.shape[0] if len(ground_truth.shape) > 1 else 1
    anchor_boxes_nms = []

    for i in range(n_gt):
        anchor_boxes_cat = anchor_boxes[i]
        ious_boxes_gts = calculate_ious(anchor_boxes_cat, np.array([ground_truth[i]]))
        argmax_iou_boxes_gt = np.argmax(ious_boxes_gts, axis=0)
        max_iou_box = anchor_boxes_cat[argmax_iou_boxes_gt][0]
        anchor_boxes_nms.append(max_iou_box)
        for j in range(anchor_boxes_cat.shape[0]):
            if j == argmax_iou_boxes_gt:
                continue
            iou_temp = calculate_iou(max_iou_box, anchor_boxes_cat[j])
            if iou_temp >= iou_threshold:
                anchor_boxes_nms.append(anchor_boxes_cat[j])

    return np.array(anchor_boxes_nms)


def time_calculator(sec):
    if sec < 60:
        return 0, 0, sec
    if sec < 3600:
        M = sec // 60
        S = sec % M
        return 0, M, S
    H = sec // 3600
    sec = sec % 3600
    M = sec // 60
    S = sec % 60
    return int(H), int(M), S


if __name__ == '__main__':

    from rpn import anchor_box_generator, anchor_label_generator, anchor_label_generatgor_2dim, \
        anchor_ground_truth_generator, anchor_boxes_generator_categorical, loc_delta_generator
    import cv2 as cv
    import matplotlib.pyplot as plt
    import copy

    img_pth = 'samples/dogs.jpg'
    img = cv.imread(img_pth)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_h_og, img_w_og, _ = img.shape
    img = cv.resize(img, (1000, 600))

    gt = np.array([[120, 70, 570, 280], [220, 270, 580, 450], [30, 440, 570, 700]])
    gt[:, 0] = gt[:, 0] * (600 / img_h_og)
    gt[:, 1] = gt[:, 1] * (1000 / img_w_og)
    gt[:, 2] = gt[:, 2] * (600 / img_h_og)
    gt[:, 3] = gt[:, 3] * (1000 / img_w_og)
    gt_numpy = gt
    gt = torch.from_numpy(gt)

    img_copy = copy.deepcopy(img)

    in_size = (600, 1000)
    ratios = [.5, 1, 2]
    scales = [128, 256, 512]
    anchor_boxes = anchor_box_generator(ratios, scales, in_size, 16)
    idx_valid_ = np.where((anchor_boxes[:, 0] >= 0) &
                          (anchor_boxes[:, 1] >= 0) &
                          (anchor_boxes[:, 2] <= in_size[0]) &
                          (anchor_boxes[:, 3] <= in_size[1]))[0]
    anchor_boxes = anchor_boxes[idx_valid_]
    anchor_labels_ = anchor_label_generator(anchor_boxes, gt_numpy, .7, .3)
    anchor_labels2_ = anchor_label_generatgor_2dim(anchor_labels_)
    anchor_gts_ = anchor_ground_truth_generator(anchor_boxes, gt)
    # anchor_gts_ = loc_delta_generator(anchor_gts_, anchor_boxes)

    anchor_boxes_cat_ = anchor_boxes_generator_categorical(anchor_boxes, gt)


    # Score 부분은 임시로 넣은 거임
    anchor_boxes_nms_ = nms(anchor_boxes_cat_, gt_numpy, anchor_labels_, .5)
    print(anchor_boxes_nms_.shape)

    for i, box in enumerate(anchor_boxes_nms_):
        # y1, x1 = int(box[0] - .5 * box[2]), int(box[1] - .5 * box[3])
        # y2, x2 = int(y1 + box[2]), int(x1 + box[3])
        y1, x1, y2, x2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_copy)
    plt.show()

    # for i, box in enumerate(anchor_boxes):
    #     # y1, x1 = int(box[0] - .5 * box[2]), int(box[1] - .5 * box[3])
    #     # y2, x2 = int(y1 + box[2]), int(x1 + box[3])
    #     y1, x1, y2, x2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    #     cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
    #
    # plt.figure(figsize=(10, 6))
    # plt.imshow(img_copy)
    # plt.show()


