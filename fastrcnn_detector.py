import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class FastRCNNDetector(nn.Module):
    def __init__(self, n_rois, n_classes, idx_valid=None):
        super(FastRCNNDetector, self).__init__()
        self.n_rois = n_rois
        self.n_classes = n_classes
        self.idx_valid = idx_valid
        self.roi_pooling = roi_pooling
        self.fc1 = nn.Linear(512 * self.n_rois * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc_cls = nn.Linear(4096, self.n_rois * self.n_classes)
        self.fc_reg = nn.Linear(4096, self.n_rois * self.n_classes * 4)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        cls = self.fc_cls(x)
        reg = self.fc_reg(x)

        cls = cls.view(cls.size(0), self.n_rois, self.n_classes)
        reg = reg.view(reg.size(0), self.n_rois * self.n_classes)

        cls = self.softmax(cls)

        return cls, reg


def roi_pooling(feature, roi_bbox, out_size):
    in_y, in_x = roi_bbox[:, 0], roi_bbox[:, 1]
    in_h, in_w = roi_bbox[:, 2] - in_y, roi_bbox[:, 3] - in_x
    in_y /= 16
    in_x /= 16
    in_h /= 16
    in_w /= 16
    out_h, out_w = out_size
    pool_h, pool_w = in_h // out_h, in_w // out_w

    in_y, in_x = in_y.int(), in_x.int()
    pool_h, pool_w = pool_h.int(), pool_w.int()

    # Brute force
    roi_feature = torch.zeros((feature.shape[0], feature.shape[1], roi_bbox.shape[0], out_h, out_w)).to(device)
    for m in range(feature.shape[0]):
        for a in range(feature.shape[1]):
            for r in range(roi_bbox.shape[0]):
                for h in range(out_h):
                    for w in range(out_w):
                        roi_feature[m, a, r, h, w] = feature[m, a, in_y[r] + pool_h[r] * h:in_y[r] + pool_h[r] * (h + 1), in_x[r] + pool_w[r] * w:in_x[r] + pool_w[r] * (w + 1)].max(dim=2)

    return roi_feature




