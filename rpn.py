import torch
import torch.nn as nn


class RPN(nn.Module):
    def __init__(self, in_dim, out_dim, n_anchor):
        super(RPN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1).cuda()
        self.conv.weight.data.normal_(0, .01)
        self.conv.bias.data.zero_()
        self.reg_layer = nn.Conv2d(out_dim, n_anchor * 4, 1, 1, 0).cuda()
        self.reg_layer.weight.data.normal_(0, .01)
        self.reg_layer.bias.data.zero_()
        self.cls_layer = nn.Conv2d(out_dim, n_anchor * 2, 1, 1 ,0).cuda()
        self.cls_layer.weight.data.normal_(0, .01)
        self.cls_layer.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        reg = self.reg_layer(x)
        cls = self.cls_layer(x)

        reg = reg.permute(0, 2, 3, 1).contiguous().view(reg.size(0), -1, 4)
        cls = cls.permute(0, 2, 3, 1).contiguous().view(cls.size(0), -1, 2)

        return [reg, cls]
