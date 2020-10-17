import torch


def smooth_l1_pytorch(predict, target):
    assert predict.shape == target.shape

    args1 = ((predict - target).absolute() < 1).nonzero()
    args2 = ((predict - target).absolute() >= 1).nonzero()

    losses = torch.zeros(predict.shape)
    args1 = (args1[:, 0], args1[:, 1])
    args2 = (args2[:, 0], args2[:, 1])

    losses[args1] = .5 * (predict[args1] - target[args1]).absolute().square()
    losses[args2] = (predict[args2] - target[args2]).absolute() - .5

    return losses


def rpn_reg_loss(predict, target, target_score):
    return target_score * smooth_l1_pytorch(predict, target)


def rpn_cls_loss(predict, target):
    # log loss over two classes (obj or not obj)
    return - target * torch.log(predict) - (1 - target) * torch.log(1 - predict)


if __name__ == '__main__':
    a = torch.Tensor([[.43, .68, .52],
                      [.27, .86, .59]])
    # b = torch.Tensor([[2.4, 5.6, 1.7],
    #                   [8.5, 3.7, 4.7]])
    b = torch.Tensor([[1, 1, 0],
                      [1, 0, 1]])


    print(smooth_l1_pytorch(a, b))
    print(rpn_cls_loss(a, b))
