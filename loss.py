import torch

from rpn import loc_delta_generator


def log_loss_pytorch(predict, target):
    return - target * torch.log(predict) - (1 - target) * torch.log(1 - predict)


def smooth_l1_pytorch(predict, target):
    assert predict.shape[1:] == target.shape[1:]

    args1 = ((predict - target).absolute() < 1).nonzero()
    args2 = ((predict - target).absolute() >= 1).nonzero()

    losses = torch.zeros(predict.shape)
    args1 = (args1[:, 0], args1[:, 1], args1[:, 2]) if args1.shape[0] > 0 else None
    args2 = (args2[:, 0], args2[:, 1], args2[:, 2]) if args2.shape[0] > 0 else None

    losses[args1] = .5 * (predict[args1] - target[args1]).absolute().square() if args1 is not None else 0
    losses[args2] = (predict[args2] - target[args2]).absolute() - .5 if args2 is not None else 0

    return losses


def rpn_reg_loss(predict, target, target_score):
    assert predict.shape[1:] == target.shape

    n_reg = target.shape[0]
    target_ = target.unsqueeze(0)

    # losses = torch.zeros(predict.shape)
    train_args, no_train_args = torch.where(target_score != -1)[0], torch.where(target_score == -1)[0]
    losses = smooth_l1_pytorch(predict, target_)
    losses = losses.sum(2)
    losses[:, no_train_args] = 0
    losses *= target_score
    loss = losses.sum() / n_reg

    return n_reg, loss


def rpn_cls_loss(predict, target, score):
    # log loss over two classes (obj or not obj)
    assert predict.shape[1:] == target.shape
    assert target.shape[:1] == score.shape

    train_args = (score != -1).nonzero()
    target_ = target.unsqueeze(0)
    losses = log_loss_pytorch(predict, target_)
    print(losses)

    # return - target * torch.log(predict) - (1 - target) * torch.log(1 - predict)


if __name__ == '__main__':
    a = torch.Tensor([[.43, .68, .52],
                      [.27, .86, .59]])
    # b = torch.Tensor([[2.4, 5.6, 1.7],
    #                   [8.5, 3.7, 4.7]])
    b = torch.Tensor([[1, 1, 0],
                      [1, 0, 1]])
    c = torch.Tensor([0, -1, 1])


    print(smooth_l1_pytorch(a, b))
    print(rpn_cls_loss(a, b))
    rpn_reg_loss(a, b, c)
