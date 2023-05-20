import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def focal_loss(output, target, gamma=2, alpha=0.5):
    """
    Focal Loss
    :param output: model output
    :param target: ground truth
    :param gamma: gamma
    :param alpha: alpha
    :return: loss
    """
    criterion = nn.CrossEntropyLoss()
    logpt = -criterion(output, target)
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * logpt
    return loss


# label smoothing
def label_smoothing(output, target, alpha=0.1):
    """
    Label Smoothing
    :param output: model output
    :param target: ground truth
    :param alpha: alpha
    :return: loss
    """
    n_class = output.size(1)
    one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
    one_hot = one_hot * (1 - alpha) + (1 - one_hot) * alpha / (n_class - 1)
    log_prb = F.log_softmax(output, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1).mean()
    return loss


def cross_entropy(output, target):
    """
    Cross Entropy
    :param output: model output
    :param target: ground truth
    :return: loss
    """
    return F.cross_entropy(output, target)


def mse_loss(output, target):
    """
    MSE Loss
    :param output: model output
    :param target: ground truth
    :return: loss
    """
    return F.mse_loss(output, target)


def bce_loss(output, target):
    """
    BCE Loss
    :param output: model output
    :param target: ground truth
    :return: loss
    """
    return F.binary_cross_entropy(output, target)


def bce_with_logits_loss(output, target):
    """
    BCE with logits loss
    :param output: model output
    :param target: ground truth
    :return: loss
    """
    return F.binary_cross_entropy_with_logits(output, target)


def l1_loss(output, target):
    """
    L1 Loss
    :param output: model output
    :param target: ground truth
    :return: loss
    """
    return F.l1_loss(output, target)


def l2_loss(output, target):
    """
    L2 Loss
    :param output: model output
    :param target: ground truth
    :return: loss
    """
    return F.mse_loss(output, target)


def smooth_l1_loss(output, target):
    """
    Smooth L1 Loss
    :param output: model output
    :param target: ground truth
    :return: loss
    """
    return F.smooth_l1_loss(output, target)
