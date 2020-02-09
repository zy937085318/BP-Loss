import torch
from torch import nn
from .CycleNet import CycleNet


class BPLoss(nn.Module):
    def __init__(self, dic=None, weights=None):
        super(BPLoss, self).__init__()
        self.net = CycleNet()
        if dic is not None:
            self.net.load_state_dict(torch.load(dic))
        self.loss = torch.nn.L1Loss()
        self.weights = weights
        self.bce = torch.nn.BCELoss()

    def forward(self, preds, labels):
        preds = torch.sigmoid(preds)
        pred_f = self.net(preds)
        label_f = self.net(labels)
        n_layer = len(pred_f)
        if self.weights is None:
            self.weights = [1] * n_layer
        loss = []
        for pred, label, weight in zip(pred_f, label_f, self.weights):
            loss.append(weight * self.loss(pred, label))
        bceloss = self.bce(preds, labels)
        return [sum(loss[0:n_layer//2]), sum(loss[n_layer//2:]), bceloss]
