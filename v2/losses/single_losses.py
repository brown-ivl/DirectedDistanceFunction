import torch
import torch.nn as nn

class SingleDepthBCELoss(nn.Module):
    Thresh = 0.7  # PARAM
    Lambda = 0.9 # PARAM
    def __init__(self, Thresh=0.7):
        super().__init__()
        self.BCEMaskLoss = nn.BCELoss(size_average=True, reduce=True)
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        intersect, depths = target
        pred_int, pred_depth = output

        L2Loss = self.L2(depths[0], pred_depth[0]) # Single intersection only
        # BCELoss = nn.BCELoss(intersect[0], pred_int[0])
        # Loss = self.Lambda * L2Loss + (1.0 - self.Lambda) * BCELoss

        Loss = L2Loss

        return Loss

    def L2(labels, predictions):
        '''
        L2 loss
        '''
        # print("L2 Loss")
        # print(labels[-10:])
        # print(predictions[-10:])
        return torch.mean(torch.square(labels - predictions))
