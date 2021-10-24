import torch
import torch.nn as nn

class SingleDepthBCELoss(nn.Module):
    Thresh = 0.7  # PARAM
    Lambda = 0.2 # PARAM
    def __init__(self, Thresh=0.7):
        super().__init__()
        self.BCEMaskLoss = nn.BCELoss(reduction='mean')
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        intersect, depths = target
        pred_int, pred_depth = output

        # if intersect[0] > 0:
        #     print(intersect)
        #     print(pred_int[0])
        #     print(depths)

        # PredIntersect = torch.argmax(pred_int[0]).to(pred_int[0].dtype)
        # PredInt_Val = pred_int[0][int(PredIntersect)]
        # BCELoss = self.BCEMaskLoss(PredIntersect, torch.squeeze(intersect[0]))
        L2Loss = self.L2(depths[0], pred_depth[0]) # Single intersection only

        # Loss = (PredInt_Val > self.Thresh) * self.Lambda * L2Loss + ((1.0 - self.Lambda) * BCELoss)
        Loss = L2Loss

        return Loss

    def L2(self, labels, predictions):
        return torch.mean(torch.square(labels - predictions))
