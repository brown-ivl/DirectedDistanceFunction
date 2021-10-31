import torch
import torch.nn as nn
import math

SINGLE_MASK_THRESH = 0.7

class SingleDepthBCELoss(nn.Module):
    Thresh = SINGLE_MASK_THRESH  # PARAM
    Lambda = 5.0 # PARAM
    def __init__(self, Thresh=SINGLE_MASK_THRESH):
        super().__init__()
        self.MaskLoss = nn.BCELoss(reduction='mean')
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        GTMask, GTDepth = target
        PredMaskConf, PredDepth = output

        if len(GTMask.size()) < 3:
            GTMask = GTMask.unsqueeze(0)
        if len(GTDepth.size()) < 3:
            GTDepth = GTDepth.unsqueeze(0)
        if len(PredMaskConf.size()) < 3:
            PredMaskConf = PredMaskConf.unsqueeze(0)
        if len(PredDepth.size()) < 3:
            PredDepth = PredDepth.unsqueeze(0)
        B, R, _ = GTDepth.size()

        PredMaskConfSig = self.Sigmoid(PredMaskConf)
        PredMaskMaxConfVal = PredMaskConfSig
        ValidRaysIdx = PredMaskMaxConfVal > self.Thresh # Use predicted mask
        # ValidRaysIdx = GTMask.to(torch.bool)  # Use ground truth mask

        Loss = 0
        for b in range(B):
            MaskLoss = self.MaskLoss(PredMaskMaxConfVal[b].to(torch.float), GTMask[b].to(torch.float))
            L2Loss = self.L2(GTDepth[b][ValidRaysIdx[b]], PredDepth[b][ValidRaysIdx[b]])

            Loss += self.Lambda * L2Loss + MaskLoss
        Loss /= B

        # # Single batch version , comment out the unsqueeze code on top too
        # MaskLoss = self.MaskLoss(PredMaskMaxConfVal.to(torch.float), GTMask.to(torch.float))
        # L2Loss = self.L2(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])
        # Loss = self.Lambda * L2Loss + MaskLoss

        return Loss

    def L2(self, labels, predictions):
        Loss = torch.mean(torch.square(labels - predictions))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)

        # print(torch.min(labels), torch.max(labels))
        # print(torch.min(predictions), torch.max(predictions))
        return Loss
