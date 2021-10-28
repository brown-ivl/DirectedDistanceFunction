import torch
import torch.nn as nn
import math

SINGLE_MASK_THRESH = 0.7

class SingleDepthBCELoss(nn.Module):
    Thresh = SINGLE_MASK_THRESH  # PARAM
    Lambda = 10.0 # PARAM
    def __init__(self, Thresh=0.7):
        super().__init__()
        self.MaskLoss = nn.BCELoss(reduction='mean')
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        GTMask, GTDepth = target
        PredMaskConf, PredDepth = output
        PredMaskConfSig = self.Sigmoid(PredMaskConf)

        # print('GTMask size:', GTMask.size())
        # print('GTDepth size:', GTDepth.size())
        # print('PredMaskLikelihood size', PredMaskConf.size())
        # print('PredDepth size', PredDepth.size())
        # print('PredMaskLikelihood', PredMaskConf)
        # print('PredDepth', PredDepth)

        # print(PredMaskConf)
        # print(GTMask)
        # print(PredDepth)
        # print(GTDepth)

        # PredMaskMaxIdx = torch.argmax(PredMaskConfSig, dim=1).to(PredMaskConf.dtype).unsqueeze(1).requires_grad_(True)
        # PredMaskMaxConfVal = torch.gather(PredMaskConfSig, dim=1, index=PredMaskMaxIdx.to(torch.long).view(-1, 1))
        PredMaskMaxConfVal = PredMaskConfSig
        ValidRaysIdx = PredMaskMaxConfVal > self.Thresh # Use predicted mask
        # ValidRaysIdx = GTMask.to(torch.bool)  # Use ground truth mask

        # print('\nGTMask:', GTMask.item())
        # print('PredMaskMaxIdx', PredMaskMaxIdx.item())
        # MaskLoss = self.MaskLoss(PredMaskMaxIdx.to(torch.float), GTMask.to(torch.float))
        MaskLoss = self.MaskLoss(PredMaskMaxConfVal.to(torch.float), GTMask.to(torch.float))
        # MaskLoss = self.MaskLoss(torch.tensor(0.001), torch.tensor(1.))
        # print('MaskLoss', MaskLoss.item())
        L2Loss = self.L2(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])
        # print(MaskLoss)
        # print(L2Loss)

        Loss = self.Lambda * L2Loss + MaskLoss

        return Loss

    def L2(self, labels, predictions):
        Loss = torch.mean(torch.square(labels - predictions))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)
        return Loss
