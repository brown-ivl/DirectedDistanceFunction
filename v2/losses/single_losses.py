import torch
import torch.nn as nn
import math

class SingleDepthBCELoss(nn.Module):
    Thresh = 0.7  # PARAM
    Lambda = 0.5 # PARAM
    def __init__(self, Thresh=0.7):
        super().__init__()
        self.MaskLoss = nn.BCEWithLogitsLoss(reduction='mean')
        # self.MaskLoss = nn.BCELoss(reduction='mean')
        # self.MaskLoss = nn.CrossEntropyLoss()
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        GTMask, GTDepth = target
        PredMaskConf, PredDepth = output

        # print('GTMask size:', GTMask.size())
        # print('GTDepth size:', GTDepth.size())
        # print('PredMaskLikelihood', PredMaskConf)
        # print('PredMaskLikelihood size', PredMaskConf.size())
        # print('PredDepth', PredDepth)
        # print('PredDepth size', PredDepth.size())

        PredMaskMaxIdx = torch.argmax(torch.squeeze(PredMaskConf), dim=1).to(PredMaskConf.dtype).requires_grad_(True)
        # PredMaskMaxConfVal = torch.gather(torch.squeeze(PredMaskConf), dim=1, index=PredMaskMaxIdx.to(torch.long).view(-1, 1))
        # ValidRaysIdx = PredMaskMaxConfVal > self.Thresh # Use predicted mask
        ValidRaysIdx = GTMask.to(torch.bool)  # Use ground truth mask

        MaskLoss = self.MaskLoss(PredMaskMaxIdx.to(torch.float), torch.squeeze(GTMask).to(torch.float))
        L2Loss = self.L2(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])

        Loss = self.Lambda * L2Loss + ((1.0 - self.Lambda) * MaskLoss)

        return Loss

    def L2(self, labels, predictions):
        Loss = torch.mean(torch.square(labels - predictions))
        if math.isnan(Loss) or math.isinf(Loss):
            return 10
        return Loss
