import torch
import torch.nn as nn
import math

SINGLE_MASK_THRESH = 0.7
SINGLE_L2_LAMBDA = 0.5

class SingleDepthBCELoss(nn.Module):
    Thresh = SINGLE_MASK_THRESH
    Lambda = SINGLE_L2_LAMBDA
    def __init__(self, Thresh=SINGLE_MASK_THRESH, Lambda=SINGLE_L2_LAMBDA):
        super().__init__()
        self.MaskLoss = nn.BCELoss(reduction='mean')
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh
        self.Lambda = Lambda

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        assert isinstance(output, list) # For custom collate
        B = len(target) # Number of batches with custom collate
        Loss = 0
        for b in range(B):
            # Single batch version
            GTMask, GTDepth = target[b]
            PredMaskConf, PredDepth = output[b]

            PredMaskConfSig = self.Sigmoid(PredMaskConf)
            PredMaskMaxConfVal = PredMaskConfSig
            ValidRaysIdx = PredMaskMaxConfVal > self.Thresh  # Use predicted mask
            # ValidRaysIdx = GTMask.to(torch.bool)  # Use ground truth mask

            MaskLoss = self.MaskLoss(PredMaskMaxConfVal.to(torch.float), GTMask.to(torch.float))
            L2Loss = self.L2(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])
            Loss += self.Lambda * L2Loss + MaskLoss
        Loss /= B

        return Loss

    def L2(self, labels, predictions):
        # print(labels.size())
        # print(predictions.size())
        Loss = torch.mean(torch.square(labels - predictions))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)

        # print(torch.min(labels), torch.max(labels))
        # print(torch.min(predictions), torch.max(predictions))
        return Loss
