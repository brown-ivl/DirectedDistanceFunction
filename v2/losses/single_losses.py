import torch
import torch.nn as nn
import math

SINGLE_MASK_THRESH = 0.7
SINGLE_L2_LAMBDA = 5.0
SINGLE_L1_LAMBDA = 5.0
REG_LAMBDA = 1e-4

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

class ADPredLoss(nn.Module):
    '''
    Computes the prediction loss for the autodecoder
    '''

    def __init__(self, Thresh=SINGLE_MASK_THRESH, Lambda=SINGLE_L2_LAMBDA):
        super().__init__()
        self.MaskLoss = nn.BCELoss(reduction='mean')
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh
        self.Lambda = Lambda

    def forward(self, output, target):
        print("PRED LOSS OUTPUT")
        print(len(output))
        print("PRED LOSS TARGET")
        print(target)
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        Output = output[0]
        assert isinstance(Output, list) # For custom collate
        B = len(target) # Number of batches with custom collate
        Loss = 0
        for b in range(B):
            # Single batch version
            GTMask, GTDepth = target[b]
            PredMaskConf, PredDepth = Output[b]

            PredMaskConfSig = self.Sigmoid(PredMaskConf)
            PredMaskMaxConfVal = PredMaskConfSig
            ValidRaysIdx = PredMaskMaxConfVal > self.Thresh  # Use predicted mask
            # ValidRaysIdx = GTMask.to(torch.bool)  # Use ground truth mask

            MaskLoss = self.MaskLoss(PredMaskMaxConfVal.to(torch.float), GTMask.to(torch.float))
            L1Loss = self.L1(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])
            PredictionLoss = self.Lambda * L1Loss + MaskLoss
            Loss += PredictionLoss
        Loss /= B

        return Loss

    def L1(self, labels, predictions):
        # print(labels.size())
        # print(predictions.size())
        Loss = torch.mean(labels - predictions)
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)

        # print(torch.min(labels), torch.max(labels))
        # print(torch.min(predictions), torch.max(predictions))
        return Loss

class ADRegLoss(nn.Module):
    '''
    Computes the regularization loss for the autoencoder
    '''
    def __init__(self, Thresh=SINGLE_MASK_THRESH, RegLambda=REG_LAMBDA):
        super().__init__()
        self.MaskLoss = nn.BCELoss(reduction='mean')
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh
        self.RegLambda = RegLambda

    def forward(self, output, target):
        print("REG LOSS OUTPUT")
        print(len(output))
        print("REG LOSS TARGET")
        print(target)
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        LatentNorms = output[1]
        assert isinstance(LatentNorms, list) # For custom collate
        B = len(output) # Number of batches with custom collate
        Loss = 0
        for b in range(B):
            # Single batch version
            # TODO: Factor in the epoch
            # TODO: How does the stdev of the latent space factor in?
            LatentLoss = REG_LAMBDA * LatentNorms[b]
            Loss += LatentLoss
        Loss /= B

        return Loss