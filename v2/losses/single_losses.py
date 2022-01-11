import torch
import torch.nn as nn
import math

SINGLE_MASK_THRESH = 0.7
SINGLE_L2_LAMBDA = 5.0
SINGLE_L1_LAMBDA = 5.0
# REG_LAMBDA = 1e-4
REG_LAMBDA = 1e-1

class SingleDepthBCELoss(nn.Module):
    Thresh = SINGLE_MASK_THRESH
    Lambda = SINGLE_L2_LAMBDA
    def __init__(self, Thresh=SINGLE_MASK_THRESH, Lambda=SINGLE_L2_LAMBDA):
        super().__init__()
        self.MaskLoss = nn.BCELoss(reduction='mean')
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh
        self.Lambda = Lambda

    def forward(self, output, target, otherInputs={}):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        # print(f"output type: {type(output)}")
        # print(output[0][1])
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

class DepthFieldRegularizingLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.GradientLoss = nn.MSELoss()
        
    def forward(self, output, target, otherInputs={}):
        return self.computeLoss(output, target, otherInputs=otherInputs)
    
    def computeLoss(self, output, target, otherInputs={}):
        assert("model" in otherInputs)
        assert("data" in otherInputs)
        data = otherInputs["data"]
        model = otherInputs["model"]

        assert isinstance(output, list) # For custom collate
        B = len(target) # Number of batches with custom collate
        Loss = 0

        for b in range(B):
            # Single batch version
            Coords = data[b]
            instance_loss = 0.

            # The size of the jacobian is input x output, so we get CUDA OOM if our batch is too large
            jacobian_batch_size = 40
            n_jacobian_batches = math.ceil(Coords.shape[0]/jacobian_batch_size)
            for jacobian_batch_number in range(n_jacobian_batches):
                curr_coords = Coords[jacobian_batch_number*jacobian_batch_size:(jacobian_batch_number+1)*jacobian_batch_size,:]
                jacobian = torch.autograd.functional.jacobian(lambda x: model([x], {})[0][1], curr_coords, create_graph=True, vectorize=True)
                directional_gradients = jacobian[torch.arange(jacobian.shape[0]),0,torch.arange(jacobian.shape[2]), 3:]
                # print("+++++++++++++++++++++")
                # print(Coords[...,3:].shape)
                # print(directional_gradients.shape)
                gradient_dots = torch.sum(curr_coords[...,3:]*directional_gradients, dim=-1)
                # The dots of the viewing directions and their gradients should be -1
                mse = torch.square(gradient_dots+1.)
                instance_loss += torch.sum(mse)
            
            Loss += instance_loss/Coords.shape[0]
        Loss /= B

        return Loss



class ADCombinedLoss(nn.Module):
    '''
    Computes the prediction loss for the autodecoder
    '''

    def __init__(self, Thresh=SINGLE_MASK_THRESH, Lambda=SINGLE_L2_LAMBDA, RegLambda=REG_LAMBDA, use_l2=False):
        super().__init__()
        self.MaskLoss = nn.BCELoss(reduction='mean')
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh
        self.Lambda = Lambda
        self.RegLambda = RegLambda
        self.use_l2 = use_l2

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        Output = output[0]
        LatentNorms = output[1]
        assert isinstance(LatentNorms, list) # For custom collate
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
            if not self.use_l2:
                DepthLoss = self.L1(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])
            else:
                DepthLoss = self.L2(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])
            PredictionLoss = self.Lambda * DepthLoss + MaskLoss
            LatentLoss = REG_LAMBDA * LatentNorms[b]
            Loss += PredictionLoss + LatentLoss
        Loss /= B

        return Loss

    def L1(self, labels, predictions):
        Loss = torch.mean(labels - predictions)
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)
        return Loss

    def L2(self, labels, predictions):
        Loss = torch.mean(torch.square(labels - predictions))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)
        return Loss

class ADPredLoss(nn.Module):
    '''
    Computes the prediction loss for the autodecoder
    '''

    def __init__(self, Thresh=SINGLE_MASK_THRESH, Lambda=SINGLE_L2_LAMBDA, use_l2=False):
        super().__init__()
        self.MaskLoss = nn.BCELoss(reduction='mean')
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh
        self.Lambda = Lambda
        self.use_l2 = use_l2

    def forward(self, output, target):
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
            print(f"Total intersecting: {torch.sum(ValidRaysIdx)}")
            if not self.use_l2:
                DepthLoss = self.L1(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])
            else:
                DepthLoss = self.L2(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])
            print(f"Depth Loss: {DepthLoss}")
            print(f"Mask Loss: {MaskLoss}")
            PredictionLoss = self.Lambda * DepthLoss + MaskLoss
            Loss += PredictionLoss
        Loss /= B

        return Loss

    def L1(self, labels, predictions):
        Loss = torch.mean(torch.abs(labels - predictions))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)
        return Loss

    def L2(self, labels, predictions):
        Loss = torch.mean(torch.square(labels - predictions))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)
        return Loss

class ADRegLoss(nn.Module):
    '''
    Computes the regularization loss for the autoencoder
    '''
    def __init__(self, RegLambda=REG_LAMBDA):
        super().__init__()
        self.MaskLoss = nn.BCELoss(reduction='mean')
        self.Sigmoid = nn.Sigmoid()
        self.RegLambda = RegLambda

    def forward(self, output, target):

        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        LatentVectors = output[1]
        assert isinstance(LatentVectors, list) # For custom collate
        B = len(output) # Number of batches with custom collate
        Loss = 0
        for b in range(B):
            # Single batch version
            # TODO: Factor in the epoch
            # TODO: How does the stdev of the latent space factor in?
            LatentLoss = self.RegLambda * torch.mean(torch.norm(LatentVectors[b], dim=-1))
            Loss += LatentLoss
        Loss /= B
        print(f"Reg Loss: {Loss}")
        return Loss