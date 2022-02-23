import torch
import torch.nn as nn
import math
import v3_utils

SINGLE_MASK_THRESH = 0.5

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True)
    return points_grad

class DepthLoss(nn.Module):
    def __init__(self, Thresh=SINGLE_MASK_THRESH):
        super().__init__()
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        assert isinstance(output, list) # For custom collate
        B = len(target) # Number of batches with custom collate
        Loss = torch.tensor(0.).to(target[0][0].device)
        for b in range(B):
            # Single batch version
            GTMask, GTDepth = target[b]

            if len(output[b]) == 2:
                PredMaskConf, PredDepth = output[b]
            else:
                PredMaskConf, PredDepth, PredMaskConst, PredConst = output[b]
                PredDepth += self.Sigmoid(PredMaskConst)*PredConst


            PredMaskConfSig = self.Sigmoid(PredMaskConf)
            PredMaskMaxConfVal = PredMaskConfSig
            ValidRaysIdx = PredMaskMaxConfVal > self.Thresh  # Use predicted mask
            # ValidRaysIdx = GTMask.to(torch.bool)  # Use ground truth mask
            # ValidRaysIdx = torch.logical_and(PredMaskMaxConfVal > self.Thresh, GTMask.to(torch.bool)) #Use both masks
            Loss += 5.0 * self.L2(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])
        Loss /= B

        return Loss

    def L2(self, labels, predictions):
        Loss = torch.mean(torch.square(labels - predictions))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)
        return Loss

class IntersectionLoss(nn.Module):
    def __init__(self, Thresh=SINGLE_MASK_THRESH):
        super().__init__()
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh
        self.BCE = nn.BCELoss(reduction='mean')


    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        assert isinstance(output, list) # For custom collate
        B = len(target) # Number of batches with custom collate
        Loss = 0
        for b in range(B):
            # Single batch version
            GTMask, _ = target[b]

            if len(output[b]) == 2:
                PredMaskConf, _ = output[b]
            else:
                PredMaskConf, _, _, _ = output[b]

            PredMaskConfSig = self.Sigmoid(PredMaskConf)
            PredMaskMaxConfVal = PredMaskConfSig

            Loss += self.BCE(PredMaskMaxConfVal.to(torch.float), GTMask.to(torch.float))

        Loss /= B

        return Loss


class DepthFieldRegularizingLoss(nn.Module):

    def __init__(self, Thresh=SINGLE_MASK_THRESH):
        super().__init__()
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh

        
    def forward(self, model, data):
        return self.computeLoss(model, data)
    
    def computeLoss(self, model, data):

        assert isinstance(data, list) # For custom collate
        B = len(data) # Number of batches with custom collate
        Loss = 0

        for b in range(B):
            # Single batch version
            TrainCoords = data[b]
            OtherCoords = torch.tensor(v3_utils.odf_domain_sampler(TrainCoords.shape[0]), dtype=torch.float32).to(TrainCoords.device)
            Coords = torch.cat([TrainCoords, OtherCoords], dim=0)

            Coords.requires_grad_()
            output = model([Coords])[0]
            if len(output) == 2:
                PredMaskConf, PredDepth = output
            else:
                PredMaskConf, PredDepth, _, _ = output
            PredMaskConfSig = self.Sigmoid(PredMaskConf)
            intersections = PredMaskConfSig.squeeze()
            depths = PredDepth

            x_grads = gradient(Coords, depths)[0][...,:3]

            odf_gradient_directions = Coords[:,3:]


            if torch.sum(intersections > self.Thresh) != 0.:
                grad_dir_loss = torch.mean(torch.abs(torch.sum(odf_gradient_directions[intersections>self.Thresh]*x_grads[intersections>self.Thresh], dim=-1) + 1.))
            else:
                grad_dir_loss = torch.tensor(0.).to(TrainCoords.device)

            Loss += 1.0 * grad_dir_loss

        Loss /= B

        return Loss


class ConstantRegularizingLoss(nn.Module):

    def __init__(self, Thresh=SINGLE_MASK_THRESH):
        super().__init__()
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh
        
    def forward(self, model, data):
        return self.computeLoss(model, data)
    
    def computeLoss(self, model, data):
        assert isinstance(data, list) # For custom collate
        B = len(data) # Number of batches with custom collate
        Loss = 0

        for b in range(B):
            # Single batch version
            TrainCoords = data[b]
            OtherCoords = torch.tensor(v3_utils.odf_domain_sampler(TrainCoords.shape[0]), dtype=torch.float32).to(TrainCoords.device)
            Coords = torch.cat([TrainCoords, OtherCoords], dim=0)

            Coords.requires_grad_()
            output = model([Coords])[0]
            assert(len(output) > 2)
            _, _, PredMaskConst, PredConst = output
            PredMaskConstSig = self.Sigmoid(PredMaskConst)
            constant_mask = PredMaskConstSig.squeeze()
            constant = PredConst

            x_grads = gradient(Coords, constant)[0][...,:3]

            odf_gradient_directions = Coords[:,3:]


            if torch.sum(constant_mask > self.Thresh) != 0.:
                grad_dir_loss = torch.mean(torch.abs(torch.sum(odf_gradient_directions[constant_mask>self.Thresh]*x_grads[constant_mask>self.Thresh], dim=-1)))
            else:
                grad_dir_loss = torch.tensor(0.).to(TrainCoords.device)

            Loss += 1.0 * grad_dir_loss

        Loss /= B

        return Loss