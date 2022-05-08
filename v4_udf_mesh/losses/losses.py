import torch
import torch.nn as nn
import math
import v3_utils

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
    def __init__(self, Thresh=v3_utils.INTERSECTION_MASK_THRESHOLD):
        super().__init__()
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh

    def forward(self, output, target, cp=0.5):
        return self.computeLoss(output, target, cp)

    def computeLoss(self, output, target, cp=0.5):
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


            #PredMaskConfSig = self.Sigmoid(PredMaskConf)
            #PredMaskMaxConfVal = PredMaskConfSig
            #ValidRaysIdx = PredMaskMaxConfVal > self.Thresh  # Use predicted mask
            ValidRaysIdx = GTMask.to(torch.bool)  # Use ground truth mask
            #ValidRaysIdx = torch.logical_and(PredMaskMaxConfVal > self.Thresh, ValidDepthMask.to(torch.bool)) #Use both masks
            #Loss += 5.0 * self.L2(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])
            #Mask = GTDepth<99
            GTDepth[torch.logical_not(ValidRaysIdx)] = 1.0
            Loss += 5.0 * self.L2(GTDepth, PredDepth, cp)
        Loss /= B

        return Loss

    def L2(self, labels, predictions, cp=0.5):
        Loss = torch.mean(torch.square(torch.minimum(labels, torch.tensor(cp)) - torch.minimum(predictions, torch.tensor(cp))))
        #Loss = torch.mean(torch.square(labels-predictions))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)
        return Loss

class IntersectionLoss(nn.Module):
    def __init__(self, Thresh=v3_utils.INTERSECTION_MASK_THRESHOLD):
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

            Loss += 1.0 * self.BCE(PredMaskMaxConfVal.to(torch.float), GTMask.to(torch.float))

        Loss /= B

        return Loss


class MultiViewRayLoss(nn.Module):
    def __init__(self, Thresh=v3_utils.INTERSECTION_MASK_THRESHOLD):
        super().__init__()
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh

    def forward(self, model, data, target, maxShift):
        return self.computeLoss(model, data, target, maxShift)

    def computeLoss(self, model, data, target, maxShift=0.1):
        B = len(target) # Number of batches with custom collate
        Loss = torch.tensor(0.).to(target[0][0].device)
        return Loss
        for b in range(B):
            # Single batch version
            GTMask, GTDepth = target[b]
            Mask = torch.logical_and(GTMask, (GTDepth<0.1)).reshape(-1)
            if torch.sum(Mask)==0:
                continue
            Coords = data[b][Mask]
            Shift = torch.rand((len(Coords), 1))*maxShift #(maxShift-0.1)+0.1
            Shift = Shift.to(target[0][0].device)
            #print(Shift.shape, GTDepth[Mask].shape)
            Coords[:, :3] = Coords[:, :3]+Coords[:, 3:]*(GTDepth[Mask]-Shift)
            output = model([Coords])[0]
            #PredMaskConf, PredDepth = output
            if len(output) == 2:
                PredMaskConf, PredDepth = output
            else:
                PredMaskConf, PredDepth, PredMaskConst, PredConst = output
                PredDepth += self.Sigmoid(PredMaskConst)*PredConst
            Mask = (PredDepth-Shift)>=-0.1
            Loss += 0.5 * self.L2(Shift[Mask], PredDepth[Mask])
            
            #Mask = torch.logical_not(Mask).reshape(-1)
            #Coords = Coords[Mask]
            #ShiftNext = PredDepth[Mask].detach()*0.95
            #Coords[:, :3] = Coords[:, :3]+Coords[:, 3:]*ShiftNext
            #outputNext = model([Coords])[0]
            #PredMaskConf, PredDepthNext = outputNext
            #Loss += 0.5 * self.L2(ShiftNext, PredDepth[Mask]-PredDepthNext.detach())

            Mask = torch.logical_not(Mask).reshape(-1)
            Coords = Coords[Mask]
            ShiftNext = PredDepth[Mask].detach()*0.95
            Coords[:, :3] = Coords[:, :3]+Coords[:, 3:]*ShiftNext
            size = len(Coords)
            Coords = torch.vstack([Coords, Coords])
            Coords[size:, 3:] *= -1
            outputNext = model([Coords])[0]
            if len(outputNext) == 2:
                PredMaskConfNext, PredDepthNext = outputNext
            else:
                PredMaskConfNext, PredDepthNext, PredMaskConstNext, PredConstNext = outputNext
                PredDepthNext += self.Sigmoid(PredMaskConstNext)*PredConstNext
            #PredMaskConf, PredDepthNext = outputNext
            # overshooting
            MaskOver = PredDepthNext[size:, :]<ShiftNext
            PredDepthNext[:size, :][MaskOver] = -PredDepthNext[size:, :][MaskOver]
            Loss += 0.5 * self.L2(ShiftNext+PredDepthNext[:size, :].detach(), PredDepth[Mask])
            
        Loss /=B
        return Loss

    def L2(self, labels, predictions):
        Loss = torch.mean(torch.square(labels-predictions))
        #Loss = torch.mean(torch.square(labels-predictions))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)
        return Loss


class DepthFieldRegularizingLoss(nn.Module):

    def __init__(self, Thresh=v3_utils.INTERSECTION_MASK_THRESHOLD):
        super().__init__()
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh

        
    def forward(self, model, data):
        return self.computeLoss(model, data)
    
    def computeLoss(self, model, data):

        assert isinstance(data, list) # For custom collate
        B = len(data) # Number of batches with custom collate
        #Loss = 0
        Loss = torch.tensor(0.).to(data[0].device)
        #return Loss
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

    def __init__(self, Thresh=v3_utils.INTERSECTION_MASK_THRESHOLD):
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
