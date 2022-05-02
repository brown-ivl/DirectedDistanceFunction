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


            #PredMaskConfSig = self.Sigmoid(PredMaskConf)
            #PredMaskMaxConfVal = PredMaskConfSig
            #ValidRaysIdx = PredMaskMaxConfVal > self.Thresh  # Use predicted mask
            ValidRaysIdx = GTMask.to(torch.bool)  # Use ground truth mask
            #ValidRaysIdx = torch.logical_and(PredMaskMaxConfVal > self.Thresh, ValidDepthMask.to(torch.bool)) #Use both masks
            #Loss += 5.0 * self.L2(GTDepth[ValidRaysIdx], PredDepth[ValidRaysIdx])
            Mask = GTDepth<99
            GTDepth[torch.logical_not(ValidRaysIdx)] = 1.0
            Loss += 5.0 * self.L2(GTDepth[Mask], PredDepth[Mask])
        Loss /= B

        return Loss

    def L2(self, labels, predictions):
        Loss = torch.mean(torch.square(torch.clamp(labels, min=-0.5, max=0.5) - torch.clamp(predictions, min=-0.5, max=0.5)))
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
    def __init__(self):
        super().__init__()

    def forward(self, model, data, output, target):
        return self.computeLoss(model, data, output, target)

    def computeLoss(self, model, data, output, target):
        assert isinstance(output, list) # For custom collate
        B = len(target) # Number of batches with custom collate
        Loss = torch.tensor(0.).to(target[0][0].device)
        return Loss
        for b in range(B):
            # Single batch version
            GTMask, GTDepth = target[b]
            Mask = (GTDepth>99).reshape(-1)
            if torch.sum(Mask)==0:
                continue
            if len(output[b]) == 2:
                PredMaskConf, PredDepth = output[b]
            else:
                PredMaskConf, PredDepth, PredMaskConst, PredConst = output[b]
                PredDepth += self.Sigmoid(PredMaskConst)*PredConst
            PredDepthA = PredDepth[Mask].reshape(-1, 1)
            depthA = PredDepthA.detach()
            for j in [0.95, 1.05]:
                Coords = data[b][Mask]
                Shift = depthA*j#.to(target[0][0].device)
                Coords[:, :3] = Coords[:, :3]+Coords[:, 3:]*Shift
                outputB = model([Coords])[0]
                if len(outputB) == 2:
                    PredMaskConf, PredDepth = outputB
                else:
                    PredMaskConf, PredDepth, PredMaskConst, PredConst = outputB
                    PredDepth += self.Sigmoid(PredMaskConst)*PredConst
                PredDepthA = torch.cat([PredDepthA, PredDepth+Shift.detach()], -1)
            
            #gt = torch.mean(PredDepthA, dim=-1)
            gt = torch.mean(PredDepthA[:,1:], dim=-1)
            #print(gt.shape, PredDepthA.shape)
            #for j in range(PredDepthA.shape[-1]):
            #    Loss += 0.05 * self.L2(gt.detach(), PredDepthA[:, j])
            Loss += 0.05 * self.L2(gt.detach(), PredDepthA[:, 0])

            #Coords = data[b][Mask]
            #Shift = PredDepthA*(torch.rand(len(PredDepthA), 1)*0.1).to(target[0][0].device)
            #print(Coords.shape, Shift.shape, PredDepthA.shape)
            #Coords[:, :3] = Coords[:, :3]+Coords[:, 3:]*Shift
            #outputB = model([Coords])[0]
            #print(len(outputB))
            #if len(outputB) == 2:
            #    PredMaskConf, PredDepth = outputB
            #else:
            #    PredMaskConf, PredDepth, PredMaskConst, PredConst = outputB
            #    PredDepth += self.Sigmoid(PredMaskConst)*PredConst
            #PredDepthB = PredDepth
            
            #Loss += 0.1 * self.L2((PredDepthB+Shift).detach(), PredDepthA)
        Loss /= B

        return Loss

    def L2(self, labels, predictions):
        Loss = torch.mean(torch.square(torch.clamp(labels, min=-0.5, max=0.5) - torch.clamp(predictions, min=-0.5, max=0.5)))
        if math.isnan(Loss) or math.isinf(Loss):
            return torch.tensor(0)
        return Loss


class MultiViewLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, data, target):
        return self.computeLoss(model, data, target)

    def computeLoss(self, model, data, target): 
        B = len(target) # Number of batches with custom collate
        Loss = torch.tensor(0.).to(target[0][0].device)
        return Loss
        for b in range(B):
            # Single batch version
            GTMask, GTDepth = target[b]
            Mask = torch.logical_and(GTDepth<99, GTMask).reshape(-1)
            if torch.sum(Mask)==0:
                continue
            GTDepthOrg = GTDepth[Mask]
            Coords = data[b][Mask]
            CoordsRevRay = data[b][Mask]
            #Shift = torch.ones(GTDepthOrg.shape).to(target[0][0].device)*0.05 #torch.where(GTDepthOrg>0, 0.05, -0.05)
            Shift = torch.rand(GTDepthOrg.shape).to(target[0][0].device)*0.1#0.05           
            RayDir = torch.tensor(v3_utils.sphere_surface_sampler(len(Coords))).to(target[0][0].device)
            Coords[:,:3] = (Coords[:,:3]+Coords[:,3:]*GTDepthOrg)-Shift*RayDir
            Coords[:,3:] = RayDir
            CoordsRevRay[:,:3] = Coords[:,:3]
            CoordsRevRay[:,3:] *= -1
            #print(Coords)
            Coords = torch.vstack([Coords, CoordsRevRay])
            #print(Coords.reshape(2, -1, 6)[0,:,:], Coords.shape)
            output = model([Coords])[0]
            if len(output) == 2:
                PredMaskConf, PredDepth = output
            else:
                PredMaskConf, PredDepth, PredMaskConst, PredConst = output
                PredDepth += self.Sigmoid(PredMaskConst)*PredConst
            #Mask = (Shift*PredDepth)>0
            #if torch.sum(Mask)==0:
            #    continue
            #Loss += 1.0 * self.L2(torch.abs(Shift[Mask]), torch.clamp(torch.abs(PredDepth[Mask]), min=0., max=0.1))
            PredDepth, _ = torch.min(torch.abs(PredDepth.reshape(2, -1)), 0)
            #Mask = PredDepth<0.1
            Mask = torch.abs(Shift.reshape(-1)-PredDepth)<0.1
            #print(PredDepth.shape)
            #Loss += 0.1 * self.L2(Shift[Mask], PredDepth[Mask])
            Loss += 5 * self.L2(Shift, PredDepth)
        Loss /= B

        return Loss

    def L2(self, labels, predictions):
        Loss = torch.mean(torch.square(labels-predictions))
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
