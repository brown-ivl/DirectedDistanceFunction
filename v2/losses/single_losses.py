import torch
import torch.nn as nn
import math
import odf_v2_utils as o2utils

SINGLE_MASK_THRESH = 0.7
SINGLE_L2_LAMBDA = 5.0
SINGLE_L1_LAMBDA = 5.0
# REG_LAMBDA = 1e-4
REG_LAMBDA = 1e-1

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True)
        # [0][:, -3:]
    # print("GRADIENT INFO")
    # print(inputs.shape)
    # print(outputs.shape)
    # print(len(points_grad))
    # print(points_grad)
    # # print(points_grad.shape)
    # print("=============")
    return points_grad



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

            if len(output[b]) == 2:
                PredMaskConf, PredDepth = output[b]
            else:
                PredMaskConf, PredDepth, PredMaskConst, PredConst = output[b]
                PredDepth += self.Sigmoid(PredMaskConst)*PredConst


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


class DepthFieldRegularizingLossGrad(nn.Module):

    def __init__(self):
        super().__init__()
        self.GradientLoss = nn.MSELoss()
        self.Sigmoid = nn.Sigmoid()

        
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
            TrainCoords = data[b]
            OtherCoords = torch.tensor(o2utils.odf_domain_sampler(TrainCoords.shape[0]), dtype=torch.float32).to(TrainCoords.device)
            Coords = torch.cat([TrainCoords, OtherCoords], dim=0)

            Coords.requires_grad_()
            # intersections, depths = model([Coords], {})[0]
            # intersections = intersections.squeeze()
            output = model([Coords], {})[0]
            if len(output) == 2:
                PredMaskConf, PredDepth = output
            else:
                PredMaskConf, PredDepth, _, _ = output
            intersections = PredMaskConf.squeeze()
            depths = PredDepth

            x_grads = gradient(Coords, depths)[0][...,:3]
            # gradient_norm_mse = torch.mean(torch.square(torch.linalg.norm(x_grads, axis=-1) - 1.))

            # Loss += gradient_norm_mse

            odf_gradient_directions = Coords[:,3:]


            if torch.sum(intersections > 0.5) != 0.:
                # grad_dir_loss = torch.mean(torch.linalg.norm((odf_gradient_directions[intersections>0.5] - x_grads[intersections > 0.5]).abs(), dim=-1))
                # grad_dir_loss = torch.mean(torch.linalg.norm((odf_gradient_directions[intersections>0.5] - x_grads[intersections > 0.5]).abs(), dim=-1))
                grad_dir_loss = torch.mean(torch.square(torch.sum(odf_gradient_directions[intersections>0.5]*x_grads[intersections>0.5], dim=-1) + 1.))
            else:
                grad_dir_loss = torch.tensor(0.).to(TrainCoords.device)
            # print(odf_gradient_directions[intersections>0.5].shape)
            # print(grad_dir_loss)
            # if torch.sum(intersections > 0.5) == 0.:
            #     print(grad_dir_loss)
                    # normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
            Loss += 1.0 * grad_dir_loss

        Loss /= B

        return Loss


class ConstantRegularizingLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.GradientLoss = nn.MSELoss()
        self.Sigmoid = nn.Sigmoid()
        
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
            TrainCoords = data[b]
            OtherCoords = torch.tensor(o2utils.odf_domain_sampler(TrainCoords.shape[0]), dtype=torch.float32).to(TrainCoords.device)
            Coords = torch.cat([TrainCoords, OtherCoords], dim=0)

            Coords.requires_grad_()
            output = model([Coords], {})[0]
            assert(len(output) > 2)
            PredMaskConf, PredDepth, PredMaskConst, PredConst = output
            constant_mask = PredMaskConf.squeeze()
            constant = PredConst

            x_grads = gradient(Coords, constant)[0][...,:3]
            # gradient_norm_mse = torch.mean(torch.square(torch.linalg.norm(x_grads, axis=-1) - 1.))

            # Loss += gradient_norm_mse

            odf_gradient_directions = Coords[:,3:]


            if torch.sum(constant_mask > 0.5) != 0.:
                # grad_dir_loss = torch.mean(torch.linalg.norm((odf_gradient_directions[intersections>0.5] - x_grads[intersections > 0.5]).abs(), dim=-1))
                # grad_dir_loss = torch.mean(torch.linalg.norm((odf_gradient_directions[intersections>0.5] - x_grads[intersections > 0.5]).abs(), dim=-1))
                grad_dir_loss = torch.mean(torch.square(torch.sum(odf_gradient_directions[constant_mask>0.5]*x_grads[constant_mask>0.5], dim=-1)))
            else:
                grad_dir_loss = torch.tensor(0.).to(TrainCoords.device)
            # print(odf_gradient_directions[intersections>0.5].shape)
            # print(grad_dir_loss)
            # if torch.sum(intersections > 0.5) == 0.:
            #     print(grad_dir_loss)
                    # normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
            Loss += 1.0 * grad_dir_loss

        Loss /= B

        return Loss


class DepthFieldRegularizingLossJacobian(nn.Module):

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
            Coords = Coords[:5, ...]
            instance_loss = 0.

            # The size of the jacobian is input x output, so we get CUDA OOM if our batch is too large
            jacobian_batch_size = 40
            n_jacobian_batches = math.ceil(Coords.shape[0]/jacobian_batch_size)
            for jacobian_batch_number in range(n_jacobian_batches):
                curr_coords = Coords[jacobian_batch_number*jacobian_batch_size:(jacobian_batch_number+1)*jacobian_batch_size,:]
                jacobian = torch.autograd.functional.jacobian(lambda x: model([x], {})[0][1], curr_coords, create_graph=True, vectorize=True)
                directional_gradients = jacobian[torch.arange(jacobian.shape[0]),0,torch.arange(jacobian.shape[2]), :3]
                # print("+++++++++++++++++++++")
                # print(Coords[...,3:].shape)
                gradient_dots = torch.sum(curr_coords[...,3:]*directional_gradients, dim=-1)
                # gradient_norm = torch.linalg.norm(directional_gradients, dim=-1)
                # The dots of the viewing directions and their gradients should be -1
                mse_dots = torch.square(gradient_dots+1.)
                # mse_norm = torch.square(gradient_norm-1.)
                instance_loss += torch.sum(mse_dots)
            
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