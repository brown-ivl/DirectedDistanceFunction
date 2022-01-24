import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad

# partial derivative calculation from IGR
# https://github.com/amosgropp/IGR
def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True)
        # [0][:, -3:]
    return points_grad

class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3,1)
        with torch.no_grad():
            self.layer.weight.copy_(torch.tensor([1.,1.,1.]))

    def forward(self, input):
        return self.layer(input)


model = SimpleModel()
X = torch.tensor([[2.,3.,4.], [5.,6.,7.], [10.,11.,12.], [13.,14.,15.]])
X = torch.tensor([[1.,1.,1.], [2.,2.,2.]])

# jacobian = torch.autograd.functional.jacobian(model, X, create_graph=True)
# print("Jacobian Shape:")
# print(jacobian.shape)
# print("Input Shape:")
# print(X.shape)
# print("Output Shape:")
# print(model(X).shape)
# print("Jacobian-")
# print(jacobian,"\n\n")
# # print(jacobian[np.arange(jacobian.shape[0]),0,np.arange(jacobian.shape[-2]), :])
# jacobian = jacobian[np.arange(jacobian.shape[0]),0,np.arange(jacobian.shape[-2]), :]

# grad_norm = torch.linalg.norm(jacobian, axis=-1)
# grad_norm_mean = torch.mean(grad_norm)
# print(grad_norm)
# grad_norm_mean.backward()
# print("Layer Gradient w.r.t. Jacobian Norm-")
# print(model.layer.weight.grad)

# Gradients wrt Jacobian Norm should all be 0.577
# d/dx  sqrt(x^2 + 2)  = x / sqrt(x^2 + 2)
# substitute in the weight (1.) 
# = 1/sqrt(3) = 0.577




# using torch.autograd.grad instead of jacobian

# mnfld_pnts = self.add_latent(mnfld_pnts, indices)
# nonmnfld_pnts = self.add_latent(nonmnfld_pnts, indices)

# # forward pass

# mnfld_pnts.requires_grad_()
# nonmnfld_pnts.requires_grad_()

# mnfld_pred = self.network(mnfld_pnts)
# nonmnfld_pred = self.network(nonmnfld_pnts)

# mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
# nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)


X.requires_grad_()
print(X.requires_grad)
print(X.grad)
X_pred = model(X)
# loss = ((torch.linalg.norm(X_pred, dim=-1) - 1) ** 2).mean()

x_grads = gradient(X, X_pred)[0]
print("X Grads: ")
print(x_grads.shape)
print(x_grads)

gradient_norm = torch.sum(torch.linalg.norm(x_grads, axis=-1))
# gradient_norm = torch.sum(x_grads)
gradient_norm.backward()
print(X.grad)
print(model.layer.weight.grad)

Z = torch.tensor([[1.,1.,1.]])
Z.requires_grad_()
Z_pred = model(Z)
z_grads = gradient(Z, Z_pred)[0]
z_norm = torch.sum(torch.linalg.norm(z_grads, axis=-1))
z_norm.backward()

print(model.layer.weight.grad)


