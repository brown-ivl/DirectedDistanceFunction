'''
An MLP that predicts the surface depth along rays
'''

import torch
import torch.nn as nn

class SimpleMLP(nn.Module):

    def __init__(self,input_size=120,n_layers=5,hidden_size=200):
        super().__init__()
        assert(n_layers > 1)
        all_layers = []
        all_layers.append(nn.Linear(input_size,hidden_size))
        for _ in range(n_layers-2):
            all_layers.append(nn.Linear(hidden_size, hidden_size))
        all_layers.append(nn.Linear(hidden_size, 3))
        
        self.network = nn.ModuleList(all_layers)
        self.activation = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.network)-1):
            x = self.network[i](x)
            x = self.activation(x)
        x = self.network[-1](x)
        occ = torch.sigmoid(x[:,0])
        intersections = torch.sigmoid(x[:,1])
        depth = self.relu(x[:,2])
        return occ, intersections, depth


class AdaptedLFN(nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    '''

    def __init__(self, input_size=120, n_layers=6, hidden_size=256):
        super().__init__()
        assert(n_layers > 1)
        all_layers = []
        all_layers.append(nn.Linear(input_size,hidden_size))
        for _ in range(n_layers-2):
            all_layers.append(nn.Linear(hidden_size, hidden_size))
        all_layers.append(nn.Linear(hidden_size, 3))

        self.network = nn.ModuleList(all_layers)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x):
        for i in range(len(self.network)-1):
            x = self.network[i](x)
            x = self.relu(x)
            x = self.layernorm(x)
        x = self.network[-1](x)
        occ = torch.sigmoid(x[:,0])
        intersections = torch.sigmoid(x[:,1])
        depth = self.relu(x[:,2])
        return occ, intersections, depth


class LF4D(nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    '''

    def __init__(self, input_size=6, n_layers=6, hidden_size=256, n_intersections=20):
        super().__init__()
        self.n_intersections = n_intersections
        assert(n_layers > 1)
        all_layers = []
        all_layers.append(nn.Linear(input_size,hidden_size))
        for _ in range(n_layers-2):
            all_layers.append(nn.Linear(hidden_size, hidden_size))
        all_layers.append(nn.Linear(hidden_size, 2*n_intersections))

        self.network = nn.ModuleList(all_layers)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x):
        for i in range(len(self.network)-1):
            x = self.network[i](x)
            x = self.relu(x)
            x = self.layernorm(x)

        intersections = torch.sigmoid(x[:self.n_intersections//2])
        depths = self.relu(x[self.n_intersections//2:])
        return intersections, depths

    def interior_depth(self, coordinates, interior_points):
        '''
        Coordinates - bounding sphere surface points and directions
        Interior_points - points within the bounding sphere that lie along the corresponding ray defined by coordinates
        Gives the positive depth to the next surface intersection from interior_points in the specified direction
        '''
        interior_distances = torch.sqrt(torch.sum(torch.square(coordinates[:,:3] - interior_points), dim=1))
        intersections, depths = self.forward(coordinates)
        depths -= torch.hstack([torch.reshape(interior_distances, (-1,1)),]*self.n_intersections)
        depths[depths < 0.] = float('inf')
        depths[intersections == 0.] = float('inf')
        depths = torch.min(depths, dim=1)[0]
        return depths



        
        

