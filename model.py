'''
An MLP that predicts the surface depth along rays
'''

import torch
import torch.nn as nn
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def points(points):
    return points

def direction(points):
    dir = points[:,3:]-points[:,:3]
    norm = torch.linalg.norm(dir, dim=1)
    norm = torch.hstack([norm.reshape(-1,1)]*3)
    dir /= norm
    return torch.hstack([points[:,:3], dir])

def pluecker(points):
    dir = points[:,3:]-points[:,:3]
    norm = torch.linalg.norm(dir, dim=1)
    norm = torch.hstack([norm.reshape(-1,1)]*3)
    dir /= norm
    m = torch.cross(points[:,:3], dir, dim=1)
    return torch.hstack([dir, m])

def pos_encoding(points):
    return torch.tensor([[x for j in range(points.shape[1]) for x in utils.positional_encoding(points[i][j])] for i in range(points.shape[0])])

# Having the model change the input parameterization at inference time allows us to use a consistent input format so we don't have to change the testing script.
# For training the input will be provided with the preprocessing already applied so that it can be done in parallel in the dataloader
preprocessing_options = {
    "points": points,
    "direction": direction,
    "pluecker": pluecker,
}

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

        # self.pos_enc = pos_enc

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

    # def query_rays(self, points, directions):
    #     '''
    #     Returns a single depth value for each point, direction pair
    #     '''
    #     x = torch.hstack([points, directions])
    #     x = pos_encoding(x) if self.pos_enc else x
    #     x = x.to(device)
    #     _, intersect, depth = self.forward(x)
    #     return intersect, depth



class LF4D(nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    '''

    def __init__(self, input_size=6, n_layers=6, hidden_size=256, n_intersections=20, radius=1.25, coord_type="direction", pos_enc=True):
        super().__init__()
        # store args
        self.n_intersections = n_intersections
        self.preprocessing = preprocessing_options[coord_type]
        self.pos_enc = pos_enc
        self.radius = radius
        assert(n_layers > 1)

        # set which layers (aside from the first) should have the positional encoding passed in
        self.pos_enc_layers = [4]

        # Define the main network body
        main_layers = []
        main_layers.append(nn.Linear(input_size,hidden_size))
        for l in range(n_layers-1):
            if l+2 in self.pos_enc_layers:
                main_layers.append(nn.Linear(hidden_size+input_size, hidden_size))
            else:
                main_layers.append(nn.Linear(hidden_size, hidden_size))
        self.network = nn.ModuleList(main_layers)
        
        # Define the intersection head
        intersection_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, n_intersections+1) #+1 because we also need a zero intersection category
        ]
        self.intersection_head = nn.ModuleList(intersection_layers)

        # Define the depth head
        depth_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, n_intersections)
        ]
        self.depth_head = nn.ModuleList(depth_layers)

        # all_layers.append(nn.Linear(hidden_size, 2*n_intersections))
        # self.network = nn.ModuleList(all_layers)
        # other layers
        self.relu = nn.ReLU()
        # No layernorm for now
        # self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, input):
        x = input
        for i in range(len(self.network)):
            if i+1 in self.pos_enc_layers:
                x = self.network[i](torch.cat([input, x], dim=1))
            else:
                x = self.network[i](x)
            x = self.relu(x)
            # x = self.layernorm(x)
        
        # intersection head
        intersections = self.intersection_head[0](x)
        intersections = self.relu(intersections)
        # intersections = self.layernorm(intersections)
        intersections = self.intersection_head[1](intersections)
        # intersections = torch.sigmoid(intersections)

        # depth head
        depths = self.depth_head[0](x)
        depths = self.relu(depths)
        # depths = self.layernorm(depths)
        # enforce strictly increasing depth values
        depths = self.depth_head[1](depths)
        depths = self.relu(depths)
        depths = torch.cumsum(depths, dim=1)
        return intersections, depths

    def interior_depth(self, surface_points, interior_points):
        '''
        Coordinates - bounding sphere surface points and directions
        Interior_points - points within the bounding sphere that lie along the corresponding ray defined by coordinates
        Gives the positive depth to the next surface intersection from interior_points in the specified direction (direction implicitly defined by interior+surface point pair)
        Used for inference only
        Returns the integer number of intersections, as well as the intersection depths
        '''
        coordinates = self.preprocessing(surface_points)
        coordinates = pos_encoding(coordinates) if self.pos_enc else coordinates
        interior_distances = torch.sqrt(torch.sum(torch.square(surface_points[:,:3] - interior_points), dim=1))
        coordinates = coordinates.to(device)

        intersections, depths = self.forward(coordinates)

        intersections = intersections.cpu()
        depths = depths.cpu()
        depths -= torch.hstack([torch.reshape(interior_distances, (-1,1)),]*self.n_intersections)
        n_ints = torch.argmax(intersections, dim=1)

        # set invalid depths to inf
        # invalid depths are ones that occur before the interior point, and ones past the predicted number of intersections
        depths[depths < 0.] = float('inf')
        depth_mask = torch.nn.functional.one_hot(n_ints.to(torch.int64), intersections.shape[1])
        depth_mask = torch.cumsum(depth_mask, dim=1)
        depth_mask = depth_mask[:,:-1]
        depths[depth_mask.to(bool)] = float('inf')

        # depths = torch.min(depths, dim=1)[0]
        intersect = torch.min(depths, dim=1)[0] < float('inf')
        return intersect, depths, n_ints

    def query_rays(self, points, directions):
        '''
        This function can be used in the same way as a 5D ODF (i.e. query ANY point in 3D space, plus a direction, and get depth)
        Returns a single depth value for each point, direction pair
        Used for inference only
        '''
        # print("QUERY")
        # print(points)
        # print(directions)
        combine_tuple = lambda x: list(x[0]) + list(x[1])
        # the sphere intersections (two surface points) will be reparameterized in interior_depth if necessary (e.g. turned into surface point + direction)
        surface_intersections = torch.tensor([combine_tuple(utils.get_sphere_intersections(points[i], directions[i], self.radius)) for i in range(points.shape[0])])
        return self.interior_depth(surface_intersections, points)

