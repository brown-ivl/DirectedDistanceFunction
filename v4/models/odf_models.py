import torch
import torch.nn as nn
import sys
import v3_utils
from spherical_harmonics import SH


class ODFSingleV3(torch.nn.Module):
    # class LF4D(nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    '''

    def __init__(self, input_size=6, n_layers=6, hidden_size=256, radius=1.25, pos_enc=True):
        super().__init__()

        # store args
        n_intersections = 1 # Single intersection only
        self.pos_enc = pos_enc
        self.radius = radius
        assert (n_layers > 1)

        # set which layers (aside from the first) should have the positional encoding passed in
        if self.pos_enc:
            self.pos_enc_layers = [4]
            input_size = 120
        else:
            self.pos_enc_layers = []
            self.skip_connect = [4]

        # Define the main network body
        main_layers = []
        main_layers.append(nn.Linear(input_size, hidden_size))
        for l in range(n_layers - 1):
            if l + 2 in self.pos_enc_layers or l + 2 in self.skip_connect:
                main_layers.append(nn.Linear(hidden_size + input_size, hidden_size))
            else:
                main_layers.append(nn.Linear(hidden_size, hidden_size))
        self.network = nn.ModuleList(main_layers)

        # Define the intersection head
        intersection_layers = [
            #nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, n_intersections)
        ]
        self.intersection_head = nn.ModuleList(intersection_layers)

        # Define the depth head
        depth_layers = [
            #nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, n_intersections)
        ]
        self.depth_head = nn.ModuleList(depth_layers)


        # other layers
        self.relu = nn.ReLU()
        # No layernorm for now
        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, input):
        Input = input

        assert isinstance(input, list)
        B = len(Input)


        CollateList = [None] * B
        feature = None
        for b in range(B):
            if not self.pos_enc:
                x = Input[b]
                BInput = Input[b]
            else:
                x = v3_utils.positional_encoding_tensor(Input[b], L=10)
                BInput = v3_utils.positional_encoding_tensor(Input[b], L=10)
            for i in range(len(self.network)):
                if i + 1 in self.pos_enc_layers or i + 1 in self.skip_connect:
                    x = self.network[i](torch.cat([BInput, x], dim=1))
                else:
                    x = self.network[i](x)
                x = self.relu(x)
                x = self.layernorm(x)
                if i == 1:
                    feature = x

            # intersection head
            intersections = self.intersection_head[0](feature)
            #intersections = self.relu(intersections)
            # intersections = self.layernorm(intersections)
            #intersections = self.intersection_head[1](intersections)
            # intersections = torch.sigmoid(intersections)
            if len(intersections.size()) == 3:
                intersections = torch.squeeze(intersections, dim=1)

            # depth head
            depths = self.depth_head[0](x)
            #depths = self.relu(depths)
            # depths = self.layernorm(depths)
            #depths = self.depth_head[1](depths)
            if len(depths.size()) == 3:
                depths = torch.squeeze(depths, dim=1)

            CollateList[b] = (intersections, depths)

        return CollateList

class ODFADV3(torch.nn.Module):
    # class LF4D(nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    '''

    def __init__(self, input_size=6, latent_size=256, n_layers=6, hidden_size=256, radius=1.25, pos_enc=True):
        super().__init__()

        # store args
        n_intersections = 1 # Single intersection only
        self.pos_enc = pos_enc
        self.radius = radius
        assert (n_layers > 1)

        # set which layers (aside from the first) should have the positional encoding passed in
        if self.pos_enc:
            self.pos_enc_layers = [4]
            input_size = 120
        else:
            self.pos_enc_layers = []
            self.skip_connect = [4]

        # Define the main network body
        main_layers = []
        main_layers.append(nn.Linear(input_size + latent_size, hidden_size))
        for l in range(n_layers - 1):
            if l + 2 in self.pos_enc_layers or l + 2 in self.skip_connect:
                main_layers.append(nn.Linear(hidden_size + input_size + latent_size, hidden_size))
            else:
                main_layers.append(nn.Linear(hidden_size, hidden_size))
        self.network = nn.ModuleList(main_layers)

        # Define the intersection head
        intersection_layers = [
            #nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, n_intersections)
        ]
        self.intersection_head = nn.ModuleList(intersection_layers)

        # Define the depth head
        depth_layers = [
            #nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, n_intersections)
        ]
        self.depth_head = nn.ModuleList(depth_layers)


        # other layers
        self.relu = nn.ReLU()
        # No layernorm for now
        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, input, embeddings):
        Input = input

        assert isinstance(input, list)
        B = len(Input)


        CollateList = [None] * B
        feature = None
        for b in range(B):
            if not self.pos_enc:
                coordinates, indices = Input[b]
                lat_vecs = embeddings(indices)
                x = torch.cat((coordinates, lat_vecs), dim=-1)
                BInput = x
            else:
                x = v3_utils.positional_encoding_tensor(Input[b], L=10)
                BInput = v3_utils.positional_encoding_tensor(Input[b], L=10)
            for i in range(len(self.network)):
                if i + 1 in self.pos_enc_layers or i + 1 in self.skip_connect:
                    x = self.network[i](torch.cat([BInput, x], dim=1))
                else:
                    x = self.network[i](x)
                x = self.relu(x)
                x = self.layernorm(x)
                if i == 1:
                    feature = x

            # intersection head
            intersections = self.intersection_head[0](feature)
            #intersections = self.relu(intersections)
            # intersections = self.layernorm(intersections)
            #intersections = self.intersection_head[1](intersections)
            # intersections = torch.sigmoid(intersections)
            if len(intersections.size()) == 3:
                intersections = torch.squeeze(intersections, dim=1)

            # depth head
            depths = self.depth_head[0](x)
            #depths = self.relu(depths)
            # depths = self.layernorm(depths)
            #depths = self.depth_head[1](depths)
            if len(depths.size()) == 3:
                depths = torch.squeeze(depths, dim=1)

            CollateList[b] = (intersections, depths)

        return CollateList

class ODFSingleV3Constant(torch.nn.Module):
    # class LF4D(nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    '''

    def __init__(self, input_size=6, n_layers=6, hidden_size=256, radius=1.25, pos_enc=True):
        super().__init__()
        # store args
        self.pos_enc = pos_enc
        self.radius = radius
        assert (n_layers > 1)

        # set which layers (aside from the first) should have the positional encoding passed in
        if self.pos_enc:
            self.pos_enc_layers = [4]
            input_size = 120
        else:
            self.pos_enc_layers = []

        # Define the main network body
        main_layers = []
        main_layers.append(nn.Linear(input_size, hidden_size))
        for l in range(n_layers - 1):
            if l + 2 in self.pos_enc_layers:
                main_layers.append(nn.Linear(hidden_size + input_size, hidden_size))
            else:
                main_layers.append(nn.Linear(hidden_size, hidden_size))
        self.network = nn.ModuleList(main_layers)

        # Define the intersection head
        intersection_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, 1)
        ]
        self.intersection_head = nn.ModuleList(intersection_layers)

        # Define the depth head
        depth_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, 1)
        ]
        self.depth_head = nn.ModuleList(depth_layers)

        # constant head
        constant_layers = [torch.nn.Linear(hidden_size, hidden_size), torch.nn.Linear(hidden_size, 1)]
        self.constant_head = torch.nn.ModuleList(constant_layers)

        #constant mask head
        constant_mask_layers = [torch.nn.Linear(hidden_size, hidden_size), torch.nn.Linear(hidden_size, 1)]
        self.constant_mask_head = torch.nn.ModuleList(constant_mask_layers)

        # other layers
        self.relu = nn.ReLU()
        # No layernorm for now
        # self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, input):
        Input = input

        assert isinstance(input, list)
        B = len(Input)

        CollateList = [None] * B
        for b in range(B):
            if not self.pos_enc:
                x = Input[b]
                BInput = Input[b]
            else:
                x = v3_utils.positional_encoding_tensor(Input[b], L=10)
                BInput = v3_utils.positional_encoding_tensor(Input[b], L=10)
            for i in range(len(self.network)):
                if i + 1 in self.pos_enc_layers:
                    x = self.network[i](torch.cat([BInput, x], dim=1))
                else:
                    x = self.network[i](x)
                x = self.relu(x)
                # x = self.layernorm(x)

            # intersection head
            intersections = self.intersection_head[0](x)
            intersections = self.relu(intersections)
            intersections = self.intersection_head[1](intersections)
            if len(intersections.size()) == 3:
                intersections = torch.squeeze(intersections, dim=1)

            # depth head
            depths = self.depth_head[0](x)
            depths = self.relu(depths)
            depths = self.depth_head[1](depths)

            # constant head
            constants = self.constant_head[0](x)
            constants = self.relu(constants)
            constants = self.constant_head[1](constants)

            # constant mask head
            constant_mask = self.constant_mask_head[0](x)
            constant_mask = self.relu(constant_mask)
            constant_mask = self.constant_mask_head[1](constant_mask)

            if len(depths.size()) == 3:
                depths = torch.squeeze(depths, dim=1)

            CollateList[b] = (intersections, depths, constant_mask, constants)

        return CollateList


class ODFSingleV3SH(torch.nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    A linear combination of the spherical harmonic as differential forward map.
    '''

    def __init__(self, input_size=6, n_layers=6, hidden_size=256, radius=1.25, pos_enc=True, degrees=[2, 2], return_coeff=False):
        super().__init__()
        
        n_intersections = 1 # Single intersection only
        self.pos_enc = pos_enc
        self.radius = radius
        self.degrees = degrees
        self.input_size = input_size
        self.return_coeff = return_coeff
        assert (n_layers > 1)

        # set which layers (aside from the first) should have the positional encoding passed in
        if self.pos_enc:
            self.pos_enc_layers = [4]
            self.input_size = 120
            self.skip_connect = [4]
        else:
            self.pos_enc_layers = []
            self.skip_connect = [4]

        # Define the main network body
        main_layers = []
        main_layers.append(nn.Linear(input_size//2, hidden_size))
        for l in range(n_layers - 1):
            if l + 2 in self.pos_enc_layers or l + 2 in self.skip_connect:
                main_layers.append(nn.Linear(hidden_size + input_size//2, hidden_size))
            else:
                main_layers.append(nn.Linear(hidden_size, hidden_size))
        self.network = nn.ModuleList(main_layers)

        # Define the intersection head
        intersection_layers = [
            #nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, n_intersections*(self.degrees[1]+1)**2)
        ]
        self.intersection_head = nn.ModuleList(intersection_layers)

        # Define the depth head
        depth_layers = [
            #nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, n_intersections*(self.degrees[0]+1)**2)
        ]
        self.depth_head = nn.ModuleList(depth_layers)

        self.relu = nn.ReLU()
        # No layernorm for now
        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, input):
        Input = input

        assert isinstance(input, list)
        B = len(Input)

        CollateList = [None] * B
        features = None
        for b in range(B):
            if not self.pos_enc:
                # only use start point
                x = Input[b][:, :3]
                BInput = Input[b][:, :3]
            else:
                x = o2utils.positional_encoding_tensor(Input[b][:, :3], L=10)
                BInput = o2utils.positional_encoding_tensor(Input[b][:, :3], L=10)
            for i in range(len(self.network)):
                if i + 1 in self.pos_enc_layers or i + 1 in self.skip_connect:
                    x = self.network[i](torch.cat([BInput, x], dim=1))
                else:
                    x = self.network[i](x)
                x = self.relu(x)
                x = self.layernorm(x)
                if i == 1:
                    features = x

            # intersection head
            intersect_coeff = self.intersection_head[0](features)
            #intersect_coeff = self.relu(intersect_coeff)
            # intersections = self.layernorm(intersections)
            #intersect_coeff = self.intersection_head[1](intersect_coeff)
            # intersections = torch.sigmoid(intersections)

            # depth head
            depth_coeff = self.depth_head[0](x)
            #depth_coeff = self.relu(depth_coeff)
            # depths = self.layernorm(depths)
            #depth_coeff = self.depth_head[1](depth_coeff)
            
            if self.return_coeff:
                CollateList[b] = (intersect_coeff, depth_coeff)           
            else:
                cart = Input[b][:, 3:]
                sh = SH(max(self.degrees), cart)
                depths = sh.linear_combination(self.degrees[0], depth_coeff).view(-1, 1)
                intersections = sh.linear_combination(self.degrees[1], intersect_coeff).view(-1, 1)

                if len(intersections.size()) == 3:
                    intersections = torch.squeeze(intersections, dim=1)

                if len(depths.size()) == 3:
                    depths = torch.squeeze(depths, dim=1)
                CollateList[b] = (intersections, depths)

        return CollateList


class ODFSingleV3ConstantSH(torch.nn.Module):
    # class LF4D(nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    '''

    def __init__(self, input_size=6, n_layers=6, hidden_size=256, radius=1.25, pos_enc=True, degrees=[2, 2, 2, 2], return_coeff=False):
        super().__init__()

        # store args
        self.pos_enc = pos_enc
        self.radius = radius
        assert (n_layers > 1)
        self.degrees = degrees
        assert (len(degrees)==4)
        self.return_coeff = return_coeff

        # set which layers (aside from the first) should have the positional encoding passed in
        if self.pos_enc:
            self.pos_enc_layers = [4]
            input_size = 120
        else:
            self.pos_enc_layers = []

        # Define the main network body
        main_layers = []
        main_layers.append(nn.Linear(input_size//2, hidden_size))
        for l in range(n_layers - 1):
            if l + 2 in self.pos_enc_layers:
                main_layers.append(nn.Linear(hidden_size + input_size//2, hidden_size))
            else:
                main_layers.append(nn.Linear(hidden_size, hidden_size))
        self.network = nn.ModuleList(main_layers)

        # Define the intersection head
        intersection_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, (self.degrees[1]+1)**2)
        ]
        self.intersection_head = nn.ModuleList(intersection_layers)

        # Define the depth head
        depth_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, (self.degrees[0]+1)**2)
        ]
        self.depth_head = nn.ModuleList(depth_layers)

        # constant head
        constant_layers = [
            torch.nn.Linear(hidden_size, hidden_size), 
            torch.nn.Linear(hidden_size, (self.degrees[2]+1)**2)
        ]
        self.constant_head = torch.nn.ModuleList(constant_layers)

        #constant mask head
        constant_mask_layers = [
            torch.nn.Linear(hidden_size, hidden_size), 
            torch.nn.Linear(hidden_size, (self.degrees[3]+1)**2)
        ]
        self.constant_mask_head = torch.nn.ModuleList(constant_mask_layers)

        # other layers
        self.relu = nn.ReLU()
        # No layernorm for now
        # self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, input):
        Input = input

        assert isinstance(input, list)
        B = len(Input)

        CollateList = [None] * B
        for b in range(B):
            if not self.pos_enc:
                x = Input[b][:, :3]
                BInput = Input[b][:, :3]
            else:
                x = o2utils.positional_encoding_tensor(Input[b][:, :3], L=10)
                BInput = o2utils.positional_encoding_tensor(Input[b][:, :3], L=10)
            for i in range(len(self.network)):
                if i + 1 in self.pos_enc_layers:
                    x = self.network[i](torch.cat([BInput, x], dim=1))
                else:
                    x = self.network[i](x)
                x = self.relu(x)
                # x = self.layernorm(x)

            # intersection head
            intersections_coeff = self.intersection_head[0](x)
            intersections_coeff = self.relu(intersections_coeff)
            intersections_coeff = self.intersection_head[1](intersections_coeff)
          
            # depth head
            depths_coeff = self.depth_head[0](x)
            depths_coeff = self.relu(depths_coeff)
            depths_coeff = self.depth_head[1](depths_coeff)

            # constant head
            constants_coeff = self.constant_head[0](x)
            constants_coeff = self.relu(constants_coeff)
            constants_coeff = self.constant_head[1](constants_coeff)

            # constant mask head
            constant_mask_coeff = self.constant_mask_head[0](x)
            constant_mask_coeff = self.relu(constant_mask_coeff)
            constant_mask_coeff = self.constant_mask_head[1](constant_mask_coeff)

            if self.return_coeff:
                CollateList[b] = (intersections_coeff, depths_coeff, constant_mask_coeff, constants_coeff)
            else:
                cart = Input[b][:, 3:]
                sh = SH(max(self.degrees), cart)

                depths = sh.linear_combination(self.degrees[0], depths_coeff, clear=True).view(-1, 1)
                intersections = sh.linear_combination(self.degrees[1], intersections_coeff).view(-1, 1)
                constants = sh.linear_combination(self.degrees[2], constants_coeff).view(-1, 1)
                constant_mask = sh.linear_combination(self.degrees[3], constant_mask_coeff).view(-1, 1)

                if len(intersections.size()) == 3:
                    intersections = torch.squeeze(intersections, dim=1)
                if len(depths.size()) == 3:
                    depths = torch.squeeze(depths, dim=1)
                CollateList[b] = (intersections, depths, constant_mask, constants)

        return CollateList
