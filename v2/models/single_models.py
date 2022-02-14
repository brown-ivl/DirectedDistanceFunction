import torch
import torch.nn as nn
import beacon.supernet as supernet
import sys
import odf_v2_utils as o2utils

class ODFSingleV3(supernet.SuperNet):
    # class LF4D(nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    '''

    def __init__(self, input_size=6, n_layers=6, hidden_size=256, radius=1.25, coord_type='direction', pos_enc=True, Args=None):
        super().__init__(Args=Args)

        # store args
        n_intersections = 1 # Single intersection only
        self.preprocessing = coord_type
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
            nn.Linear(hidden_size, n_intersections)
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

    def forward(self, input, otherParameters):
        Input = input

        assert isinstance(input, list)
        B = len(Input)

        # BIntersects = [None] * B
        # BDepths = [None] * B
        CollateList = [None] * B
        for b in range(B):
            if not self.pos_enc:
                x = Input[b]
                BInput = Input[b]
            else:
                x = o2utils.positional_encoding_tensor(Input[b], L=10)
                BInput = o2utils.positional_encoding_tensor(Input[b], L=10)
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
            # intersections = self.layernorm(intersections)
            intersections = self.intersection_head[1](intersections)
            # intersections = torch.sigmoid(intersections)
            if len(intersections.size()) == 3:
                intersections = torch.squeeze(intersections, dim=1)

            # depth head
            depths = self.depth_head[0](x)
            depths = self.relu(depths)
            # depths = self.layernorm(depths)
            # enforce strictly increasing depth values
            depths = self.depth_head[1](depths)
            # depths = self.relu(depths) # todo: Avoid relu at the last layer?
            # depths = torch.cumsum(depths, dim=1)
            if len(depths.size()) == 3:
                depths = torch.squeeze(depths, dim=1)
            # BIntersects[b] = intersections
            # BDepths[b] = depths
            CollateList[b] = (intersections, depths)

        return CollateList
        # return BIntersects, BDepths

class ODFSingleV3Constant(supernet.SuperNet):
    # class LF4D(nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    '''

    def __init__(self, input_size=6, n_layers=6, hidden_size=256, radius=1.25, coord_type='direction', pos_enc=True, Args=None):
        super().__init__(Args=Args)

        # store args
        self.preprocessing = coord_type
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

    def forward(self, input, otherParameters):
        Input = input

        assert isinstance(input, list)
        B = len(Input)

        # BIntersects = [None] * B
        # BDepths = [None] * B
        CollateList = [None] * B
        for b in range(B):
            if not self.pos_enc:
                x = Input[b]
                BInput = Input[b]
            else:
                x = o2utils.positional_encoding_tensor(Input[b], L=10)
                BInput = o2utils.positional_encoding_tensor(Input[b], L=10)
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
            # BIntersects[b] = intersections
            # BDepths[b] = depths
            CollateList[b] = (intersections, depths, constant_mask, constants)

        return CollateList

class LF4DSingle(supernet.SuperNet):
    # class LF4D(nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    '''

    def __init__(self, input_size=6, n_layers=6, hidden_size=256, radius=1.25, coord_type='direction', pos_enc=True, Args=None):
        super().__init__(Args=Args)

        # store args
        n_intersections = 1 # Single intersection only
        self.preprocessing = coord_type
        self.pos_enc = pos_enc
        self.radius = radius
        assert (n_layers > 1)

        # set which layers (aside from the first) should have the positional encoding passed in
        if self.pos_enc:
            self.pos_enc_layers = [4]
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
            nn.Linear(hidden_size, n_intersections)
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

    def forward(self, input, otherParameters):
        Input = input

        assert isinstance(input, list)
        B = len(Input)

        BIntersects = [None] * B
        BDepths = [None] * B
        CollateList = [None] * B
        for b in range(B):
            x = Input[b]
            BInput = Input[b]
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
            # intersections = self.layernorm(intersections)
            intersections = self.intersection_head[1](intersections)
            # intersections = torch.sigmoid(intersections)
            if len(intersections.size()) == 3:
                intersections = torch.squeeze(intersections, dim=1)

            # depth head
            depths = self.depth_head[0](x)
            depths = self.relu(depths)
            # depths = self.layernorm(depths)
            # enforce strictly increasing depth values
            depths = self.depth_head[1](depths)
            # depths = self.relu(depths) # todo: Avoid relu at the last layer?
            # depths = torch.cumsum(depths, dim=1)
            if len(depths.size()) == 3:
                depths = torch.squeeze(depths, dim=1)
            BIntersects[b] = intersections
            BDepths[b] = depths
            CollateList[b] = (intersections, depths)

        return CollateList
        # return BIntersects, BDepths

class LF4DSingleAutoDecoder(supernet.SuperNet):
    # class LF4D(nn.Module):
    '''
    A DDF with structure adapted from this LFN paper https://arxiv.org/pdf/2106.02634.pdf
    This is the autodecoder version
    '''

    def __init__(self, input_size=6, n_layers=6, hidden_size=256, radius=1.25, coord_type='direction', pos_enc=True, latent_size=256, Args=None):
        super().__init__(Args=Args)

        # store args
        n_intersections = 1 # Single intersection only
        self.preprocessing = coord_type
        self.pos_enc = pos_enc
        self.radius = radius
        self.LatentSize = latent_size
        self.InputSize = input_size + self.LatentSize
        assert (n_layers > 1)

        # set which layers (aside from the first) should have the positional encoding passed in
        if self.pos_enc:
            self.pos_enc_layers = [4]
        else:
            self.pos_enc_layers = []

        # Define the main network body
        main_layers = []
        main_layers.append(nn.Linear(self.InputSize, hidden_size))
        for l in range(n_layers - 1):
            if l + 2 in self.pos_enc_layers:
                main_layers.append(nn.Linear(hidden_size + self.InputSize, hidden_size))
            else:
                main_layers.append(nn.Linear(hidden_size, hidden_size))
        self.network = nn.ModuleList(main_layers)

        # Define the intersection head
        intersection_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, n_intersections)
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

    def forward(self, input, otherParameters):
        if(isinstance(input[0], tuple)):
            print(f"Input on cuda: {input[0][0].is_cuda}")
        else:
            print(f"Input on cuda: {input[0].is_cuda}")
        assert("Latent Vectors" in otherParameters)
        Input = input
        assert isinstance(input, list)
        B = len(Input)

        BIntersects = [None] * B
        BDepths = [None] * B
        CollateList = [None] * B
        LatentVectors = [None] * B
        for b in range(B):
            coords, indices = Input[b]
            x = torch.cat([coords, otherParameters["Latent Vectors"](indices)], dim=1)
            LatentVectors[b] = otherParameters["Latent Vectors"](indices)
            BInput = Input[b]
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
            # intersections = self.layernorm(intersections)
            intersections = self.intersection_head[1](intersections)
            # intersections = torch.sigmoid(intersections)
            if len(intersections.size()) == 3:
                intersections = torch.squeeze(intersections, dim=1)

            # depth head
            depths = self.depth_head[0](x)
            depths = self.relu(depths)
            # depths = self.layernorm(depths)
            # enforce strictly increasing depth values
            depths = self.depth_head[1](depths)
            depths = self.relu(depths) # todo: Avoid relu at the last layer?
            depths = torch.cumsum(depths, dim=1)
            if len(depths.size()) == 3:
                depths = torch.squeeze(depths, dim=1)
            BIntersects[b] = intersections
            BDepths[b] = depths
            CollateList[b] = (intersections, depths)
        return CollateList, LatentVectors
        # return BIntersects, BDepths
