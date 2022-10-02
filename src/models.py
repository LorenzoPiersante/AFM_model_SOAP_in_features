
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Conv3dBlock, Conv2dBlock, _get_padding
from layers import AttentionConvQuery2d, AttentionConvQuery3d
from layers import UNetAttentionConv
from layers import attention_mech
from utils import Atom, MoleculeGraph, find_gaussian_peaks
import SOAP_utils as Sutil

class GNN_env(nn.Module):
    '''
    This is a modified version of the GNN that does MP over the node features associated with the
    different chemical environments.

    1) calculates the messages between nodes within each chem environments
    2) aggregates message to node features using GRU

    Init args:
    - hidden_size: size of message in MPNN
    - iters: number of MP iterations
    - n_node_features: size of node hidden states (number of node features)
    - n_edge_features: size of edge hidden states (number of edge features)

    Forward args:
    - node_features: dim(num_env, tot_nodes, n_node_features), it is the output of the initialisation
    class node_feat_init(), they are the node features before MP
    - edges: tensor of dtype=torch.int of dim(2, num_edges), it contains the start and end point of
    each edge in the graph
    - device: device where tensor is stored

    Returns:
    - node_features: dim(num_env, tot_nodes, n_node_features), node features after message
    passing.
    '''

    def __init__(self, hidden_size=64, iters=3, n_node_features=20, n_edge_features=20):
        super().__init__()

        self.iters = iters
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features

        # message net
        self.msg_net = nn.Sequential(
            nn.Linear(2 * n_node_features, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, n_edge_features)
        )
        # aggergate node hidden state with message
        self.gru = nn.GRU(n_edge_features, n_node_features, batch_first=False)

    def forward(self, node_features, edges, device):
        
        n_env = node_features.size(0)
        n_nodes = node_features.size(1)

        if edges.size(0):
            # Symmetrise directional edge connections
            edges = torch.cat([edges, edges[[1, 0]]], dim=1)

        for i in range(self.iters):
            
            store_messg = torch.zeros(n_env, n_nodes, self.n_edge_features).to(device)
            if edges.size(0):  # No messages if no edges

                # Gather start and end nodes of edges
                src_features = torch.index_select(node_features, 1, edges[0])
                dst_features = torch.index_select(node_features, 1, edges[1])
                inputs = torch.cat([src_features, dst_features], dim=2)

                # Calculate messages for all edges and add them to both start and end nodes
                messages = self.msg_net(inputs)
                store_messg.index_add_(1, edges[0], messages)

            # separate the tensor into the individual chemical environments
            node_features = list(torch.split(node_features, 1, dim=0))
            store_messg = list(torch.split(store_messg, 1, dim=0))
            # update, using the messages, the node features for each chem environment
            for count, (n, a) in enumerate(zip(node_features, store_messg)):
                # Update node features
                new_n = self.gru(a, n)[1][0]
                node_features[count] = new_n
            # stack them back together removing extra dummy dim
            node_features = torch.squeeze(torch.stack(node_features), dim=1)

        return node_features

# 22-08-2022 Removed the normalisation layers
class env_attention_mech(nn.Module):
    '''
    This class defines the attention mechanism applied to the environmental node features of each node.
    The env node_features of each node undergo various rounds of self attention and they are brought to
    the final expanded size in the process.

    Init args:
    - node_features: input size of the environmental node features.
    - hidden_attn_size: intermediate size of attentioned features.
    - expanded_size: final expanded size.
    - device: device where the tensors are stored.

    Forward args:
    - node_features: output of MPNN, tensor of dim(tot_nodes, tot_env, node_features).

    Returns:
    - env_features: output of attention mechanism, tensor of dim(tot_nodes, tot_env, expanded_size).

    Where tot_nodes is the total number of nodes in the batch and tot_env is the total numbe of chem env
    in the system.
    '''

    def __init__(self, node_features, hidden_attn_size, expanded_size, device):
        super().__init__()

        self.attn1 = attention_mech(node_features, node_features, expanded_size, device)
        self.attn2 = attention_mech(expanded_size, expanded_size, expanded_size, device)

    def forward(self, node_features):
        # Attention mech over the chem env SOAP for each node
        attn_out = self.attn1(node_features, node_features)
        env_features = attn_out + self.attn2(attn_out, attn_out) #second layer with skip connection

        return env_features

class AttentionEncoderUNet(nn.Module):
    '''
    Pytorch Unet encoder-decoder network with attention gates.
    Arguments:
        conv3d_in_channels: int. Number of channels in input.
        conv3d_block_channels: list of ints. Number channels in 3D conv blocks.
        conv3d_block_depth: int. Number of layers in each 3D conv block.
        encoding_block_channels: list of ints. Number channels in encoding 3D conv blocks.
            Should same number of blocks as U-net 3D blocks.
        encoding_block_depth: int. Number of layers in each encoding 3D conv block.
        upscale_block_channels: list of int of length one smaller than conv3d_block_channels. Number of channels in
            each 3D conv block after upscale before skip connection.
        upscale_block_depth: int. Number of layers in each 3D conv block after upscale before skip connection.
        upscale_block_channels2: list of int of length one smaller than conv3d_block_channels. Number of channels in
            each 3D conv block after skip connection.
        upscale_block_depth2: int. Number of layers in each 3D conv block after skip connection.
        attention_channels: list of int of length one smaller than conv3d_block_channels. Number of channels in
            conv layers within each attention block.
        query_size: int. Number of features in the attention query vector.
        res_connections: Boolean. Whether to use residual connections in conv blocks.
        hidden_dense_units: list of ints. Number of units in hidden MLP layers.
        out_units: int. Number of units in final MLP layer.
        activation: str ('relu', 'lrelu', or 'elu') or nn.Module. Activation to use after every layer.
        padding_mode: str. Type of padding in each convolution layer. 'zeros', 'reflect', 'replicate' or 'circular'.
        pool_z_strides: list of int of same length as conv3d_block_channels. Stride of pool layers in z direction.
        decoder_z_sizes: list of int of length one smaller than conv3d_block_channels. Upscale sizes of decoder
            stages in the z dimension.
        attention_actication: str. Type of activation to use for attention map. 'sigmoid' or 'softmax'.
    References:
        https://doi.org/10.1016/j.media.2019.01.012
    '''
    def __init__(self, 
        conv3d_in_channels,
        conv3d_block_channels=[4, 8, 16, 32],
        conv3d_block_depth=2,
        encoding_block_channels=[4, 8, 16, 32],
        encoding_block_depth=2,
        upscale_block_channels=[32, 16, 8],
        upscale_block_depth=2,
        upscale_block_channels2=[32, 16, 8],
        upscale_block_depth2=2,
        attention_channels=[32, 32, 32],
        query_size=64,
        res_connections=True,
        hidden_dense_units=[],
        out_units=128,
        activation='relu',
        padding_mode='zeros',
        pool_type='avg',
        pool_z_strides=[2, 1, 2],
        decoder_z_sizes=[5, 10, 20],
        attention_activation='softmax'
    ):
        assert (
            len(encoding_block_channels)
            == len(conv3d_block_channels)
            == len(pool_z_strides) + 1
            == len(upscale_block_channels) + 1
            == len(upscale_block_channels2) + 1
        )

        super().__init__()

        self.out_units = out_units
        self.num_blocks = len(conv3d_block_channels)
        self.query_size = query_size
        self.decoder_z_sizes = decoder_z_sizes
        self.upsample_mode = 'trilinear'
        
        if isinstance(activation, nn.Module):
            self.act = activation
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU()
        elif activation == 'elu':
            self.act = nn.ELU()
        else:
            raise ValueError(f'Unknown activation function {activation}')
        
        # Encoder conv blocks
        conv3d_in_channels = [conv3d_in_channels] + conv3d_block_channels[:-1]
        self.conv3d_blocks = nn.ModuleList([
            Conv3dBlock(conv3d_in_channels[i], conv3d_block_channels[i], 3,
                conv3d_block_depth, padding_mode, res_connections, self.act)
            for i in range(self.num_blocks)
        ])
        self.encoding_conv_blocks = nn.ModuleList([
            Conv3dBlock(conv3d_block_channels[i], encoding_block_channels[i], 3, 
                encoding_block_depth, padding_mode, res_connections, self.act)
            for i in range(self.num_blocks)
        ])

        # Attention blocks
        self.encoding_attentions = nn.ModuleList([
            AttentionConvQuery3d(encoding_block_channels[i], encoding_block_channels[-1],
                query_size, 3, padding_mode, attention_activation)
            for i in range(self.num_blocks)
        ])
        self.unet_attentions = nn.ModuleList([
            UNetAttentionConv(conv3d_block_channels[-(i+2)], conv3d_block_channels[-1], attention_channels[i],
                3, padding_mode, self.act, attention_activation, upsample_mode='trilinear', ndim=3)
            for i in range(self.num_blocks - 1)
        ])

        # Decoder blocks
        upscale_block_channels2 = [conv3d_block_channels[-1]] + upscale_block_channels2
        self.upscale_blocks = nn.ModuleList([
            Conv3dBlock(upscale_block_channels2[i], upscale_block_channels[i], 3,
                    upscale_block_depth, padding_mode, res_connections, self.act, False)
            for i in range(self.num_blocks - 1)
        ])
        self.upscale_blocks2 = nn.ModuleList([
            Conv3dBlock(upscale_block_channels[i]+conv3d_block_channels[-(i+2)], upscale_block_channels2[i+1],
                3, upscale_block_depth2, padding_mode, res_connections, self.act, False)
            for i in range(self.num_blocks - 1)
        ])
        self.out_conv = nn.Conv3d(upscale_block_channels2[-1], 1, kernel_size=3,
            padding=_get_padding(3, 3), padding_mode=padding_mode)
        
        # Dense layers for image encoding output
        if hidden_dense_units:
            self.dense_layers = nn.ModuleList([nn.Linear(sum(encoding_block_channels), hidden_dense_units[0])])
            for i in range(len(hidden_dense_units)-1):
                self.dense_layers.append(nn.Linear(hidden_dense_units[i], hidden_dense_units[i+1]))
            self.dense_out = nn.Linear(hidden_dense_units[-1], out_units)
        else:
            self.dense_layers = []
            self.dense_out = nn.Linear(sum(encoding_block_channels), out_units)
        
        if pool_type == 'avg':
            pool = nn.AvgPool3d
        elif pool_type == 'max':
            pool = nn.MaxPool3d
        self.pools = nn.ModuleList([pool(2, stride=(2, 2, pz)) for pz in pool_z_strides])
    
    def _flatten(self, x):
        return x.permute(0,1,4,2,3).reshape(x.size(0), -1, x.size(2), x.size(3))

    def predict_grid(self, x):

        x_3ds = []
        x_encs = []
        for i in range(self.num_blocks):

            conv3d = self.conv3d_blocks[i]
            conv_enc = self.encoding_conv_blocks[i]

            # Apply 3D conv block
            x = self.act(conv3d(x))

            # Apply encoding 3D conv block for later use
            x_enc = self.act(conv_enc(x))

            # Store feature maps for attention gating later
            x_3ds.append(x)
            x_encs.append(x_enc)

            # Down-sample for next iteration of convolutions
            if i < self.num_blocks-1:
                x = self.pools[i](x)

        x_3ds, x = x_3ds[:-1], x_3ds[-1]
        x_3ds.reverse()

        # Compute skip-connection attention maps
        unet_attention_maps = []
        x_gated = []
        for attention, x_3d in zip(self.unet_attentions, x_3ds):
            xg, a = attention(x_3d, x)
            x_gated.append(xg)
            unet_attention_maps.append(a)

        # Decode
        for i, (conv1, conv2, xg) in enumerate(zip(self.upscale_blocks, self.upscale_blocks2, x_gated)):

            # Upsample and apply first conv block
            target_size = x_3ds[i].shape[2:4] + (self.decoder_z_sizes[i],)
            x = F.interpolate(x, size=target_size, mode=self.upsample_mode, align_corners=False)
            xg = F.interpolate(xg, size=target_size, mode=self.upsample_mode, align_corners=False)
            x = self.act(conv1(x))

            # Concatenate attention-gated skip connections and apply second conv block
            x = torch.cat([x, xg], dim=1)
            x = self.act(conv2(x))

        # Get grid output
        x_grid = self.out_conv(x).squeeze(1)

        return x_grid, x_encs, unet_attention_maps

    def encode_features(self, x_encs, q):

        # Get attention-gated features from encoder
        g = x_encs[-1]
        encoding_attention_maps = []
        attention_features = []
        for x, attention in zip(x_encs, self.encoding_attentions):
            x_attn, attn = attention(x, q, g)
            attention_features.append(x_attn)
            encoding_attention_maps.append(attn)

        # Apply MLP to attention-gated features to get image encoding
        x = torch.cat(attention_features, dim=1)
        for fc in self.dense_layers:
            x = self.act(fc(x))
        img_encoding = self.dense_out(x)

        return img_encoding, encoding_attention_maps
        
    def forward(self, x, q, return_attention=False):
        
        # Encode input and predict grid output
        x_grid, x_encs, unet_attention_maps = self.predict_grid(x)
        
        # Further encode into feature vector
        img_encoding, encoding_attention_maps = self.encode_features(x_encs, q)

        out = x_grid, img_encodingFUNCTION
        if return_attention:
            out = out + (unet_attention_maps, encoding_attention_maps)

        return out

class GridGraphImgNet(nn.Module):
    '''
    Grid-based GNN model for predicting a graph node from an AFM image.

    Arguments:
        cnn: AttentionEncoderUNet. Image encoder network.
        gnn_class: MPNN over chem environments for classification task.
        gnn_edge: MPNN over chemical environments for edge prediction task.
        n_classes: int. Number of node_classes.
        expansion_hidden: int. Number of units in hidden layer of expansion network.
        expanded_size: int. Final size of expanded node features.
        query_hidden: int. Number of units in hidden layer of attention query network.
        class_hidden: int. Number of units in hidden layer of class prediction network.
        edge_hidden: int. Number of units in hidden layer of edge prediction network.
        peak_std: float. Standard deviation of atom position distribution peaks in angstroms.
        match_threshold: float. Detection threshold for matching when finding atom position peaks.
        match_method: str. Method for template matching when finding atom position peaks.
        See graph_utils.find_gaussian_peaks for options.
        dist_threshold: float. Minimum distance of new atoms to existing atoms. If 0, skip
        checking distances.
        teacher_forcing_rate: float in [0, 1]. Probability for using teacher forcing in training.
        device: str. Device to store model on.
    '''
    def __init__(self,
        cnn,
        gnn_class,
        gnn_edge,
        n_classes,
        SOAP_size = Sutil.SOAP_size,
        hidden_attn_size=64,
        expanded_size=128,
        query_hidden=128,
        class_hidden=32,
        edge_hidden=32,
        peak_std=0.3,
        match_threshold=0.7,
        match_method='msd_norm',
        dist_threshold=0.5,
        teacher_forcing_rate = 0.5,
        device='cuda'
    ):
        super().__init__()

        assert dist_threshold >= 0, 'dist_threshold has to be non-negative.'
        assert 0.0 <= teacher_forcing_rate <= 1.0, 'teacher_forcing rate has to be between 0.0 and 1.0'
        
        self.cnn = cnn
        self.gnn_class = gnn_class
        self.gnn_edge = gnn_edge

        self.n_classes = n_classes
        self.SOAP_size = SOAP_size
        self.expanded_size = expanded_size
        self.peak_std = peak_std
        self.match_threshold = match_threshold
        self.match_method = match_method
        self.dist_threshold = dist_threshold
        self.teacher_forcing_rate = teacher_forcing_rate

        #node initialisation net - node classification
        self.encode_node_inputs = nn.Sequential(nn.Linear(3+n_classes+SOAP_size, self.gnn_class.n_node_features),
                                                nn.ReLU(),
                                                nn.Linear(self.gnn_class.n_node_features, self.gnn_class.n_node_features))

        #attention mechanism for environments expansion - node classification
        self.env_attention = env_attention_mech(self.gnn_class.n_node_features, hidden_attn_size, expanded_size, device)

        #gate net for environment aggregation in a single node hidden state - node classification
        self.env_gate = nn.Sequential(nn.Linear(expanded_size, expanded_size), nn.ReLU(),
                                      nn.Linear(expanded_size, expanded_size))
        
        #node initialisation net - node classification
        self.encode_node_inputs2 = nn.Sequential(nn.Linear(3+n_classes+SOAP_size, self.gnn_edge.n_node_features),
                                                nn.ReLU(),
                                                nn.Linear(self.gnn_edge.n_node_features, self.gnn_edge.n_node_features))

        #attention mechanism for environments expansion - edge prediction
        self.env_attention2 = env_attention_mech(self.gnn_edge.n_node_features, hidden_attn_size, expanded_size, device)

        #gate net for environment aggregation in a single node hidden state - edge prediction
        self.env_gate2 = nn.Sequential(nn.Linear(expanded_size, expanded_size), nn.ReLU(),
                                      nn.Linear(expanded_size, expanded_size))
        
        #gate net for environment aggregation in a single node hidden state, for new nodes - edge prediction
        self.new_env_gate = nn.Sequential(nn.Linear(expanded_size, expanded_size), nn.ReLU(),
                                          nn.Linear(expanded_size, expanded_size))

        #gate net for node aggregation in a single graph encoding
        self.node_gate = nn.Sequential(nn.Linear(expanded_size, expanded_size), nn.ReLU(),
                                       nn.Linear(expanded_size, expanded_size))

        self.query_net = nn.Sequential(
            nn.Linear(expanded_size+3, query_hidden), nn.ReLU(),
            nn.Linear(query_hidden, self.cnn.query_size)
        )
        self.class_net = nn.Sequential(
            nn.Linear(expanded_size+self.cnn.out_units, class_hidden), nn.ReLU(),
            nn.Linear(class_hidden, self.n_classes)
        )
        #adapt the size of the edge net to the expanded size of the node features after
        #environment attention
        self.edge_net = nn.Sequential(
            nn.Linear(2*expanded_size+self.cnn.out_units, edge_hidden), nn.ReLU(),
            nn.Linear(edge_hidden, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        self.device = device
        self.to(device)

    #new graph encoding function
    def encode_graph(self, node_inputs, edges, Ns):
        '''
        Args:
            node_inputs: dim(num_env, tot_nodes, SOAP+pos+one-hot), new_nodes_input output of the function
    add_SOAP2node_input_Genc(), they are the node features before MP.
            edges: tensor of dtype=torch.int of dim(2, num_edges), it contains the start and end point of
    each edge in the graph.
            Ns: list of integers s.t. sum(Ns)=tot_nodes that indicates how to split the batched input into
    the respective subgraphs.

        Returns:
            graph_encoding: tensor of dim(len(Ns), expanded_size), each row of this tensor is the aggregated
    graph encoding for a subgraph in the batch.
        '''

        if node_inputs.size(1) > 0:

            # GNN
            # initialise hidden state vectors
            node_features = self.encode_node_inputs(node_inputs)
            # MPNN
            node_features = self.gnn_class(node_features, edges, self.device)  # output is dim(env, N, 20)

            # FEATURE AGGREGATION
            # reshape input
            node_features = torch.transpose(node_features, 0, 1)  # dim(N, env, 20)
            # apply attention mechanism to chem env
            env_features = self.env_attention(node_features)

            # aggregate featues over chem environments
            env_gate = self.sigmoid(self.env_gate(env_features))  # calculate env gate
            gated_env_feat = env_features * env_gate  # apply gate
            node_env_feat = torch.sum(gated_env_feat, dim=1)  # sum over env

            # aggregate features over subgraphs
            node_gate = self.sigmoid(self.node_gate(node_env_feat))  # calculate node gate
            gated_node_env_feat = node_gate * node_env_feat  # apply gate
            # split features by subgraphs
            gated_node_env_feat = torch.split(gated_node_env_feat, split_size_or_sections=Ns)
            # create graph encodings for each subgraph
            graph_encoding = []
            for feat in gated_node_env_feat:
                graph_encoding.append(torch.sum(feat, dim=0))  # sum is 0 for empty subgraph
            # stack encodings
            graph_encoding = torch.stack(graph_encoding, dim=0)

        else:
            # Zero encoding for empty graph
            graph_encoding = torch.zeros((len(Ns), self.expanded_size)).to(self.device)

        return graph_encoding

    def encode_afm(self, x_encs, graph_encoding, new_node_pos):

        # Get attention query vector from graph encoding and new atom position
        q = self.query_net(torch.cat([graph_encoding, new_node_pos], dim=1))

        # Encode AFM image
        img_encoding, encoding_attention_maps = self.cnn.encode_features(x_encs, q)

        return img_encoding, encoding_attention_maps

    def predict_class(self, encoding):
        classes = self.class_net(encoding)
        return classes

    def input_predict_edges(self, node_inputs2, new_nodes2, edges, Ns):
        '''
        Args:
            node_inputs2: dim(num_env, tot_nodes, input_size), 0th output of generate_Eenc_input() function,
    dim=-1 is SOAP+pos+one-hot.
            new_nodes2: dim(num_env, len(Ns), input_size), 1st output of generate_Eenc_input() function,
    dim=-1 is SOAP+pos+one-hot.
    Where SOAP is updated to account for the newly classified node.
            edges: tensor of dtype=torch.int of dim(2, num_edges), it contains the start and end point of
    each edge in the graph.
            Ns: list of integers s.t. sum(Ns)=tot_nodes that indicates how to split the batched input into
    the respective subgraphs.

        Returns:
            node_env_feat: dim(tot_nodes, expanded_size).
            new_node_env_feat: dim(len(Ns), expanded_size).
        They will be used as input for the subsequent parts of the predict_edges() function in the complete
    model.
        '''

        # encode the SOAP+pos+one-hot input
        node_features = self.encode_node_inputs2(node_inputs2)
        new_nodes_features = self.encode_node_inputs2(new_nodes2)

        # MPNN for previously classified nodes x 2
        node_features = self.gnn_edge(node_features, edges, self.device)

        # reshape input
        node_features = torch.transpose(node_features, 0, 1)
        new_nodes_features = torch.transpose(new_nodes_features, 0, 1)

        # attention mechanism
        env_features = self.env_attention2(node_features)
        new_env_features = self.env_attention2(new_nodes_features)

        # env aggregation for:
        # old nodes
        env_gate = self.sigmoid(self.env_gate2(env_features))
        gated_env_feat = env_features * env_gate
        node_env_feat = torch.sum(gated_env_feat, dim=1)
        # new nodes
        new_env_gate = self.sigmoid(self.env_gate(new_env_features))
        gated_new_env_feat = new_env_features * new_env_gate
        new_node_env_feat = torch.sum(gated_new_env_feat, dim=1)

        return node_env_feat, new_node_env_feat

    #This function will take node_feature and new_node_features inputs from
    #the input_predict_edges() function
    #node_features=node_env_feat, new_node_features=new_node_env_feat
    def predict_edges(self, node_features, new_node_features, edges, img_encoding, Ns):
        #new_node_features: tensor of initialised encodings for the newly classified nodes
        
        # Propagate again and concatenate new node features to node features of each old node
        split_features = torch.split(node_features, split_size_or_sections=Ns)
        combined_features = []
        for i, (feat, N) in enumerate(zip(split_features, Ns)):
            new_feat = new_node_features[i].repeat(N,1)
            img_feat = img_encoding[i].repeat(N,1)
            combined_features.append(torch.cat([feat, new_feat, img_feat], dim=1))
        combined_features = torch.cat(combined_features, dim=0)
        
        # Predict edge connections to new node
        new_edges = self.edge_net(combined_features).squeeze(1)
        new_edges = self.sigmoid(new_edges)
        new_edges = list(torch.split(new_edges, split_size_or_sections=Ns))

        return new_edges

    def find_new_atoms(self, pos_dist, node_inputs, Ns, ref_atoms, box_borders):

        assert len(pos_dist) == len(Ns)

        with torch.no_grad():

            # Find positions of peaks in predicted distribution.
            peak_pos, _, _ = find_gaussian_peaks(pos_dist, box_borders, match_threshold=self.match_threshold,
                std=self.peak_std, method=self.match_method)

            # Loop over batch items
            N_start = 0
            pos_new = []
            for p, r, N in zip(peak_pos, ref_atoms, Ns):

                if len(p) > 0 and N > 0 and self.dist_threshold > 0:

                    # Pick existing atoms corresponding to batch item
                    n = node_inputs[N_start:N_start+N, :3]

                    # Compute all pairwise distances
                    d = torch.cdist(p, n)
                    
                    # Pick out those new atoms which are not too close to existing ones
                    p = p[torch.all(d > self.dist_threshold, dim=1)]

                if len(p) > 0:
                    pos_new.append(p)
                else:
                    # No peaks found => pick reference atom position
                    pos_new.append(r[None, :3])

                N_start += N

        return pos_new

    def forward(self, X, node_inputs, edges, Ns, ref_atoms, box_borders,
        return_attention=False, X_is_encoded=False):

        if return_attention and X_is_encoded:
            raise ValueError('Cannot return attention maps when X_is_encoded==True')

        # Predict position grid
        if not X_is_encoded:
            pos_dist, x_encs, unet_attention_maps = self.cnn.predict_grid(X)
        else:
            x_encs = X
            pos_dist = []
        
        # Get atom positions
        teacher_forcing = random.random() <= self.teacher_forcing_rate
        if teacher_forcing or X_is_encoded:
            pos = ref_atoms[:, :3]
        else:
            new_atoms = self.find_new_atoms(pos_dist, node_inputs, Ns, ref_atoms, box_borders)
            pos = torch.stack([random.choice(a) for a in new_atoms], dim=0)

        # Add SOAP vector to node inputs
        SOAP_nodes_input, class_chunks, position_chunks = Sutil.add_SOAP2node_input_Genc(node_inputs, Ns,
                                                                                         self.SOAP_size, self.device)

        # Encode input graph
        graph_encoding = self.encode_graph(SOAP_nodes_input, edges, Ns)

        # Get AFM image encoding
        img_encoding, encoding_attention_maps = self.encode_afm(x_encs, graph_encoding, pos)
        
        # Combine encodings
        encoding = torch.cat([img_encoding, graph_encoding], dim=1)

        # Predict classes using encoding and atom positions
        classes = self.predict_class(encoding)
        nodes = torch.cat([pos, classes], dim=-1).squeeze(1)

        # Get encoded features of the new node - move inside if statement so that the inputs for the edge predictor
        # are evaluated only if other nodes are present in the graph
        new_nodes = torch.cat([pos, self.softmax(classes)], dim=-1)

        # Predict edge connections to new node
        if len(node_inputs) > 0:
            # Generate updated node features for input nodes and new nodes
            SOAP_nodes_input, SOAP_new_nodes = Sutil.generate_Eenc_input(class_chunks, position_chunks,
                                                                         node_inputs, new_nodes, self.device)
            # Produce node features for all nodes
            node_features, new_node_features =  self.input_predict_edges(SOAP_nodes_input, SOAP_new_nodes,
                                                                    edges, Ns)
            new_edges = self.predict_edges(node_features, new_node_features, edges, img_encoding, Ns)
        else:
            # No edges for empty graph
            new_edges = [torch.empty(0).to(self.device) for _ in range(len(Ns))]

        out = nodes, new_edges, pos_dist
        if return_attention:
            out = out + (unet_attention_maps, encoding_attention_maps)

        return out

    def predict_sequence(self, X, box_borders, bonds_threshold=0.5, max_length=50, order=None):
        '''
        Predict full molecule graph in a sequence of steps of single atom predictions based on an input AFM image.
        Arguments:
            X: torch.Tensor of shape (batch_size, 1, x, y, z). Input AFM images.
            box_borders: tuple ((x_start, y_start, z_start),(x_end, y_end, z_end)). Real-space extent of the
                output distribution region.
            bonds_threshold: float in [0, 1]. Threshold probability for bond connections. Predicted probability above
                bonds_threshold results in a bond being added to the graph.
            max_length: int. Maximum number of steps in sequence.
            order: 'x', 'y', 'z' or None. Construct graphs in order of decreasing x, y, or z coordinate
                of atoms, or if None, do construction in random order.
        Returns:
            pred_graphs: list of MoleculeGraph. Predicted molecule graphs.
            pred_dist: torch.Tensor of shape (batch_size, x, y, z_grid). Predicted position distributions
            pred_sequence: list of lists of tuples (mol, atom, bonds). Each item in the outer list corresponds to 
                one batch item and each item in the inner list corresponds to one prediction step. In each step,
                mol is a MoleculeGraph that was the input for the step, atom is an Atom object that is the predicted
                atom for the step, bonds is a list of floats that indicates the bond connection weights from atom to mol
                for the step.
            completed: list of Bool. List indicating for each batch item whether the sequence completed or was terminated
                due to exceeding max_length.
        '''

        with torch.no_grad():

            # Initialize
            batch_size = len(X)
            X = X.to(self.device)
            pred_graphs = [MoleculeGraph([], []) for _ in range(batch_size)]
            pred_sequence = [[] for _ in range(batch_size)]
            mols = [MoleculeGraph([], []) for _ in range(batch_size)]

            # Do distribution prediction and AFM encoding once and reuse it on each iteration
            pred_dist, x_encs, _ = self.cnn.predict_grid(X)

            # Get atom positions from the predicted distribution.
            atom_pos, _, _ = find_gaussian_peaks(pred_dist, box_borders, match_threshold=self.match_threshold,
                std=self.peak_std, method=self.match_method)

            if order in ['x', 'y', 'z']:
                ind = 'xyz'.index(order)
                atom_pos = [a[torch.argsort(a[:, ind], descending=True)] for a in atom_pos]
            elif order is not None:
                raise ValueError(f'Unknown construction order `{order}`.')

            if any([len(a) > max_length for a in atom_pos]):
                print('Warning: Sequence has more atoms than the specified max length. Some atoms will be cut'
                    ' from the sequence.')

            # Keep track of which batch items are complete
            not_complete = [len(a) > 0 for a in atom_pos]
            
            # Check if some batch items don't have any atoms in prediction
            del_inds = [i for i in range(len(not_complete)) if not not_complete[i]]
            if del_inds:
                keep_inds = torch.ones(len(mols), dtype=torch.bool)
                keep_inds[del_inds] = False
                for del_ind in reversed(del_inds):
                    del mols[del_ind]
                    del atom_pos[del_ind]
                x_encs = [x[keep_inds] for x in x_encs]

            step = 0
            while any(not_complete) and step < max_length:
                
                
                # Combine input graphs into one for GNN and transfer to device
                node_inputs, edges, Ns = _combine_graphs(mols)
                node_inputs = node_inputs.to(self.device)
                edges = edges.to(self.device)

                # Pick atom positions to add to the graph and delete from original list afterwards
                new_pos = torch.stack([a[0] for a in atom_pos], dim=0)
                atom_pos = [a[1:] for a in atom_pos]

                # Forward
                new_nodes, new_edges, _ = self(x_encs, node_inputs, edges, Ns, new_pos, box_borders,
                    return_attention=False, X_is_encoded=True)
                new_nodes[:,3:] = self.softmax(new_nodes[:,3:])

                # Back to host
                new_nodes = new_nodes.cpu().numpy()
                new_edges = [e.cpu().numpy() for e in new_edges]

                # Indices of batch items that are not complete
                inds = [i for i in range(batch_size) if not_complete[i]]

                del_inds = []
                for i, (new_node, new_bonds) in enumerate(zip(new_nodes, new_edges)):

                    ind = inds[i]

                    new_atom = Atom(new_node[:3], class_weights=new_node[3:])
                    new_bonds_thresholded = (new_bonds > bonds_threshold) * 1
                    pred_sequence[ind].append((mols[i], new_atom, new_bonds))

                    new_atom_one_hot = Atom(new_node[:3], class_weights=np.eye(self.n_classes)[new_atom.class_index])
                    pred_graphs[ind] = pred_graphs[ind].add_atom(new_atom, new_bonds_thresholded)
                    mols[i] = mols[i].add_atom(new_atom_one_hot, new_bonds_thresholded)

                    if len(atom_pos[i]) == 0: # This was the last atom in the list
                        not_complete[ind] = False
                        del_inds.append(i)

                # Delete batch items which are complete
                keep_inds = torch.ones(len(mols), dtype=torch.bool)
                keep_inds[del_inds] = False
                for del_ind in reversed(del_inds):
                    del mols[del_ind]
                    del atom_pos[del_ind]
                x_encs = [x[keep_inds] for x in x_encs]

                step += 1

            completed = [not b for b in not_complete]

        return pred_graphs, pred_dist, pred_sequence, completed

def _combine_graphs(mols):

    mol_arrays = []
    edges = []
    Ns = []
    ind_count = 0

    for i, mol in enumerate(mols):

        if (mol_array := mol.array(xyz=True, class_weights=True)) != []:
            mol_arrays.append(mol_array)
        edges += [[b[0]+ind_count, b[1]+ind_count] for b in mol.bonds]

        ind_count += len(mol)
        Ns.append(len(mol))

    if len(mol_arrays) > 0:
        node_inputs = torch.from_numpy(np.concatenate(mol_arrays, axis=0)).float()
    else:
        node_inputs = torch.empty((0))
    edges = torch.tensor(edges).long().T

    return node_inputs, edges, Ns

class GridMultiAtomLoss(nn.Module):
    '''
    Grid MSE + classification loss for molecule graph prediction.
    
    Arguments:
        pos_factor: float. Multiplicative constant for position MSE.
        class_factor: float. Multiplicative constant for node class NLL.
        edge_factor: float. Multiplicative constant for edge NLL.
    '''
    def __init__(self, pos_factor=1.0, class_factor=1.0, edge_factor=1.0):
        super().__init__()
        self.pos_factor = pos_factor
        self.class_factor = class_factor
        self.edge_factor = edge_factor
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.bce = nn.BCELoss(reduction='mean')
        
    def forward(self, pred, ref, separate_batch_items=False, separate_loss_factors=False,
        return_scaled=False):
        
        pred_nodes, pred_edges, pred_dist = pred
        ref_nodes, ref_edges, ref_terminate, ref_dist = ref

        # Calculate position distribution MSE and class NLL (= Cat. cross entropy) losses
        mse_pos = torch.mean((pred_dist - ref_dist) ** 2, dim=(1,2,3))
        
        nll_class = []
        nll_edge = []
        min_inds = []

        # Loop over batch items for class and edge losses
        for pred_i, ref_i, pred_edge, ref_edge, term in zip(
            pred_nodes, ref_nodes, pred_edges, ref_edges, ref_terminate):

            pos_p = pred_i[None, :3]; cls_p = pred_i[None, 3:]
            pos_r = ref_i[:, :3]; cls_r = ref_i[:, 3].long()

            # Find the closest reference atom
            min_ind = torch.argmin(( (pos_r - pos_p)**2 ).sum(dim=-1), dim=0)

            # Class NLL (= Cat. cross entropy) and edge NLL (= binary cross entropy)
            nll_class_ = (term == 0) * self.cross_entropy(cls_p, cls_r[None, min_ind]).mean()
            nll_edge_ = (term == 0) * self.bce(pred_edge[None, :], ref_edge[min_ind, None, :])

            nll_class.append(nll_class_)
            nll_edge.append(nll_edge_)
            min_inds.append(min_ind)
        
        # Gather losses
        nll_class = torch.stack(nll_class)
        nll_edge = torch.stack(nll_edge)
        nll_edge[torch.isnan(nll_edge)] = 0.0 # Empty molecule -> no edges -> set manually to zero

        if not separate_batch_items:
            mse_pos = mse_pos.mean()
            nll_class = nll_class.mean()
            nll_edge = nll_edge.mean()
        
        mse_pos_scaled = self.pos_factor * mse_pos
        nll_class_scaled = self.class_factor * nll_class
        nll_edge_scaled = self.edge_factor * nll_edge
        
        loss = mse_pos_scaled + nll_class_scaled + nll_edge_scaled
        if separate_loss_factors:
            if return_scaled:
                loss = [loss, mse_pos_scaled, nll_class_scaled, nll_edge_scaled]
            else:
                loss = [loss, mse_pos, nll_class, nll_edge]

        min_inds = torch.stack(min_inds, dim=0)
        
        return loss, min_inds

def load_pretrained_model(weights_type='random', device='cpu'):
    '''
    Load GridGraphImgNet model with pretrained weights.

    Arguments:
        weights_type: 'random', 'y', or 'z'. Type of weights to load. Different types correspond to 
            different graph constructions orders used during training.

    Returns: GridGraphImgNet.
    '''

    weights_dir = os.path.abspath(os.path.split(__file__)[0] + '/../pretrained_weights')
    if weights_type == 'random':
        weights_path = os.path.join(weights_dir, 'model_random.pth')
    elif weights_type == 'y':
        weights_path = os.path.join(weights_dir, 'model_y.pth')
    elif weights_type == 'z':
        weights_path = os.path.join(weights_dir, 'model_z.pth')
    else:
        raise ValueError(f'Unrecognized weights type `{weights_type}`.')

    gnn = GNN_env(
        hidden_size     = 64,
        iters           = 3,
        n_node_features = 20,
        n_edge_features = 20
    )
    cnn = AttentionEncoderUNet(
        conv3d_in_channels      = 1,
        conv3d_block_channels   = [4, 8, 16, 32],
        conv3d_block_depth      = 2,
        encoding_block_channels = [4, 8, 16, 32],
        encoding_block_depth    = 2,
        upscale_block_channels  = [32, 16, 8],
        upscale_block_depth     = 2,
        upscale_block_channels2 = [32, 16, 8],
        upscale_block_depth2    = 2,
        attention_channels      = [32, 32, 32],
        query_size              = 64,
        res_connections         = True,
        hidden_dense_units      = [],
        out_units               = 128,
        activation              = 'relu',
        padding_mode            = 'zeros',
        pool_type               = 'avg',
        pool_z_strides          = [2, 1, 2],
        decoder_z_sizes         = [4, 10, 20],
        attention_activation    = 'softmax'
    )
    model = GridGraphImgNet(
        cnn                  = cnn,
        gnn                  = gnn,
        n_classes            = 5,
        expansion_hidden     = 32,
        expanded_size        = 128,
        query_hidden         = 64,
        class_hidden         = 32,
        edge_hidden          = 32,
        peak_std             = 0.25,
        match_method         = 'msd_norm',
        match_threshold      = 0.7,
        dist_threshold       = 0.5,
        teacher_forcing_rate = 1.0,
        device               = device
    )

    state = torch.load(weights_path)
    model.load_state_dict(state['model_params'])

    return model
