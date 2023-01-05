import torch
import torch.nn as nn
from .Modeling import AttLayer, Intermediate, Output 

import math
from copy import deepcopy

LayerNorm = torch.nn.LayerNorm

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

class ConvLayer(nn.Module):
    def __init__(self, config, dim_in, dim_out, kernel_size, padding, dilation, stride = 1, dropout = None):
        super(ConvLayer, self).__init__()
        self.config = config
        self.conv = nn.Conv1d(in_channels = dim_in, out_channels = dim_out, kernel_size = kernel_size,
                            padding = padding, dilation = dilation, stride = stride)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        self.act = nn.ReLU()

    def forward(self, prots):

        if len(prots.size()) == 2:
            prots = prots.unsqueeze(1)
        
        x = self.conv(prots)
        x = self.act(x)

        return x 
"""
class PocketConvModule(nn.Module):
    def __init__(self, config):
        super(PocketConvModule, self).__init__()
        
        self.config = config
        self.first_layers = nn.ModuleList()
        
        self.first_layers.append(ConvLayer(self.config, dim_in = 1024, dim_out = 256, 
            kernel_size = 3, padding = 1, dilation = 1, stride = 1))
            
    def forward(self, embeddings = None, prots = None, attention_mask = None):
        first_feats = prots.transpose(1,2)
        
        for layer_module in self.first_layers:
            first_feats = layer_module(first_feats)

        feats = first_feats
        
        return feats.transpose(1,2)

"""
class PocketConvModule(nn.Module):
    def __init__(self, config):
        super(PocketConvModule, self).__init__()
        
        self.config = config

        dim_in_tuple = (1024, self.config["architectures"]["hidden_size"], self.config["architectures"]["hidden_size"], 
                            self.config["architectures"]["hidden_size"], self.config["architectures"]["hidden_size"])
        dim_out_tuple = (self.config["architectures"]["hidden_size"], self.config["architectures"]["hidden_size"], 
                            self.config["architectures"]["hidden_size"], self.config["architectures"]["hidden_size"], 
                                self.config["architectures"]["hidden_size"])
       
        dilation_tuple = (1, 2, 3)
        
        self.first_ = nn.ModuleList()
        self.second_ = nn.ModuleList()
        self.third_ = nn.ModuleList()

        for idx, dilation_rate in enumerate(dilation_tuple):
            self.first_.append(ConvLayer(self.config, dim_in = dim_in_tuple[idx], dim_out = dim_out_tuple[idx], 
                kernel_size = 3, padding = dilation_rate, dilation = dilation_rate))

        for idx, dilation_rate in enumerate(dilation_tuple):
            self.second_.append(ConvLayer(self.config, dim_in = dim_in_tuple[idx], dim_out = dim_out_tuple[idx], 
                kernel_size = 5, padding = 2 * dilation_rate, dilation = dilation_rate))

        for idx, dilation_rate in enumerate(dilation_tuple):
            self.third_.append(ConvLayer(self.config, dim_in = dim_in_tuple[idx], dim_out = dim_out_tuple[idx], 
                kernel_size = 7, padding = 3 * dilation_rate , dilation = dilation_rate)) 
                
                
        """
        for idx in range(3):
            self.first_.append(ConvLayer(self.config, dim_in = dim_in_tuple[idx], dim_out = dim_out_tuple[idx], 
                kernel_size = 3, padding = 1, dilation = 1))

        for idx in range(3):
            self.second_.append(ConvLayer(self.config, dim_in = dim_in_tuple[idx], dim_out = dim_out_tuple[idx], 
                kernel_size = 3, padding = 2, dilation = 2))

        for idx in range(3):
            self.third_.append(ConvLayer(self.config, dim_in = dim_in_tuple[idx], dim_out = dim_out_tuple[idx], 
                kernel_size = 3, padding = 4, dilation = 4)) 
        """
        
        
    def forward(self, query_embeddings, aa_embeddings, attention_mask):    
        first_aa_embeddings, second_aa_embeddings, third_aa_embeddings = aa_embeddings.transpose(1,2), aa_embeddings.transpose(1,2), aa_embeddings.transpose(1,2)

        for layer_module in self.first_:
            first_aa_embeddings = layer_module(first_aa_embeddings)

        for layer_module in self.second_:
            second_aa_embeddings = layer_module(second_aa_embeddings)

        for layer_module in self.third_:
            third_aa_embeddings = layer_module(third_aa_embeddings) 

        aa_embeddings = first_aa_embeddings + second_aa_embeddings + third_aa_embeddings
        
        return aa_embeddings.transpose(1, 2)
        
class PocketATTModule(nn.Module):
    def __init__(self, config):
        super(PocketATTModule, self).__init__()
        
        self.config = config
                
        # cross-attention layer
        self.attention = AttLayer(self.config)
        
        self.inter = Intermediate(self.config)
        self.output = Output(self.config)       
    
    def forward(self, embeddings, prots, attention_mask):

        prots_cross_output = self.attention(embeddings, prots, attention_mask)

        prots_inter_outputs = self.inter(prots)
        prots_outputs = self.output(prots_inter_outputs, prots_cross_output)
        
        return prots_outputs

class Pseq2Sites(nn.Module):
    def __init__(self, config):
        super(Pseq2Sites, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(self.config["train"]["dropout"])

        # define embeddings
        self.aa_position_embeddings = nn.Embedding(config["prots"]["max_lengths"], 1024) # (Max_position, Hidden_size)
        self.aa_chain_embeddings = nn.Embedding(config["prots"]["max_chains"], 1024, padding_idx=0)
        self.aa_LayerNorm = LayerNorm(1024, eps=1e-12)

        self.protein_position_embeddings = nn.Embedding(config["prots"]["max_lengths"], 256)
        self.protein_chain_embeddings = nn.Embedding(config["prots"]["max_chains"], 256, padding_idx=0)
        self.protein_layer = nn.Linear(1024, self.config["architectures"]["hidden_size"])
        self.protein_LayerNorm = LayerNorm(1024, eps=1e-12)
        
        self.query_LayerNorm = LayerNorm(256, eps=1e-12)
        
        self.cnn_attenion_modules = nn.ModuleList()
        self.cnn_attenion_modules.append(PocketConvModule(self.config))
        self.cnn_attenion_modules.append(PocketATTModule(self.config))

        self.fc = nn.Sequential(
                        GeLU(),
                        LayerNorm(512, eps=1e-12),
                        self.dropout,
                        nn.Linear(512, 128),
                        GeLU(),
                        LayerNorm(128, eps=1e-12),
                        self.dropout,
                        nn.Linear(128, 64),
                        GeLU(),
                        LayerNorm(64, eps=1e-12),
                        self.dropout,
                        nn.Linear(64, 1)  
                    )         

    def forward(self, aa_feats, protein_feats, attention_mask, position_ids, chain_ids):
        
        aa_position_embeddings = self.aa_position_embeddings(position_ids) # (B, L, H)
        aa_chain_embeddings = self.aa_chain_embeddings(chain_ids) # (B, L, H)
        
        protein_position_embeddings = self.protein_position_embeddings(position_ids)
        protein_chain_embeddings = self.protein_chain_embeddings(chain_ids)
        
        # residue embeddings
        residue_embeddings = aa_feats + aa_position_embeddings + aa_chain_embeddings
        residue_embeddings = self.aa_LayerNorm(residue_embeddings) # (B, L, H)
        residue_embeddings = self.dropout(residue_embeddings) # (B, L, H)

        # protein embeddings
        protein_embeddings = self.protein_LayerNorm(protein_feats) # (B, L, H)
        protein_embeddings = self.protein_layer(protein_embeddings) # (B, L, H')
        protein_embeddings = self.dropout(protein_embeddings) # (B, L, H')

        # query embeddings
        query_embeddings = protein_embeddings + protein_position_embeddings + protein_chain_embeddings # (B, L, H')
        query_embeddings = self.query_LayerNorm(query_embeddings) # (B, L, H')
        query_embeddings = self.dropout(query_embeddings) # (B, L, H')
        
        prots_extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (Batch, 1, 1, Seq_length)
        prots_extended_attention_mask = prots_extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        prots_extended_attention_mask = (1.0 - prots_extended_attention_mask) * -1e9        

        for layer_module in self.cnn_attenion_modules:
            residue_embeddings = layer_module(query_embeddings, residue_embeddings, prots_extended_attention_mask)

        outputs = self.fc(torch.cat([residue_embeddings, protein_embeddings], dim = 2)).squeeze(-1)
        
        return outputs
        