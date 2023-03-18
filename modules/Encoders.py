import torch 
import torch.nn as nn
from .BertModeling import BertEmbeddings, BertLayer, BertCrossattLayer, BertSelfattLayer, BertIntermediate, BertOutput 

import math
from copy import deepcopy

BertLayerNorm = torch.nn.LayerNorm

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
        
class PocketEncoderLayer(nn.Module):
    def __init__(self, config, dim_in, dim_out, kernel_size, padding, dilation, stride, dropout = None):
        super(PocketEncoderLayer, self).__init__()
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

class Pseq2Sites(nn.Module):
    def __init__(self, config):
        super(Pseq2Sites, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(self.config["train"]["dropout"])

        # Query Embedding
        self.position_embeddings = nn.Embedding(config["prots"]["max_lengths"], 1024)
        self.token_type_embeddings = nn.Embedding(13, 1024, padding_idx=0)
        self.LayerNorm = BertLayerNorm(1024, eps=1e-12)

        self.protein_position_embeddings = nn.Embedding(config["prots"]["max_lengths"], 256)
        self.protein_token_type_embeddings = nn.Embedding(13, 256, padding_idx=0)

        self.LayerNorm2 = BertLayerNorm(1024, eps=1e-12)
        self.LayerNorm3 = BertLayerNorm(256, eps=1e-12)
        
        self.fc2 = nn.Sequential(
                        GeLU(),
                        BertLayerNorm(512, eps=1e-12),
                        self.dropout,
                        nn.Linear(512, 128),
                        GeLU(),
                        BertLayerNorm(128, eps=1e-12),
                        self.dropout,
                        nn.Linear(128, 64),
                        GeLU(),
                        BertLayerNorm(64, eps=1e-12),
                        self.dropout,
                        nn.Linear(64, 1)  
                    )         
                    
        self.cross_att_layers = nn.ModuleList()
        
        self.cross_att_layers.append(PocketConvLayer(self.config, 1024, 256, 1, 1))
        self.cross_att_layers.append(PocketATTLayer(self.config))

        self.protein_features = nn.Linear(1024, self.config["architectures"]["hidden_size"])
        
    def forward(self, prots, total_prots_data, attention_mask, position_ids, token_type_ids):
        
        feats = deepcopy(prots)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        protein_position_embeddings = self.protein_position_embeddings(position_ids)
        protein_token_type_embeddings = self.protein_token_type_embeddings(token_type_ids)
        
        # input features
        feats = feats + position_embeddings + token_type_embeddings
        feats = self.LayerNorm(feats)
        feats = self.dropout(feats)
        
        # protein features
        prot_feats = self.LayerNorm2(total_prots_data)
        
        prot_feats = self.protein_features(prot_feats)
        prot_feats = self.dropout(prot_feats)

        # query features
        query_embeddings = prot_feats + protein_position_embeddings + protein_token_type_embeddings
        query_embeddings = self.LayerNorm3(query_embeddings)
        query_embeddings = self.dropout(query_embeddings)

        prots_extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        prots_extended_attention_mask = prots_extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        prots_extended_attention_mask = (1.0 - prots_extended_attention_mask) * -1e9
        
        for layer_module in self.cross_att_layers:
            feats, att_probs = layer_module(query_embeddings, feats, prots_extended_attention_mask)

        outputs = self.fc2(torch.cat([feats, prot_feats], dim = 2)).squeeze(-1)
        
        return feats, outputs, att_probs

class PocketATTLayer(nn.Module):
    def __init__(self, config):
        super(PocketATTLayer, self).__init__()
        
        self.config = config
                
        # cross-attention layer
        self.cross_attention = BertCrossattLayer(self.config)
        
        # Intermediate and output layers (FFNs)
        self.inter = BertIntermediate(self.config)
        self.output = BertOutput(self.config)       
    
    def forward(self, embeddings, prots, attention_mask):

        prots_cross_output, prot_att_probs = self.cross_attention(embeddings, prots, attention_mask)

        prots_inter_outputs = self.inter(prots)
        prots_outputs = self.output(prots_inter_outputs, prots_cross_output)
        
        return prots_outputs, prot_att_probs    

class PocketConvLayer(nn.Module):
    def __init__(self, config, dim_in, dim_out, padding = 1, stride = 1):
        super(PocketConvLayer, self).__init__()
        
        self.config = config

        self.first_layers = nn.ModuleList()

        self.first_layers.append(PocketEncoderLayer(self.config, dim_in = dim_in, dim_out = dim_out, 
            kernel_size = 3, padding = padding, dilation = 1, stride = stride))
            
    def forward(self, embeddings = None, prots = None, attention_mask = None):
        first_feats = prots.transpose(1,2)
        
        for layer_module in self.first_layers:
            first_feats = layer_module(first_feats)

        feats = first_feats
        
        return feats.transpose(1,2), None