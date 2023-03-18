import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config["architectures"]["intermediate_size"], config["architectures"]["hidden_size"])
        self.LayerNorm = LayerNorm(config["architectures"]["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config["train"]["dropout"])
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
               
        return hidden_states 

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config["architectures"]["hidden_size"], config["architectures"]["intermediate_size"])
        self.intermediate_act_fn = ACT2FN[config["architectures"]["hidden_act"]]
        self.dropout = nn.Dropout(config["train"]["dropout"])
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = Attention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):

        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output
        
class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.config = config
        if self.config["architectures"]["hidden_size"] % self.config["architectures"]["num_attention_heads"] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.config["architectures"]["hidden_size"], self.config["architectures"]["num_attention_heads"]))
        self.num_attention_heads = self.config["architectures"]["num_attention_heads"] 
        self.attention_head_size = int(self.config["architectures"]["hidden_size"] / self.config["architectures"]["num_attention_heads"])
        self.all_head_size = self.config["architectures"]["num_attention_heads"] * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = self.config["architectures"]["hidden_size"]
        
        self.query = nn.Linear(self.config["architectures"]["hidden_size"], self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)
        
        self.dropout = nn.Dropout(self.config["train"]["dropout"])
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.config["architectures"]["num_attention_heads"], self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # key_layer.transpose(-1, -2): (Batch, Num_heads, Head_size, Max_lengths)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
    
        # Normalize the attention scores to probabilities.
        #attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = F.log_softmax(attention_scores, dim=0)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) 

        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs 

class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config["architectures"]["hidden_size"], config["architectures"]["hidden_size"])
        self.LayerNorm = LayerNorm(config["architectures"]["hidden_size"], eps=1e-12) 
        self.dropout = nn.Dropout(config["train"]["dropout"]) 
        self.transform_act_fn = ACT2FN[config["architectures"]["hidden_act"]]

    def forward(self, hidden_states, input_tensor):
        
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) 
        hidden_states = self.transform_act_fn(hidden_states)     
        
        return hidden_states
        
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config["comps"]["vocab_size"], config["architectures"]["hidden_size"], padding_idx=0) # (Vocab_size, Hidden_size)
        self.position_embeddings = nn.Embedding(config["comps"]["max_lengths"], config["architectures"]["hidden_size"], padding_idx=0) # (Max_position, Hidden_size)

        self.LayerNorm = BertLayerNorm(config["architectures"]["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config["train"]["dropout"])

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings 

class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):

        output, att_probs = self.att(input_tensor, ctx_tensor, ctx_att_mask)
     
        attention_output = self.output(output, input_tensor)

        return attention_output, att_probs

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config) 
        self.intermediate = BertIntermediate(config) 
        self.output = BertOutput(config) 
    
    def forward(self, hidden_states, attention_mask):

        attention_output = self.attention(hidden_states, attention_mask) # (Batch, Max_lengths, Hidden_size)
        intermediate_output = self.intermediate(attention_output) # (Batch, Max_lengths, Intermediate_size)
        layer_output = self.output(intermediate_output, attention_output) # (Batch, Max_lengths, Hidden_size)

        return layer_output        