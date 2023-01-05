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

class Output(nn.Module):
    def __init__(self, config):
        super(Output, self).__init__()
        self.dense = nn.Linear(config["architectures"]["intermediate_size"], config["architectures"]["hidden_size"]) # (Intermediate_size, Hidden_size)
        self.LayerNorm = LayerNorm(config["architectures"]["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config["train"]["dropout"])
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states) # (Batch, Max_lengths, Hidden_size)
        hidden_states = self.dropout(hidden_states) # (Batch, Max_lengths, Hidden_size) 
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # (Batch, Max_lengths, Hidden_size)
               
        return hidden_states 

class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(config["architectures"]["hidden_size"], config["architectures"]["intermediate_size"]) # (Hidden_size, Intermediate_size)
        self.intermediate_act_fn = ACT2FN[config["architectures"]["hidden_act"]]
        self.dropout = nn.Dropout(config["train"]["dropout"])
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states) # (Batch, Max_lengths, Intermediate_size)
        hidden_states = self.intermediate_act_fn(hidden_states) # (Batch, Max_lengths, Intermediate_size)
        hidden_states = self.dropout(hidden_states) # (Batch, Max_lengths, Intermediate_size) 
        return hidden_states

class AttLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = Attention(config)
        self.output = AttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):

        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output
        
class Attention(nn.Module):
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
        
        self.query = nn.Linear(self.config["architectures"]["hidden_size"], self.all_head_size) # (Hidden_size, all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size) # (Hidden_size, all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size) # (Hidden_size, all_head_size)
        
        self.dropout = nn.Dropout(self.config["train"]["dropout"])
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.config["architectures"]["num_attention_heads"], self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):

        mixed_query_layer = self.query(hidden_states) # (Batch, Max_lengths, Hidden_size)
        mixed_key_layer = self.key(context) # (Batch, Max_lengths, Hidden_size)
        mixed_value_layer = self.value(context) # (Batch, Max_lengths, Hidden_size)

        query_layer = self.transpose_for_scores(mixed_query_layer) # (Batch, Num_heads, Max_lengths, Head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer) # (Batch, Num_heads, Max_lengths, Head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer) # (Batch, Num_heads, Max_lengths, Head_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # key_layer.transpose(-1, -2): (Batch, Num_heads, Head_size, Max_lengths)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # (Batch, Num_heads, Max_lengths, Max_lengths)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # (Batch, Num_heads, Max_lengths, Max_lengths)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask # attention_scores: (Batch, Num_heads, Max_lengths, Max_lengths), attention_mask: (Batch, Max_lengths)
    
        # Normalize the attention scores to probabilities.
        #attention_probs = nn.Softmax(dim=-1)(attention_scores) # (Batch, Num_heads, Max_lengths, Max_lengths)
        attention_probs = F.log_softmax(attention_scores, dim=0)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs) # ?? (Batch, Num_heads, Hidden_size, Hidden_size) /// (Batch, Num_heads, Max_lengths, Max_lengths)

        context_layer = torch.matmul(attention_probs, value_layer) # (Batch, Num_heads, Max_lengths, Hidden_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (Batch, Max_lengths, Num_heads, Hidden_size)

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) 

        context_layer = context_layer.view(*new_context_layer_shape) # (Batch, Max_lengths, Hidden_size)
        
        return context_layer

class AttOutput(nn.Module):
    def __init__(self, config):
        super(AttOutput, self).__init__()
        self.dense = nn.Linear(config["architectures"]["hidden_size"], config["architectures"]["hidden_size"]) # (Hidden_size, Hidden_size)
        self.LayerNorm = LayerNorm(config["architectures"]["hidden_size"], eps=1e-12) 
        self.dropout = nn.Dropout(config["train"]["dropout"]) 
        self.transform_act_fn = ACT2FN[config["architectures"]["hidden_act"]]

    def forward(self, hidden_states, input_tensor):
        
        hidden_states = self.dense(hidden_states) # 
        hidden_states = self.dropout(hidden_states) #
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # 
        hidden_states = self.transform_act_fn(hidden_states) #        
        
        return hidden_states