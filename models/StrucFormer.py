# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:17:32 2024

@author: CUPK-K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_proj(input_size, dim, bias=False):

    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin




class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class FeedForward(nn.Module):
    """
    Standard Feed Forward Layer
    """
    def __init__(self, input_channels, output_channels, ff_dim, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(input_channels, ff_dim)
        self.w_2 = nn.Linear(ff_dim, output_channels)
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tensor, **kwargs):
        tensor = self.w_1(tensor)
        tensor = self.activation(tensor)
        tensor = self.dropout(tensor)
        tensor = self.w_2(tensor)
        tensor = self.dropout2(tensor)
        return tensor




class CAttentionHead(nn.Module):
    """
    Constrained Attention Head 
    """
    def __init__(self, input_size, dim, dim_k, dropout):
        super(CAttentionHead, self).__init__()
        self.E =   get_proj(input_size,dim_k)
        self.F =   get_proj(input_size,dim_k)
        self.dim = dim
        self.dropout = nn.Dropout(dropout)




    def forward(self, Q, K, V, **kwargs):
        
        
        K = K.transpose(1,2)
        K = self.E(K)
        Q = torch.matmul(Q, K)

        P_bar = Q/torch.sqrt(torch.tensor(self.dim).type(Q.type())).to(Q.device)
        P_bar = P_bar.softmax(dim=-1)

        P_bar = self.dropout(P_bar)

        V = V.transpose(1,2)
        V = self.F(V)
        V = V.transpose(1,2)
        out_tensor = torch.matmul(P_bar, V)

        return out_tensor









class CMHAttention(nn.Module):
    """
    Constrained Multihead Attention
    """
    def __init__(self, input_size, head_dim, input_channels, dim_k, nhead, dropout, mode):
        super(CMHAttention, self).__init__()
        
        self.input_size = input_size
        self.input_channels = input_channels
        self.dim_k = dim_k
        self.head_dim   = head_dim
        self.nhead      = nhead
        self.mh_dropout = nn.Dropout(dropout)
        self.mode       = mode
        
        self.heads = nn.ModuleList()
        self.to_q = nn.ModuleList()
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()
    
        for _ in range(self.nhead):
            cattn = CAttentionHead(self.input_size, self.head_dim, self.dim_k, dropout)
            self.heads.append(cattn)
            self.to_q.append(nn.Linear(self.input_channels, self.head_dim, bias=False))
            self.to_k.append(nn.Linear(self.input_channels, self.head_dim, bias=False))
            self.to_v.append(nn.Linear(self.input_channels, self.head_dim, bias=False))
            self.w_o = nn.Linear(self.head_dim*self.nhead, self.input_channels)
        

    def forward(self, tensor, **kwargs):
        batch_size, input_len, channels = tensor.shape

        head_outputs = []
        for index, head in enumerate(self.heads):
            Q = self.to_q[index](tensor)
            K = self.to_k[index](tensor) if self.mode == 'encoder' else self.to_k[index](kwargs["embeddings"])
            V = self.to_v[index](tensor) if self.mode == 'encoder' else self.to_k[index](kwargs["embeddings"])
            head_outputs.append(head(Q,K,V))

        out = torch.cat(head_outputs, dim=-1)
        out = self.w_o(out)
        out = self.mh_dropout(out)
        return out





class Encoder(nn.Module):

    
    def __init__(self, input_size, channels, dim_k, dim_ff=128, dropout_ff=0.15, dropout=0.10, nhead=6, depth=3):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.channels = channels
        self.depth = depth
        self.nhead = nhead
        self.dim_k = dim_k

        for index in range(depth):
            input_channels  = self.channels
            output_channels = self.channels
            head_dim        = self.channels
            
            attn_layer = CMHAttention(self.input_size, head_dim, input_channels, self.dim_k, self.nhead, dropout, 'encoder')
            ff_layer   = FeedForward(input_channels, output_channels, dim_ff, dropout_ff)
            
            self.layers.append(nn.ModuleList([
                PreNorm(output_channels, attn_layer),
                PreNorm(output_channels, ff_layer)]))        
        
        
    def forward(self, tensor, **kwargs):
        for attn, ff in self.layers:
            tensor = attn(tensor) + tensor
            tensor = ff(tensor) + tensor
        return tensor



class Decoder(nn.Module):

    
    def __init__(self, input_size, embedding_size, channels, dim_k, dim_ff=128, dropout_ff=0.15, dropout=0.10, nhead=6, depth=3):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.channels = channels
        self.depth = depth
        self.nhead = nhead
        self.dim_k = dim_k
        
        self.project_out = nn.Linear(channels, 1);
        


        for index in range(depth):
            input_channels  = self.channels
            output_channels = self.channels
            head_dim        = self.channels
            
            attn_layer = CMHAttention(self.input_size, head_dim, input_channels, self.dim_k, self.nhead, dropout, 'encoder')
            ff_layer   = FeedForward(input_channels, output_channels, dim_ff, dropout_ff)
            
            self.layers.append(nn.ModuleList([
                PreNorm(output_channels, attn_layer),
                PreNorm(output_channels, ff_layer)]))        
            
            attn_context_layer = CMHAttention(self.embedding_size, head_dim, input_channels, self.dim_k, self.nhead, dropout, 'decoder')
            ff_context_layer   = FeedForward(input_channels, output_channels, dim_ff, dropout_ff)
            self.layers.append(nn.ModuleList([
                PreNorm(output_channels, attn_context_layer),
                PreNorm(output_channels, ff_context_layer)]))
        
        
    def forward(self, tensor, **kwargs):

        
        for attn, ff in self.layers:
            tensor = attn(tensor, **kwargs) + tensor
            tensor = ff(tensor) + tensor
            
        return self.project_out(tensor)









class StrucFormer(nn.Module):

    #input_size: sensed value size
    #decode_size: unsensed value size    

    def __init__(self, input_size, decode_size, channels, dim_k, dim_ff=128, dropout_ff=0.15, dropout=0.1, nhead=6, depth=3):
        super(StrucFormer, self).__init__()


        self.Encoder = Encoder(input_size, channels, dim_k, dim_ff, dropout_ff, dropout, nhead, depth);
        self.Decoder = Decoder(decode_size,input_size,channels, dim_k, dim_ff, dropout_ff, dropout, nhead, depth);
        

            


    def forward(self, x, y=None, **kwargs):
        encoder_output = self.Encoder(x);
        return self.Decoder(y, embeddings=encoder_output);


































