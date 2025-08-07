import numpy as np 
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from flash_attn.flash_attention import FlashMHA
from einops import rearrange
from flash_attn.flash_attention import FlashAttention
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

# Custom Flash Attention optimized to work with Tesla GPU 
class CustomFlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CustomFlashAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.softmax_scale = embed_dim ** -0.5  # Scaling factor for softmax

    def forward(self, q, k, v, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention with separate query, key, and value tensors.
        Arguments
        ---------
            q: Tensor containing the query. Shape (B, S_q, H, D)
            k: Tensor containing the key. Shape (B, S_k, H, D)
            v: Tensor containing the value. Shape (B, S_v, H, D)
            key_padding_mask: a bool tensor of shape (B, S_k)
            causal: Boolean indicating if causal attention should be used.
            cu_seqlens: Optional for efficient attention on padded sequences.
            max_s: Optional for maximum sequence length.
            need_weights: Boolean indicating if attention weights should be returned.
        """
        #assert not need_weights
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda

        batch_size = q.shape[0]
        seqlen_q = q.shape[1]
        seqlen_k = k.shape[1]

        if cu_seqlens is None:
            if key_padding_mask is None:
                # Reshape q, k, v to the required shape for flash attention
                q = rearrange(q, 'b s h d -> (b s) h d')   # Shape: (batch_size * seqlen_q, num_heads, head_dim)
                k = rearrange(k, 'b s h d -> (b s) h d')   # Shape: (batch_size * seqlen_k, num_heads, head_dim)
                v = rearrange(v, 'b s h d -> (b s) h d')   # Shape: (batch_size * seqlen_k, num_heads, head_dim)

                # Concatenate q, k, v into qkv
                # Use stack instead of cat to include the three components
                qkv = torch.stack([q, k, v], dim=1)  # Shape: (batch_size * seq_len_q, 3, num_heads, head_dim)

                # Reshape for flash attention (total, 3, nheads, headdim)
                qkv = qkv.reshape(-1, 3, qkv.size(2), qkv.size(3))  # Shape: (total, 3, num_heads, head_dim)
                
                max_s = seqlen_k
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                        device=q.device)

                # Call the unpadded attention function
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = output.view(batch_size, seqlen_k, -1)  # Reshape to (batch_size, seq_len, hidden_dim)
                if output.dim() != 3:
                    output = rearrange(output, 'b s h d -> b s (h d)'), None
            else:
                # Handle the padded case
                nheads = q.shape[-2]
                # Unpad inputs
                q_unpad, indices_q, cu_seqlens, max_s = unpad_input(q, key_padding_mask)
                k_unpad, _, _, _ = unpad_input(k, key_padding_mask)
                v_unpad, _, _, _ = unpad_input(v, key_padding_mask)

                # Compute sequence lengths based on unpadded data (assuming max_s is computed correctly)
                seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]

                # Concatenate unpadded q, k, v for flash attention
                qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)  # Shape: (nnz, 3, h, d)

                # Correct cu_seqlens to have shape (batch_size + 1)
                cu_seqlens = torch.cat([torch.tensor([0], device=q.device), torch.cumsum(seq_lens, dim=0)])

                # Call the unpadded attention function
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    qkv_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )

                # Pad the output back to the original shape
                output = pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                indices_q, batch_size, seqlen_q)

                # Rearrange the output to match the expected shape
                output = rearrange(output, 'b s (h d) -> b s (h d)', h=nheads)

        else:
            # Handle the padded case
            nheads = q.shape[-2]
            q_unpad, indices_q, cu_seqlens, max_s = unpad_input(q, key_padding_mask)
            k_unpad, _, _, _ = unpad_input(k, key_padding_mask)
            v_unpad, _, _, _ = unpad_input(v, key_padding_mask)

            # Ensure max_s is computed (should be the maximum sequence length)
            assert max_s is not None, "max_s must not be None. Ensure unpad_input returns it."
            
            # Ensure cu_seqlens is correctly shaped
            assert cu_seqlens is not None, "cu_seqlens must not be None."

            # Concatenate unpadded q, k, v for flash attention
            qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)  # Shape: (nnz, 3, h, d)

            # Call the unpadded attention function
            output_unpad = flash_attn_unpadded_qkvpacked_func(
                qkv_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

            # Pad the output back to the original shape
            output = pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                            indices_q, batch_size, seqlen_q)

            # Rearrange the output to match the expected shape
            output = rearrange(output, 'b s (h d) -> b s (h d)', h=nheads)
        # Compute attention weights if needed
        if need_weights:
            attn_scores = torch.einsum('bqhd,bkhd->bhqk', q.view(batch_size, seqlen_q, self.num_heads, self.embed_dim//self.num_heads),
                                        k.view(batch_size, seqlen_k, self.num_heads, self.embed_dim//self.num_heads)) * self.softmax_scale
            attn_weights = torch.softmax(attn_scores, dim=-1)  # Shape: (batch, num_heads, seq_len_q, seq_len_k)
        else:
            attn_weights = None
        return output, attn_weights
    
# %% 
class CustomFlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        # Separate projection layers for query and key-value
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)

        self.inner_attn = CustomFlashAttention(embed_dim, num_heads, dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, query, key_value, key_padding_mask=None, need_weights=False, cu_seqlens=None):
        """query: (batch, seqlen_q, hidden_dim) - the query input (decoder)
           key_value: (batch, seqlen_kv, hidden_dim) - the key-value input (encoder)
           key_padding_mask: bool tensor of shape (batch, seqlen_kv)
        """
        # Project query and key-value
        q = self.Wq(query)
        kv = self.Wkv(key_value)
        # Reshape for multi-head attention
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = rearrange(kv, 'b s (two h d) -> b s two h d', two=2, h=self.num_heads)
        # Split key_value into key and value
        k, v = kv.unbind(dim=2)  # Separate into key and value, k: (batch, seq_len, n_heads, head_dim), v: same

        # Perform the cross-attention with separate query, key, and value
        context, attn_weights = self.inner_attn(q, k, v, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal, cu_seqlens=cu_seqlens)
        
        # Apply output projection
        return self.out_proj(context), attn_weights

class FlashAttentionWeights(nn.Module):
   """Scaled dot product attention with optional attention weight output."""
   def __init__(self, softmax_scale=None, attention_dropout=0.0):
       super().__init__()
       self.softmax_scale = softmax_scale
       self.dropout_p = attention_dropout
   def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
               max_s=None, need_weights=False):
       """
       Arguments:
           qkv: (B, S, 3, H, D) if key_padding_mask is None
           key_padding_mask: bool tensor of shape (B, S)
           need_weights: If True, return attention weights.
       Returns:
           output: (B, S, H, D)
           attn_weights: (B, H, S, S) if need_weights=True, else None
       """
       assert qkv.dtype in [torch.float16, torch.bfloat16]
       assert qkv.is_cuda
       if cu_seqlens is None:
           batch_size, seqlen, _, num_heads, head_dim = qkv.shape
           qkv = rearrange(qkv, 'b s ... -> (b s) ...')
           max_s = seqlen
           cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=qkv.device)
           output = flash_attn_unpadded_qkvpacked_func(
               qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
               softmax_scale=self.softmax_scale, causal=causal
           )
           output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
           if need_weights:
               q, k, _ = qkv.view(batch_size, seqlen, 3, num_heads, head_dim).unbind(dim=2)  # Separate Q, K, V
               attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k)  # (batch, num_heads, seq_len, seq_len)
               attn_weights = torch.softmax(attn_scores * (self.softmax_scale or 1.0 / head_dim ** 0.5), dim=-1)
           else:
               attn_weights = None
       else:
           assert max_s is not None
           output = flash_attn_unpadded_qkvpacked_func(
               qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
               softmax_scale=self.softmax_scale, causal=causal
           )
           attn_weights = None  # Not computed for packed sequences
       return output, attn_weights  

class FlashMHAWeights(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttentionWeights(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        context, attn_weights = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights

# %% 
# Custom transformer decoder layer that uses flash attention (should be faster)
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(CustomTransformerDecoderLayer, self).__init__()
        # Flash attention for self-attention and encoder-decoder attention
        self.self_attn = FlashMHAWeights(embed_dim=hidden_dim, num_heads=num_heads, causal=True)
        self.multihead_attn = CustomFlashMHA(embed_dim=hidden_dim, num_heads=num_heads)
        
        # Feed-forward network
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        
        # Dropout and normalization layers
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cu_seqlens=None):
        # Self-attention with residual connection and layer normalization
        tgt2, self_attention_weights = self.self_attn(tgt, key_padding_mask=tgt_mask, need_weights=True)  # Apply self-attention to tgt
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with encoder output (memory), residual connection, and layer normalization
        tgt2, cross_attention_weights = self.multihead_attn(tgt, memory, key_padding_mask=memory_mask, need_weights=True, cu_seqlens=cu_seqlens)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward network with residual connection and layer normalization
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt, {"self": self_attention_weights, "cross": cross_attention_weights}
# %%    
class CustomTransformerDecoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, dropout=0.1):
        super(CustomTransformerDecoder, self).__init__()
        # Stack of CustomTransformerDecoderLayers
        self.layers = nn.ModuleList([CustomTransformerDecoderLayer(hidden_dim, num_heads, dropout)
                                     for _ in range(num_layers)])
        # Final normalization layer
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cu_seqlens=None):
        """
        Args:
            tgt (Tensor): Target sequence (batch_size, target_seq_len, hidden_dim)
            memory (Tensor): Encoder output (batch_size, source_seq_len, hidden_dim)
            tgt_mask (Tensor, optional): Mask for target sequence (batch_size, target_seq_len, target_seq_len)
            memory_mask (Tensor, optional): Mask for encoder memory (batch_size, target_seq_len, source_seq_len)
        
        Returns:
            Tensor: Output of the decoder (batch_size, target_seq_len, hidden_dim)
        """
        output = tgt
        self_attentions = [] 
        cross_attentions = []
        
        # Pass through all decoder layers
        for layer in self.layers:
            output, attn = layer(output, memory, tgt_mask, memory_mask, cu_seqlens)
            self_attentions.append(attn["self"])
            cross_attentions.append(attn["cross"])
        # Apply final normalization
        output = self.norm(output)
        
        return output, {"self": self_attentions, "cross": cross_attentions}

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = FlashMHAWeights(embed_dim=hidden_dim, num_heads=num_heads, causal=True)

        # Feed-forward network
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

        # Dropout and normalization layers
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, src, src_mask=None):
        # Self-attention with residual connection and layer normalization
        src2, attention_weights = self.self_attn(src, key_padding_mask=src_mask, need_weights = True)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # Feed-forward network with residual connection and layer normalization
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src, attention_weights
# %% 
# Custom Transformer Encoder
class CustomTransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, dropout=0.1):
        super(CustomTransformerEncoder, self).__init__()
        # Stack of CustomTransformerEncoderLayers
        self.layers = nn.ModuleList([CustomTransformerEncoderLayer(hidden_dim, num_heads, dropout)
                                     for _ in range(num_layers)])
        # Final normalization layer
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, src, src_mask=None):
        output = src
        attention_weights_all_layers = []
        # Pass through all encoder layers
        for layer in self.layers:
            output, attention_weights = layer(output, src_mask)
            attention_weights_all_layers.append(attention_weights)
        
        # Apply final normalization
        output = self.norm(output)
        
        return output, attention_weights_all_layers

# %% 
class TransformerEncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads=2, dropout=0.5, contextual_dim=None, lstm_hidden_dim=None, lstm_num_layers=1, conv_out_channels=None, conv_kernel_size=3, conv_stride=1):
        super(TransformerEncoderDecoder, self).__init__()
        self.num_heads = num_heads 
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
         # Optional Convolutional Layer
        if conv_out_channels is not None:
            self.conv_layer = nn.Conv1d(input_dim, conv_out_channels, kernel_size=conv_kernel_size, 
                                        stride=conv_stride, padding=(conv_kernel_size - 1)//2)
            self.input_embedding = nn.Sequential(nn.Linear(conv_out_channels, hidden_dim), 
                                                 nn.LayerNorm(hidden_dim))  # Adjust input dim based on conv layer output
            self.lstm = None
        elif lstm_hidden_dim is not None:
            self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=lstm_num_layers, batch_first=True)
            self.input_embedding = nn.Sequential(nn.Linear(lstm_hidden_dim, hidden_dim), 
                                                 nn.LayerNorm(hidden_dim))  # Embedding after LSTM output
        else:
            self.conv_layer = None       
            self.input_embedding = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                                 nn.LayerNorm(hidden_dim)) 
            self.lstm = None

        self.positional_encoding = PositionalEncoding(hidden_dim, 1024, dropout)

        # Initialize Encoder
        self.encoder_layer = CustomTransformerEncoderLayer(hidden_dim=hidden_dim, num_heads=self.num_heads, dropout=dropout)
        self.transformer_encoder = CustomTransformerEncoder(num_layers=num_layers, hidden_dim=hidden_dim, num_heads=self.num_heads, dropout=dropout)
        
        # Initialize custom transformer decoder layers
        if contextual_dim: 
            self.decoder_layer = CustomTransformerDecoderLayer(hidden_dim=hidden_dim*2, num_heads=self.num_heads, dropout=dropout)
            self.transformer_decoder = CustomTransformerDecoder(num_layers=num_layers-1, hidden_dim=hidden_dim*2, num_heads=self.num_heads, dropout=dropout)
            self.contextual_embedding = nn.Linear(contextual_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else: 
            self.decoder_layer = CustomTransformerDecoderLayer(hidden_dim=hidden_dim, num_heads=self.num_heads, dropout=dropout)
            self.transformer_decoder = CustomTransformerDecoder(num_layers=num_layers-1, hidden_dim=hidden_dim, num_heads=self.num_heads, dropout=dropout)
            self.contextual_embedding = None
            self.fc = nn.Linear(hidden_dim, output_dim)

    def freeze_layers(self):
        # Freeze LSTM layers (if present)
        if self.lstm is not None:
            for param in self.lstm.parameters():
                param.requires_grad = False

        # Freeze input embedding layer
        for param in self.input_embedding.parameters():
            param.requires_grad = False

        # Freeze positional encoding layer (if it has parameters)
        if hasattr(self.positional_encoding, 'parameters'):
            for param in self.positional_encoding.parameters():
                param.requires_grad = False

        # Freeze encoder layers
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
            
    def forward(self, src, lengths=None):
        # src shape: (batch_size, seq_len, input_dim)
        
        # If LSTM is used, pass through LSTM first
        if self.lstm is not None:
            lstm_out, _ = self.lstm(src)  # lstm_out shape: (batch_size, seq_len, lstm_hidden_dim)
            x = self.input_embedding(lstm_out)  # Apply embedding after LSTM
        elif self.conv_layer is not None: 
            # Rearrange dimensions for Conv1d: (batch_size, input_dim, seq_len)
            src_perm = src.permute(0, 2, 1)
            conv_out = self.conv_layer(src_perm)  # Apply convolution
            # Rearrange back: (batch_size, seq_len, conv_out_channels)
            conv_out = conv_out.permute(0, 2, 1)
            x = self.input_embedding(conv_out)  # Apply embedding after Conv1d            
        else:
            x = self.input_embedding(src)  # Apply embedding directly
        
        # Apply positional encoding
        x = self.positional_encoding(x)

        # Pass through the transformer encoder
        memory, encoder_attention = self.transformer_encoder(x)

        # Apply contextual embedding (concatenate memory with src)
        if self.contextual_embedding is not None:
            contextual_embedded = self.contextual_embedding(src)
            memory = torch.cat((memory, contextual_embedded), dim=2)

        # If lengths are provided, create masks and cumulative lengths
        if lengths is not None:
            batch_size, seq_len = x.size(0), x.size(1)
            key_padding_mask = torch.ones(batch_size, seq_len, device=x.device)
            for i in range(batch_size):
                key_padding_mask[i, lengths[i]:] = 0  # Set positions beyond lengths to 1
            
            key_padding_mask = key_padding_mask.bool()  # Convert to boolean
            cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device)
            cu_seqlens[1:] = lengths.cumsum(dim=0)

            # Call the transformer decoder with the mask and lengths
            x, decoder_attention = self.transformer_decoder(memory, memory, tgt_mask=key_padding_mask, memory_mask=key_padding_mask, cu_seqlens=cu_seqlens)
        else:
            # If lengths are not provided, just use memory directly
            x, decoder_attention = self.transformer_decoder(memory, memory)

        # Apply the final fully connected layer
        x = self.fc(x)  # (batch_size, seq_len, output_dim)

        return x, {"enc": encoder_attention, "dec_self": decoder_attention["self"], "dec_cross": decoder_attention["cross"]} 

